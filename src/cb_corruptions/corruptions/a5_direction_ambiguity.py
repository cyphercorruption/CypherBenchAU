"""A5 - Direction Ambiguity: question doesn't clarify relationship direction."""

from __future__ import annotations

import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_relation_types
from cb_corruptions.graph_analysis import (
    DirectionAmbiguityCandidate,
    find_direction_ambiguity_candidates,
)
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    RewrittenQuestionResponse,
)


def _reverse_arrows(cypher: str) -> str:
    """Swap all arrow directions in a Cypher query.

    -[r]->  becomes  <-[r]-    (forward to reverse)
    <-[r]-  becomes  -[r]->    (reverse to forward)
    """
    import re
    # Step 1: mark forward arrows -[...]-> with placeholder PREFIX_FWD
    result = re.sub(
        r'-(\[[^\]]+\])->',
        lambda m: f'FWD_L{m.group(1)}FWD_R',
        cypher,
    )
    # Step 2: mark reverse arrows <-[...]- with placeholder PREFIX_REV
    result = re.sub(
        r'<-(\[[^\]]+\])-',
        lambda m: f'REV_L{m.group(1)}REV_R',
        result,
    )
    # Step 3: resolve — forward becomes reverse, reverse becomes forward
    result = result.replace('FWD_L', '<-').replace('FWD_R', '-')
    result = result.replace('REV_L', '-').replace('REV_R', '->')
    return result


@register_corruption("A5")
class DirectionAmbiguityCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list[DirectionAmbiguityCandidate]:
        return find_direction_ambiguity_candidates(schema, graph_info)

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list[DirectionAmbiguityCandidate],
    ) -> list[tuple[Nl2CypherSample, DirectionAmbiguityCandidate]]:
        """Match samples using self-referential, non-symmetric relations."""
        rel_to_candidate = {c.relation.label: c for c in candidates}

        pairs = []
        for sample in samples:
            used_rels = get_relation_types(sample.gold_cypher)
            for rel in used_rels:
                if rel in rel_to_candidate:
                    pairs.append((sample, rel_to_candidate[rel]))
                    break

        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: DirectionAmbiguityCandidate,
        graph_name: str,
    ) -> CorruptedSample | None:
        rel_label = candidate.relation.label
        entity_label = candidate.entity_label

        # Rewrite with symmetric language so direction is unclear
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at crafting ambiguous questions for "
                        "a text-to-query benchmark. Your goal is to corrupt a "
                        "clear question into one where the direction of a "
                        "relationship is ambiguous — making two exclusive "
                        "interpretations equally valid."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The relationship '{rel_label}' connects {entity_label} "
                        f"to {entity_label} (self-referential). The direction "
                        f"matters: A-[:{rel_label}]->B is different from "
                        f"B-[:{rel_label}]->A.\n\n"
                        f"Original question: {sample.nl_question}\n\n"
                        "Rewrite the question so that:\n"
                        "1. A reader cannot tell which direction the "
                        "relationship goes — both directions are equally "
                        "plausible readings\n"
                        "2. Each direction would produce a DIFFERENT answer\n"
                        "3. Someone reading the question would need to ask "
                        "'do you mean A to B or B to A?' to clarify\n"
                        "4. The question sounds natural — like something a "
                        "real person would casually ask\n"
                        "5. Keep the question as close to the original as "
                        "possible — only change the words that indicate direction\n"
                        "6. Do NOT use technical graph language"
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        # Generate both direction variants by swapping all arrow directions
        reverse = _reverse_arrows(sample.gold_cypher)

        valid_cyphers = [sample.gold_cypher, reverse]

        return CorruptedSample(
            corruption_id=f"A5-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.A5,
                corruption_category="ambiguity",
                original_element=rel_label,
                corrupted_element=f"{rel_label} (direction ambiguous)",
                candidate_interpretations=[f"{rel_label} (forward)", f"{rel_label} (reverse)"],
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            valid_cyphers=valid_cyphers,
            expected_answer="AMBIGUOUS",
        )
