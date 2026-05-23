"""A1 - Relation Ambiguity: question uses a term that could refer to 2+ relation types."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_relation_types
from cb_corruptions.graph_analysis import (
    RelationAmbiguityCandidate,
    find_relation_ambiguity_candidates,
)
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    HypernymListResponse,
    RelationSubgroupResponse,
    RewrittenQuestionResponse,
)

logger = logging.getLogger(__name__)


@register_corruption("A1")
class RelationAmbiguityCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        # Pre-generated hypernyms per subgroup, populated during analyze()
        self._hypernyms: dict[str, list[str]] = {}
        # Round-robin index per subgroup
        self._hypernym_idx: dict[str, int] = defaultdict(int)

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list[RelationAmbiguityCandidate]:
        full_groups = find_relation_ambiguity_candidates(schema)

        if self.llm is None:
            return full_groups

        expanded: list[RelationAmbiguityCandidate] = []
        for candidate in full_groups:
            if len(candidate.relations) <= 2:
                expanded.append(candidate)
                continue

            subgroups = self._find_semantic_subgroups(candidate)
            if subgroups:
                expanded.extend(subgroups)
            else:
                expanded.append(candidate)

        # Pre-generate all plausible hypernyms for each subgroup
        for candidate in expanded:
            self._pregenerate_hypernyms(candidate)

        return expanded

    def _find_semantic_subgroups(
        self,
        candidate: RelationAmbiguityCandidate,
    ) -> list[RelationAmbiguityCandidate]:
        """Ask the LLM to cluster relations into semantically coherent subgroups."""
        relation_names = [r.label for r in candidate.relations]

        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find natural semantic overlaps between knowledge-graph "
                        "relationships — cases where the same everyday word could "
                        "plausibly refer to multiple different relationships."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"These relationships all connect {candidate.subj_label} to "
                        f"{candidate.obj_label}:\n"
                        + "\n".join(f"- {name}" for name in relation_names)
                        + "\n\nFind all subgroups of 2 or more relationships that have "
                        "natural semantic overlap — where the same everyday word "
                        "could plausibly refer to any of them because they describe "
                        "similar concepts.\n"
                        "Rules:\n"
                        "- Only group relationships whose meanings genuinely overlap "
                        "in everyday language\n"
                        "- Do NOT group relationships with no shared meaning\n"
                        "- Include subgroups at different granularities when the "
                        "overlap exists at multiple levels\n"
                        "- Subgroups can overlap — a relationship may appear in "
                        "multiple subgroups\n"
                        "- Only include the full set if ALL relationships share "
                        "genuine semantic overlap"
                    ),
                },
            ],
            response_model=RelationSubgroupResponse,
        )

        rel_by_label = {r.label: r for r in candidate.relations}
        subgroup_candidates: list[RelationAmbiguityCandidate] = []

        for sg in resp.subgroups:
            rels = [rel_by_label[name] for name in sg.relations if name in rel_by_label]
            if len(rels) >= 2:
                subgroup_candidates.append(
                    RelationAmbiguityCandidate(
                        subj_label=candidate.subj_label,
                        obj_label=candidate.obj_label,
                        relations=rels,
                    )
                )

        logger.info(
            "  A1 subgroups for (%s)->(%s): %d relations -> %d subgroups",
            candidate.subj_label,
            candidate.obj_label,
            len(candidate.relations),
            len(subgroup_candidates),
        )
        return subgroup_candidates

    def _pregenerate_hypernyms(
        self,
        candidate: RelationAmbiguityCandidate,
    ) -> None:
        """Ask the LLM to list all plausible hypernyms for a subgroup."""
        relation_names = [r.label for r in candidate.relations]
        group_key = ",".join(sorted(relation_names))

        if group_key in self._hypernyms:
            return

        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find everyday words that naturally cover multiple "
                        "knowledge-graph relationships because they share "
                        "similar meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"These relationships connect {candidate.subj_label} to "
                        f"{candidate.obj_label}:\n"
                        + "\n".join(f"- {name}" for name in relation_names)
                        + "\n\nList everyday words or short phrases that a real "
                        "person would use in a question, where the word naturally "
                        "covers all of these relationships because they share "
                        "similar meaning.\n"
                        "Requirements:\n"
                        "- The word must be a natural synonym or near-synonym that "
                        "sits at the intersection of these relationships\n"
                        "- NOT a vague umbrella term — it must feel like a natural "
                        "way to refer to any of these specific relationships\n"
                        "- It must sound natural in a question\n"
                        "- Return only words that genuinely work — an empty list "
                        "is better than forced terms"
                    ),
                },
            ],
            response_model=HypernymListResponse,
        )

        if not resp.hypernyms:
            logger.warning("  A1: no valid hypernyms for [%s] — subgroup will be skipped", group_key)

        self._hypernyms[group_key] = resp.hypernyms
        logger.info(
            "  A1 hypernyms for [%s]: %s",
            group_key,
            resp.hypernyms,
        )

    def _next_hypernym(self, candidate: RelationAmbiguityCandidate) -> str | None:
        """Pick the next hypernym for this subgroup, cycling through the list."""
        relation_names = [r.label for r in candidate.relations]
        group_key = ",".join(sorted(relation_names))
        hypernyms = self._hypernyms.get(group_key, [])
        if not hypernyms:
            return None
        idx = self._hypernym_idx[group_key] % len(hypernyms)
        self._hypernym_idx[group_key] += 1
        return hypernyms[idx]

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list[RelationAmbiguityCandidate],
    ) -> list[tuple[Nl2CypherSample, RelationAmbiguityCandidate]]:
        """Match samples whose gold_cypher uses a relation from an ambiguous group."""
        rel_to_candidates: dict[str, list[RelationAmbiguityCandidate]] = defaultdict(list)
        for candidate in candidates:
            for rel in candidate.relations:
                rel_to_candidates[rel.label].append(candidate)

        pairs: list[tuple[Nl2CypherSample, RelationAmbiguityCandidate]] = []
        for sample in samples:
            used_rels = get_relation_types(sample.gold_cypher)
            seen: set[int] = set()
            for rel_label in used_rels:
                for candidate in rel_to_candidates.get(rel_label, []):
                    cand_id = id(candidate)
                    if cand_id not in seen:
                        pairs.append((sample, candidate))
                        seen.add(cand_id)

        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: RelationAmbiguityCandidate,
        graph_name: str,
    ) -> CorruptedSample | None:
        relation_names = [r.label for r in candidate.relations]

        # Step 1: Pick next hypernym
        hypernym = self._next_hypernym(candidate)
        if hypernym is None:
            return None

        # Step 2: Identify which relation the original query uses
        used_rels = get_relation_types(sample.gold_cypher)
        original_rel = next(
            (r for r in used_rels if r in {r.label for r in candidate.relations}),
            None,
        )
        if original_rel is None:
            return None

        # Step 3: Rewrite the question
        alternatives_str = "\n".join(
            f"- {rel.label}" for rel in candidate.relations
        )
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at crafting ambiguous questions for "
                        "a text-to-query benchmark. Your goal is to corrupt a "
                        "clear question into one that admits multiple exclusive "
                        "interpretations — each leading to a different, equally "
                        "valid answer. The corrupted question must force someone "
                        "to ask a follow-up clarification before they can answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {sample.nl_question}\n\n"
                        f"This question currently asks about the relationship "
                        f"'{original_rel}'. However, the following relationships "
                        f"are all plausible alternatives:\n{alternatives_str}\n\n"
                        f"Suggested ambiguous term: '{hypernym}'\n\n"
                        "Rewrite the question so that:\n"
                        "1. The rewritten question has MULTIPLE EXCLUSIVE "
                        "interpretations — each corresponding to one of the "
                        "alternative relationships above\n"
                        "2. Each interpretation would produce a DIFFERENT answer, "
                        "and a reader cannot determine which one is intended\n"
                        "3. Someone reading the question would need to ask back "
                        "'do you mean X or Y?' before being able to answer\n"
                        "4. The question sounds natural — like something a real "
                        "person would casually ask\n"
                        "5. Use singular form to force a single-choice reading\n"
                        "6. Keep entity names and the rest of the question intact\n\n"
                        "You may use the suggested term or find a better one. "
                        "The goal is ambiguity, not word substitution."
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        # Step 4: Generate valid Cypher for each relation in the group
        valid_cyphers = []
        for rel in candidate.relations:
            cypher_variant = sample.gold_cypher.replace(
                f":{original_rel}]", f":{rel.label}]"
            )
            valid_cyphers.append(cypher_variant)

        return CorruptedSample(
            corruption_id=f"A1-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.A1,
                corruption_category="ambiguity",
                original_element=original_rel,
                corrupted_element=hypernym,
                candidate_interpretations=relation_names,
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            valid_cyphers=valid_cyphers,
            expected_answer="AMBIGUOUS",
        )
