"""U2 - Missing Relation: question presupposes a relation not in the schema."""

from __future__ import annotations

import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_relation_types, is_in_optional_match, is_in_union_branch
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    FakeRelationResponse,
    RewrittenQuestionResponse,
)


@register_corruption("U2")
class MissingRelationCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self._used_fake_rels: set[str] = set()

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list:
        # No schema-level analysis needed — any sample with a relation is a candidate
        return []

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list,
    ) -> list[tuple[Nl2CypherSample, None]]:
        """Select any sample that uses at least one relation."""
        pairs = []
        for sample in samples:
            if get_relation_types(sample.gold_cypher):
                pairs.append((sample, None))
        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: object,
        graph_name: str,
    ) -> CorruptedSample | None:
        existing_rels = get_relation_types(sample.gold_cypher)

        # Pick a relation that is in a required position (not OPTIONAL MATCH or one UNION branch)
        original_rel = None
        for rel in existing_rels:
            if not is_in_optional_match(sample.gold_cypher, rel) and \
               not is_in_union_branch(sample.gold_cypher, rel):
                original_rel = rel
                break
        if original_rel is None:
            return None

        # Find the entity types the original relation connects
        original_subj = None
        original_obj = None
        for r in self.schema.relations:
            if r.label == original_rel:
                original_subj = r.subj_label
                original_obj = r.obj_label
                break

        if original_subj is None or original_obj is None:
            return None

        all_rel_labels = [r.label for r in self.schema.relations]

        # Step 1: Generate a fake relation between the SAME entity types
        avoid_clause = ""
        if self._used_fake_rels:
            avoid_clause = (
                f"\n\nDo NOT use any of these relation names (already used): "
                f"{', '.join(sorted(self._used_fake_rels))}. "
                "Generate a different relation name."
            )

        # Find ALL existing relations between these same entity types
        same_type_rels = [
            r.label for r in self.schema.relations
            if r.subj_label == original_subj and r.obj_label == original_obj
        ]

        fake_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": "You generate plausible but nonexistent relationship types for knowledge graphs.",
                },
                {
                    "role": "user",
                    "content": (
                        f"In the '{graph_name}' knowledge graph, the relationship '{original_rel}' "
                        f"connects {original_subj} to {original_obj}.\n\n"
                        f"All existing relationships between {original_subj} and {original_obj}:\n"
                        + "\n".join(f"- {r}" for r in same_type_rels)
                        + f"\n\nAll relationships in the graph:\n"
                        + "\n".join(f"- {r}" for r in all_rel_labels)
                        + f"\n\nGenerate a relationship name that:\n"
                        f"1. Could plausibly connect {original_subj} to {original_obj}\n"
                        f"2. Does NOT exist in the lists above\n"
                        f"3. Is SEMANTICALLY DISTINCT from all existing relationships — it must "
                        f"represent a genuinely different concept, NOT a synonym or paraphrase\n"
                        f"4. A question using this relationship CANNOT be answered using any "
                        f"existing relationship in the graph\n"
                        "Use camelCase naming."
                        + avoid_clause
                    ),
                },
            ],
            response_model=FakeRelationResponse,
        )

        if fake_resp.relation_name in all_rel_labels:
            return None

        self._used_fake_rels.add(fake_resp.relation_name)

        # Step 2: Minimally rewrite the question — same entities, different relation
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You minimally rewrite natural language questions to reference "
                        "a different relationship between the same entities. "
                        "Keep the question as close to the original as possible."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Minimally rewrite this question by replacing the concept of "
                        f"'{original_rel}' (connecting {original_subj} to {original_obj}) "
                        f"with the concept of '{fake_resp.relation_name}'.\n\n"
                        f"Original question: {sample.nl_question}\n\n"
                        "IMPORTANT: Keep the same entities, structure, and all other details. "
                        "Only change the words that describe the relationship. "
                        "The result must sound like a natural variation of the original."
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        return CorruptedSample(
            corruption_id=f"U2-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.U2,
                corruption_category="unanswerability",
                original_element=original_rel,
                corrupted_element=fake_resp.relation_name,
                reason_unanswerable=(
                    f"Relation '{fake_resp.relation_name}' does not exist in the schema. "
                    f"Existing relations: {', '.join(all_rel_labels)}"
                ),
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            expected_answer="UNANSWERABLE",
        )
