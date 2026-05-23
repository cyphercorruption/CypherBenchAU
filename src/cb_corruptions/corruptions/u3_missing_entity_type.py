"""U3 - Missing Entity Type: question refers to an entity type not in the graph."""

from __future__ import annotations

import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_entity_labels, is_in_optional_match, is_in_union_branch
from cb_corruptions.graph_analysis import get_all_entity_labels
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    FakeEntityTypeResponse,
    RewrittenQuestionResponse,
)


@register_corruption("U3")
class MissingEntityTypeCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self._used_fake_types: set[str] = set()

    def analyze(self, schema: PropertyGraphSchema, graph_info: GraphInfo) -> list:
        # Any graph is a candidate
        return []

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list,
    ) -> list[tuple[Nl2CypherSample, None]]:
        """Any sample can be corrupted with a missing entity type."""
        return [(sample, None) for sample in samples]

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: object,
        graph_name: str,
    ) -> CorruptedSample | None:
        existing_labels = get_all_entity_labels(self.schema)

        # Identify entity types used in this sample's query
        sample_entities = get_entity_labels(sample.gold_cypher)
        # Pick an entity in a required position (not OPTIONAL MATCH or one UNION branch)
        target_entity = None
        for e in sample_entities:
            if e in existing_labels and \
               not is_in_optional_match(sample.gold_cypher, e) and \
               not is_in_union_branch(sample.gold_cypher, e):
                target_entity = e
                break
        if target_entity is None:
            return None

        # Step 1: Generate a plausible but nonexistent entity type
        avoid_clause = ""
        if self._used_fake_types:
            avoid_clause = (
                f"\n\nDo NOT use any of these entity types (already used): "
                f"{', '.join(sorted(self._used_fake_types))}. "
                "Generate a different entity type."
            )

        fake_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": "You generate plausible but nonexistent entity types for knowledge graphs.",
                },
                {
                    "role": "user",
                    "content": (
                        f"The '{graph_name}' knowledge graph has these entity types:\n"
                        + "\n".join(f"- {label}" for label in existing_labels)
                        + f"\n\nGenerate an entity type name that:\n"
                        f"1. Does NOT exist in the list above\n"
                        f"2. Would be plausible for this domain\n"
                        f"3. Is SEMANTICALLY DISTINCT from all existing entity types — "
                        f"NOT a synonym, paraphrase, or subset of any existing type\n"
                        f"4. A question about this entity type CANNOT be answered by "
                        f"querying any existing entity type\n"
                        f"The type should relate to the same domain as '{target_entity}'. "
                        "Use PascalCase."
                        + avoid_clause
                    ),
                },
            ],
            response_model=FakeEntityTypeResponse,
        )

        if fake_resp.entity_type in existing_labels:
            return None

        self._used_fake_types.add(fake_resp.entity_type)

        # Step 2: Write a coherent question about the fake entity type
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write natural-sounding questions about knowledge graph "
                        "entities for a benchmark. The question must be logically "
                        "coherent and sound like something a real person would ask."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Domain: {graph_name}\n"
                        f"The entity type '{fake_resp.entity_type}' does NOT exist "
                        f"in this knowledge graph (existing types: "
                        f"{', '.join(existing_labels)}).\n\n"
                        f"Here is an original question about '{target_entity}' for "
                        f"reference:\n\"{sample.nl_question}\"\n\n"
                        f"Write a NEW question that:\n"
                        "1. Asks about '{fake_resp.entity_type}' in a way that is "
                        "logically coherent — the entity type must fit naturally\n"
                        "2. Is inspired by the original question (similar complexity "
                        "and domain context) but adapted so the sentence makes sense\n"
                        "3. Sounds like something a real person would ask about "
                        "this domain\n"
                        "4. Does NOT use brackets, placeholders, or schema labels\n\n"
                        "You may restructure the sentence as needed — coherence is "
                        "more important than similarity to the original."
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        return CorruptedSample(
            corruption_id=f"U3-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.U3,
                corruption_category="unanswerability",
                original_element=target_entity,
                corrupted_element=fake_resp.entity_type,
                reason_unanswerable=(
                    f"Entity type '{fake_resp.entity_type}' does not exist in the schema. "
                    f"Existing types: {', '.join(existing_labels)}"
                ),
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            expected_answer="UNANSWERABLE",
        )
