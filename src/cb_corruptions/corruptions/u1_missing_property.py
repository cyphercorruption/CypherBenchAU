"""U1 - Missing Property: question asks for a property not in the schema."""

from __future__ import annotations

import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_returned_properties, is_return_only_property, parse_cypher
from cb_corruptions.graph_analysis import get_entity_property_names
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    FakePropertyResponse,
    RewrittenQuestionResponse,
)


@register_corruption("U1")
class MissingPropertyCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self._used_fake_props: set[str] = set()

    def analyze(self, schema: PropertyGraphSchema, graph_info: GraphInfo) -> list:
        # No schema analysis needed; any entity with properties is a candidate
        return []

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list,
    ) -> list[tuple[Nl2CypherSample, None]]:
        """Select samples that query a non-name property in a structurally essential position.

        Skip samples where the target property only appears in RETURN — replacing
        it would just produce null values while the query still returns rows.
        """
        pairs = []
        for sample in samples:
            returned = get_returned_properties(sample.gold_cypher)
            for _, prop in returned:
                if prop != "name" and not is_return_only_property(sample.gold_cypher, prop):
                    pairs.append((sample, None))
                    break
        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: object,
        graph_name: str,
    ) -> CorruptedSample | None:
        parsed = parse_cypher(sample.gold_cypher)
        returned = get_returned_properties(sample.gold_cypher)

        # Find a property that is structurally essential (not RETURN-only)
        original_prop = None
        entity_label = None
        for var, prop in returned:
            if prop != "name" and not is_return_only_property(sample.gold_cypher, prop):
                original_prop = prop
                entity_label = parsed.node_labels.get(var)
                break

        if original_prop is None or entity_label is None:
            return None

        # Get existing properties for context
        existing_props = get_entity_property_names(self.schema, entity_label)

        # Step 1: Generate a plausible but nonexistent property
        avoid_clause = ""
        if self._used_fake_props:
            avoid_clause = (
                f"\n\nDo NOT use any of these names (already used): "
                f"{', '.join(sorted(self._used_fake_props))}. "
                "Generate a different property name."
            )

        fake_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": "You generate plausible but nonexistent properties for knowledge graph entities.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Entity type '{entity_label}' in the '{graph_name}' knowledge graph "
                        f"has these properties:\n"
                        + "\n".join(f"- {p}" for p in existing_props)
                        + f"\n\nGenerate a property name that:\n"
                        "1. Does NOT exist in the list above\n"
                        "2. Would be plausible for this entity type\n"
                        "3. Is SEMANTICALLY DISTINCT from all existing properties — "
                        "NOT a synonym, paraphrase, or closely related concept\n"
                        "4. A question about this property CANNOT be answered using "
                        "any existing property\n"
                        "Use snake_case naming."
                        + avoid_clause
                    ),
                },
            ],
            response_model=FakePropertyResponse,
        )

        # Verify it's actually not in the schema
        if fake_resp.property_name in existing_props:
            return None

        self._used_fake_props.add(fake_resp.property_name)

        # Step 2: Rewrite the question to ask about the fake property
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite natural language questions to ask about a different property.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Rewrite this question to ask about '{fake_resp.property_name}' instead "
                        f"of '{original_prop}'.\n\n"
                        f"Original question: {sample.nl_question}\n\n"
                        "Output a natural-sounding question. Keep the rest of the question "
                        "as close to the original as possible."
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        return CorruptedSample(
            corruption_id=f"U1-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.U1,
                corruption_category="unanswerability",
                original_element=original_prop,
                corrupted_element=fake_resp.property_name,
                reason_unanswerable=(
                    f"Property '{fake_resp.property_name}' does not exist on entity type "
                    f"'{entity_label}'. Existing properties: {', '.join(existing_props)}"
                ),
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            expected_answer="UNANSWERABLE",
        )
