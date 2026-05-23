"""Unified U1-U4 Unanswerability: corrupt a question by replacing a real schema
element (extracted from the Cypher) with a plausible but nonexistent one.

Step 1: Parse the original Cypher to extract the exact element to corrupt.
Step 2: LLM invents a fake replacement and rewrites the question.
"""

from __future__ import annotations

import logging
import random
import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import (
    get_entity_labels,
    get_relation_types,
    get_returned_properties,
    get_where_properties,
    is_in_optional_match,
    is_in_union_branch,
    is_return_only_property,
    parse_cypher,
)
from cb_corruptions.graph_analysis import get_entity_property_names
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    RewrittenQuestionResponse,
    FakePropertyResponse,
    FakeRelationResponse,
    FakeEntityTypeResponse,
)

logger = logging.getLogger(__name__)


def _build_schema_summary(schema: PropertyGraphSchema) -> str:
    """Build a concise schema description for the LLM prompt."""
    lines = ["Entity types and their properties:"]
    for entity in schema.entities:
        props = [p for p in entity.properties if p != "name"]
        lines.append(f"  {entity.label}: {', '.join(props) if props else '(no properties)'}")
    lines.append("\nRelationships:")
    for rel in schema.relations:
        lines.append(f"  ({rel.subj_label})-[:{rel.label}]->({rel.obj_label})")
    return "\n".join(lines)


class _UnanswerableCorruption(BaseCorruption):
    """Base for U1-U4. Subclasses implement _extract_target and _generate_fake."""

    _MAX_REUSE = 3

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self._element_counts: dict[str, int] = {}
        self._rng = random.Random(42)

    def analyze(self, schema: PropertyGraphSchema, graph_info: GraphInfo) -> list:
        return []

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list,
    ) -> list[tuple[Nl2CypherSample, None]]:
        """Subclasses override to filter for relevant samples."""
        return [(s, None) for s in samples]

    def _exhausted_elements(self) -> list[str]:
        return [e for e, c in self._element_counts.items() if c >= self._MAX_REUSE]

    def _track(self, element: str) -> None:
        key = element.lower()
        self._element_counts[key] = self._element_counts.get(key, 0) + 1

    def _extract_target(self, sample: Nl2CypherSample) -> tuple[str, str] | None:
        """Extract (element_name, entity_label) from the Cypher. Override per type."""
        raise NotImplementedError

    def _corruption_type(self) -> CorruptionType:
        raise NotImplementedError

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: object,
        graph_name: str,
    ) -> CorruptedSample | None:
        # Step 1: Parse Cypher to get the exact element to corrupt
        target = self._extract_target(sample)
        if target is None:
            return None
        original_element, context_label = target

        # Step 2: LLM generates a fake replacement + rewritten question
        result = self._generate_fake_and_rewrite(
            sample, original_element, context_label, graph_name
        )
        if result is None:
            return None

        fake_element, corrupted_question, reason = result
        self._track(fake_element)

        return CorruptedSample(
            corruption_id=f"{self._corruption_type().value}-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=self._corruption_type(),
                corruption_category="unanswerability",
                original_element=original_element,
                corrupted_element=fake_element,
                reason_unanswerable=reason,
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=corrupted_question,
            expected_answer="UNANSWERABLE",
        )

    def _generate_fake_and_rewrite(
        self,
        sample: Nl2CypherSample,
        original_element: str,
        context_label: str,
        graph_name: str,
    ) -> tuple[str, str, str] | None:
        """Override per type. Returns (fake_element, corrupted_question, reason)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# U1: Missing Property (asked in RETURN)
# ---------------------------------------------------------------------------

@register_corruption("U1")
class U1MissingProperty(_UnanswerableCorruption):

    def _corruption_type(self) -> CorruptionType:
        return CorruptionType.U1

    def select_samples(self, samples, candidates):
        pairs = []
        for sample in samples:
            returned = get_returned_properties(sample.gold_cypher)
            if any(prop != "name" for _, prop in returned):
                pairs.append((sample, None))
        return pairs

    def _extract_target(self, sample):
        parsed = parse_cypher(sample.gold_cypher)
        returned = get_returned_properties(sample.gold_cypher)
        for var, prop in returned:
            if prop != "name":
                entity_label = parsed.node_labels.get(var, "")
                return (prop, entity_label)
        return None

    def _generate_fake_and_rewrite(self, sample, original_element, context_label, graph_name):
        existing_props = get_entity_property_names(self.schema, context_label)

        avoid = self._exhausted_elements()
        avoid_clause = ""
        if avoid:
            avoid_clause = (
                f"\nDo NOT use: {', '.join(avoid)}. Invent a different name."
            )

        fake_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You generate plausible but nonexistent properties for knowledge graph entities."},
                {"role": "user", "content": (
                    f"Entity type '{context_label}' has these properties:\n"
                    + "\n".join(f"- {p}" for p in existing_props)
                    + f"\n\nGenerate a property name that:\n"
                    "1. Does NOT exist in the list above\n"
                    "2. Would be plausible for this entity type\n"
                    "3. Is semantically distinct from all existing properties\n"
                    "Use snake_case." + avoid_clause
                )},
            ],
            response_model=FakePropertyResponse,
        )

        if fake_resp.property_name in existing_props:
            return None

        rewrite_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You rewrite questions to ask about a different property."},
                {"role": "user", "content": (
                    f"Rewrite this question to ask about '{fake_resp.property_name}' "
                    f"instead of '{original_element}'.\n\n"
                    f"Original question: {sample.nl_question}\n\n"
                    "Keep the rest of the question as close to the original as possible."
                )},
            ],
            response_model=RewrittenQuestionResponse,
        )

        reason = (
            f"Property '{fake_resp.property_name}' does not exist on entity type "
            f"'{context_label}'. Existing: {', '.join(existing_props)}"
        )
        return (fake_resp.property_name, rewrite_resp.question, reason)


# ---------------------------------------------------------------------------
# U2: Missing Relation
# ---------------------------------------------------------------------------

@register_corruption("U2")
class U2MissingRelation(_UnanswerableCorruption):

    def _corruption_type(self) -> CorruptionType:
        return CorruptionType.U2

    def select_samples(self, samples, candidates):
        pairs = []
        for sample in samples:
            rels = get_relation_types(sample.gold_cypher)
            for rel in rels:
                if not is_in_optional_match(sample.gold_cypher, rel) and \
                   not is_in_union_branch(sample.gold_cypher, rel):
                    pairs.append((sample, None))
                    break
        return pairs

    def _extract_target(self, sample):
        rels = get_relation_types(sample.gold_cypher)
        for rel in rels:
            if not is_in_optional_match(sample.gold_cypher, rel) and \
               not is_in_union_branch(sample.gold_cypher, rel):
                # Find entity types this relation connects
                for r in self.schema.relations:
                    if r.label == rel:
                        return (rel, f"{r.subj_label}->{r.obj_label}")
                return (rel, "")
        return None

    def _generate_fake_and_rewrite(self, sample, original_element, context_label, graph_name):
        all_rel_labels = [r.label for r in self.schema.relations]

        # Find entity types for context
        original_subj, original_obj = "", ""
        for r in self.schema.relations:
            if r.label == original_element:
                original_subj = r.subj_label
                original_obj = r.obj_label
                break

        same_type_rels = [
            r.label for r in self.schema.relations
            if r.subj_label == original_subj and r.obj_label == original_obj
        ]

        avoid = self._exhausted_elements()
        avoid_clause = ""
        if avoid:
            avoid_clause = f"\nDo NOT use: {', '.join(avoid)}."

        fake_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You generate plausible but nonexistent relationship types for knowledge graphs."},
                {"role": "user", "content": (
                    f"Relationship '{original_element}' connects {original_subj} to {original_obj}.\n\n"
                    f"Existing relationships between {original_subj} and {original_obj}:\n"
                    + "\n".join(f"- {r}" for r in same_type_rels)
                    + f"\n\nAll relationships in the graph:\n"
                    + "\n".join(f"- {r}" for r in all_rel_labels)
                    + f"\n\nGenerate a relationship name that:\n"
                    f"1. Could plausibly connect {original_subj} to {original_obj}\n"
                    "2. Does NOT exist in the lists above\n"
                    "3. Is semantically distinct from existing relationships\n"
                    "Use camelCase." + avoid_clause
                )},
            ],
            response_model=FakeRelationResponse,
        )

        if fake_resp.relation_name in all_rel_labels:
            return None

        rewrite_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You minimally rewrite questions to reference a different relationship."},
                {"role": "user", "content": (
                    f"Rewrite this question by replacing the concept of "
                    f"'{original_element}' with '{fake_resp.relation_name}'.\n\n"
                    f"Original question: {sample.nl_question}\n\n"
                    "Keep the same entities and structure. Only change the relationship words."
                )},
            ],
            response_model=RewrittenQuestionResponse,
        )

        reason = (
            f"Relation '{fake_resp.relation_name}' does not exist. "
            f"Existing: {', '.join(all_rel_labels)}"
        )
        return (fake_resp.relation_name, rewrite_resp.question, reason)


# ---------------------------------------------------------------------------
# U3: Missing Entity Type
# ---------------------------------------------------------------------------

@register_corruption("U3")
class U3MissingEntityType(_UnanswerableCorruption):

    def _corruption_type(self) -> CorruptionType:
        return CorruptionType.U3

    def _extract_target(self, sample):
        existing = get_entity_labels(self.schema.model_dump()["name"]) if False else \
            {e.label for e in self.schema.entities}
        sample_entities = get_entity_labels(sample.gold_cypher)
        for e in sample_entities:
            if e in existing and \
               not is_in_optional_match(sample.gold_cypher, e) and \
               not is_in_union_branch(sample.gold_cypher, e):
                return (e, "")
        return None

    def _generate_fake_and_rewrite(self, sample, original_element, context_label, graph_name):
        existing_labels = [e.label for e in self.schema.entities]

        avoid = self._exhausted_elements()
        avoid_clause = ""
        if avoid:
            avoid_clause = f"\nDo NOT use: {', '.join(avoid)}."

        fake_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You generate plausible but nonexistent entity types for knowledge graphs."},
                {"role": "user", "content": (
                    f"The '{graph_name}' knowledge graph has these entity types:\n"
                    + "\n".join(f"- {l}" for l in existing_labels)
                    + f"\n\nGenerate an entity type that:\n"
                    "1. Does NOT exist in the list above\n"
                    "2. Would be plausible for this domain\n"
                    "3. Is semantically distinct from existing types\n"
                    f"Should relate to '{original_element}'. Use PascalCase." + avoid_clause
                )},
            ],
            response_model=FakeEntityTypeResponse,
        )

        if fake_resp.entity_type in existing_labels:
            return None

        rewrite_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": (
                    "You write natural questions about knowledge graph entities. "
                    "The question must be logically coherent."
                )},
                {"role": "user", "content": (
                    f"Original question (about '{original_element}'): {sample.nl_question}\n\n"
                    f"Write a new question that asks about '{fake_resp.entity_type}' instead "
                    f"of '{original_element}'. The question must be coherent — adapt the "
                    "sentence as needed so it makes sense with the new entity type. "
                    "Keep it as close to the original as possible."
                )},
            ],
            response_model=RewrittenQuestionResponse,
        )

        reason = (
            f"Entity type '{fake_resp.entity_type}' does not exist. "
            f"Existing: {', '.join(existing_labels)}"
        )
        return (fake_resp.entity_type, rewrite_resp.question, reason)


# ---------------------------------------------------------------------------
# U4: Out-of-Schema Constraint (property in WHERE clause)
# ---------------------------------------------------------------------------

@register_corruption("U4")
class U4OutOfSchemaConstraint(_UnanswerableCorruption):

    def _corruption_type(self) -> CorruptionType:
        return CorruptionType.U4

    def select_samples(self, samples, candidates):
        pairs = []
        for sample in samples:
            where_props = get_where_properties(sample.gold_cypher)
            parsed = parse_cypher(sample.gold_cypher)
            for var, prop in where_props:
                if prop != "name" and var in parsed.node_labels:
                    pairs.append((sample, None))
                    break
        return pairs

    def _extract_target(self, sample):
        parsed = parse_cypher(sample.gold_cypher)
        where_props = get_where_properties(sample.gold_cypher)
        for var, prop in where_props:
            if prop != "name" and var in parsed.node_labels:
                return (prop, parsed.node_labels[var])
        return None

    def _generate_fake_and_rewrite(self, sample, original_element, context_label, graph_name):
        existing_props = get_entity_property_names(self.schema, context_label)

        avoid = self._exhausted_elements()
        avoid_clause = ""
        if avoid:
            avoid_clause = f"\nDo NOT use: {', '.join(avoid)}."

        fake_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You generate plausible but nonexistent properties for knowledge graph entities."},
                {"role": "user", "content": (
                    f"Entity type '{context_label}' has these properties:\n"
                    + "\n".join(f"- {p}" for p in existing_props)
                    + f"\n\nGenerate a property that:\n"
                    "1. Does NOT exist in the list above\n"
                    "2. Would be plausible as a filter condition\n"
                    "3. Is semantically distinct from existing properties\n"
                    "Use snake_case." + avoid_clause
                )},
            ],
            response_model=FakePropertyResponse,
        )

        if fake_resp.property_name in existing_props:
            return None

        rewrite_resp = self.llm.structured(
            messages=[
                {"role": "system", "content": "You rewrite questions to use a different filter condition."},
                {"role": "user", "content": (
                    f"Rewrite this question to filter by '{fake_resp.property_name}' "
                    f"instead of '{original_element}'.\n\n"
                    f"Original question: {sample.nl_question}\n\n"
                    "Use a concrete filter value (not placeholders). "
                    "Keep the rest of the question intact."
                )},
            ],
            response_model=RewrittenQuestionResponse,
        )

        reason = (
            f"Property '{fake_resp.property_name}' does not exist on '{context_label}'. "
            f"Existing: {', '.join(existing_props)}"
        )
        return (fake_resp.property_name, rewrite_resp.question, reason)
