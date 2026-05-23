"""Data models for corrupted benchmark samples."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class CorruptionType(str, Enum):
    """The 9 corruption types (A4 excluded)."""

    A1 = "A1"  # Relation Ambiguity
    A2 = "A2"  # Property Ambiguity
    A3 = "A3"  # Entity Type Ambiguity
    A5 = "A5"  # Direction Ambiguity
    U1 = "U1"  # Missing Property
    U2 = "U2"  # Missing Relation
    U3 = "U3"  # Missing Entity Type
    U4 = "U4"  # Out-of-Schema Constraint
    U5 = "U5"  # Temporal Unanswerability


AMBIGUITY_TYPES = {CorruptionType.A1, CorruptionType.A2, CorruptionType.A3, CorruptionType.A5}
UNANSWERABILITY_TYPES = {
    CorruptionType.U1,
    CorruptionType.U2,
    CorruptionType.U3,
    CorruptionType.U4,
    CorruptionType.U5,
}

ALL_CORRUPTION_TYPES = sorted(CorruptionType, key=lambda t: t.value)


class CorruptionMetadata(BaseModel):
    """What was corrupted and why."""

    corruption_type: CorruptionType
    corruption_category: Literal["ambiguity", "unanswerability"]

    # The schema element that was targeted
    original_element: str  # e.g. "directedBy", "height_cm"
    corrupted_element: str  # e.g. "involved with", "salary"

    # Ambiguity: the 2+ valid interpretations
    candidate_interpretations: list[str] = []

    # Unanswerability: why no valid Cypher exists
    reason_unanswerable: str | None = None


class CorruptedSample(BaseModel):
    """A single corrupted benchmark sample."""

    corruption_id: str
    corruption: CorruptionMetadata

    # Original sample info (preserved for reference)
    original_qid: str
    original_graph: str
    original_nl_question: str
    original_gold_cypher: str

    # Corrupted output
    corrupted_nl_question: str

    # Ambiguity: list of valid Cypher interpretations
    valid_cyphers: list[str] = []

    # Expected label
    expected_answer: Literal["AMBIGUOUS", "UNANSWERABLE"]


# ---------------------------------------------------------------------------
# LLM response models (structured output schemas)
# ---------------------------------------------------------------------------


class HypernymResponse(BaseModel):
    """LLM output: a hypernym covering 2+ schema elements."""

    hypernym: str
    explanation: str


class HypernymListResponse(BaseModel):
    """LLM output: all plausible hypernyms for a group of relations."""

    hypernyms: list[str]


class RewrittenQuestionResponse(BaseModel):
    """LLM output: a rewritten NL question."""

    question: str


class MergedQuestionResponse(BaseModel):
    """LLM output: a single ambiguous question merging 2+ source questions."""

    question: str
    ambiguous_term: str


class PropertyClusterResponse(BaseModel):
    """LLM output: a hypernym and the cluster of similar properties it covers."""

    hypernym: str
    similar_properties: list[str]
    explanation: str


class FakePropertyResponse(BaseModel):
    """LLM output: a plausible but nonexistent property."""

    property_name: str
    explanation: str


class FakeRelationResponse(BaseModel):
    """LLM output: a plausible but nonexistent relation."""

    relation_name: str
    explanation: str


class FakeEntityTypeResponse(BaseModel):
    """LLM output: a plausible but nonexistent entity type."""

    entity_type: str
    explanation: str


class UnanswerableCorruptionResponse(BaseModel):
    """LLM output: a one-shot unanswerable corruption of a question."""

    corrupted_question: str
    original_element: str  # Exact schema name (e.g. "creation_year", "displayedAt", "Painting")
    corrupted_element: str  # Exact fake element name (e.g. "material", "restoredBy", "Installation")
    corruption_type: str  # "missing_property", "missing_relation", "missing_entity_type", "out_of_schema_constraint"
    reason_unanswerable: str


class TemporalQuestionResponse(BaseModel):
    """LLM output: a question asking about temporal info."""

    question: str


class AmbiguityValidationResponse(BaseModel):
    """LLM output: whether a rewritten question is genuinely ambiguous."""

    is_ambiguous: bool
    reason: str


class RelationSubgroupResponse(BaseModel):
    """LLM output: semantically coherent subgroups of relations."""

    class Subgroup(BaseModel):
        relations: list[str]
        rationale: str

    subgroups: list[Subgroup]
