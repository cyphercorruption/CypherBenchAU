"""Static schema analysis: find corruption candidates from graph schemas.

Each function takes a PropertyGraphSchema (and optionally GraphInfo) and
returns a list of candidate dataclasses that downstream corruption
generators can use to select and corrupt benchmark samples.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cb_corruptions.schema import GraphInfo, PropertyGraphSchema, RelationSchema

# ---------------------------------------------------------------------------
# Candidate dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RelationAmbiguityCandidate:
    """A1: 2+ relations sharing the same (subj_label, obj_label) pair."""

    subj_label: str
    obj_label: str
    relations: list[RelationSchema]


@dataclass
class PropertyAmbiguityCandidate:
    """A2: 2+ semantically similar properties on the same entity type."""

    entity_label: str
    properties: list[str]
    # The hypernym is generated later by the LLM


@dataclass
class EntityTypeAmbiguityCandidate:
    """A3: 2+ entity types connected to the same target type."""

    target_label: str
    sources: list[tuple[str, str]]  # [(entity_label, relation_label), ...]


@dataclass
class DirectionAmbiguityCandidate:
    """A5: A self-referential relation (subj == obj) that is NOT symmetric."""

    relation: RelationSchema
    entity_label: str


@dataclass
class TemporalCandidate:
    """U5: A relation with no temporal properties."""

    relation: RelationSchema

    @property
    def label(self) -> str:
        return self.relation.label


@dataclass
class DisconnectedPair:
    """U2: Two entity types with no relation between them."""

    entity_a: str
    entity_b: str


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

TEMPORAL_PROPERTY_NAMES = {"start_year", "end_year", "year", "date"}


def find_relation_ambiguity_candidates(
    schema: PropertyGraphSchema,
) -> list[RelationAmbiguityCandidate]:
    """A1: Group relations by (subj_label, obj_label), keep groups with 2+ relations."""
    groups: dict[tuple[str, str], list[RelationSchema]] = defaultdict(list)
    for rel in schema.relations:
        groups[(rel.subj_label, rel.obj_label)].append(rel)

    return [
        RelationAmbiguityCandidate(subj_label=subj, obj_label=obj, relations=rels)
        for (subj, obj), rels in groups.items()
        if len(rels) >= 2
    ]


def find_property_ambiguity_candidates(
    schema: PropertyGraphSchema,
) -> list[PropertyAmbiguityCandidate]:
    """A2: Find entity types with 2+ non-trivial properties (excluding 'name').

    The actual semantic clustering is done by the LLM at corruption time.
    Here we just identify entities with enough properties to be ambiguous.
    """
    candidates = []
    for entity in schema.entities:
        props = [p for p in entity.properties if p != "name"]
        if len(props) >= 2:
            candidates.append(
                PropertyAmbiguityCandidate(entity_label=entity.label, properties=props)
            )
    return candidates


def find_entity_type_ambiguity_candidates(
    schema: PropertyGraphSchema,
) -> list[EntityTypeAmbiguityCandidate]:
    """A3: Find different entity types connected to the same target via the SAME relation.

    Only groups source types that share a relation label, so that Cypher
    variants only need to swap the entity label (not the relation).

    E.g. (Painting)-[:displayedAt]->(Museum) AND (Sculpture)-[:displayedAt]->(Museum):
    target=Museum, sources=[(Painting, displayedAt), (Sculpture, displayedAt)]
    """
    # Group by (target_label, relation_label): which source types use this?
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rel in schema.relations:
        groups[(rel.obj_label, rel.label)].append(rel.subj_label)

    candidates = []
    for (target, rel_label), source_labels in groups.items():
        # Only keep groups with 2+ distinct source entity types sharing the same relation
        distinct = list(set(source_labels))
        if len(distinct) >= 2:
            sources = [(src, rel_label) for src in distinct]
            candidates.append(
                EntityTypeAmbiguityCandidate(target_label=target, sources=sources)
            )
    return candidates


def find_direction_ambiguity_candidates(
    schema: PropertyGraphSchema,
    graph_info: GraphInfo,
) -> list[DirectionAmbiguityCandidate]:
    """A5: Find self-referential relations (subj == obj) that are NOT symmetric."""
    symmetric_labels = {
        rel.label for rel in graph_info.relations if rel.is_symmetric
    }

    return [
        DirectionAmbiguityCandidate(relation=rel, entity_label=rel.subj_label)
        for rel in schema.relations
        if rel.subj_label == rel.obj_label and rel.label not in symmetric_labels
    ]


def find_temporal_candidates(
    schema: PropertyGraphSchema,
    graph_info: GraphInfo,
) -> list[TemporalCandidate]:
    """U5: Find relations that have NO temporal properties AND are not time-sensitive."""
    time_sensitive_labels = {
        rel.label for rel in graph_info.relations if rel.is_time_sensitive
    }

    candidates = []
    for rel in schema.relations:
        has_temporal_props = bool(set(rel.properties.keys()) & TEMPORAL_PROPERTY_NAMES)
        if not has_temporal_props and rel.label not in time_sensitive_labels:
            candidates.append(TemporalCandidate(relation=rel))
    return candidates


def find_disconnected_pairs(
    schema: PropertyGraphSchema,
) -> list[DisconnectedPair]:
    """U2: Find entity type pairs NOT connected by any relation."""
    all_labels = {e.label for e in schema.entities}

    # Build set of connected pairs (in both directions)
    connected: set[tuple[str, str]] = set()
    for rel in schema.relations:
        connected.add((rel.subj_label, rel.obj_label))
        connected.add((rel.obj_label, rel.subj_label))

    pairs = []
    labels_list = sorted(all_labels)
    for i, a in enumerate(labels_list):
        for b in labels_list[i + 1 :]:
            if (a, b) not in connected:
                pairs.append(DisconnectedPair(entity_a=a, entity_b=b))
    return pairs


def get_entity_property_names(schema: PropertyGraphSchema, entity_label: str) -> list[str]:
    """Get property names for an entity type (excluding 'name')."""
    for entity in schema.entities:
        if entity.label == entity_label:
            return [p for p in entity.properties if p != "name"]
    return []


def get_all_entity_labels(schema: PropertyGraphSchema) -> list[str]:
    """Get all entity type labels in the schema."""
    return [e.label for e in schema.entities]
