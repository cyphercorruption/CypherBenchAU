"""A2 - Property Ambiguity: question refers to a property that maps to 2+ properties.

Pre-computes property clusters during analyze() — one LLM call per entity type
to find all semantic overlap groups. Then corrupt() just picks a cluster and rewrites.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_returned_properties, parse_cypher
from cb_corruptions.graph_analysis import (
    PropertyAmbiguityCandidate,
    find_property_ambiguity_candidates,
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


@dataclass
class PropertyCluster:
    """A group of semantically overlapping properties on the same entity."""
    entity_label: str
    properties: list[str]


@dataclass
class PropertyClusterCandidate:
    """An entity's property cluster with pre-generated ambiguous terms."""
    entity_label: str
    properties: list[str]
    hypernyms: list[str] = field(default_factory=list)


# Response model for property subgroup clustering
class PropertySubgroupResponse(RelationSubgroupResponse):
    """Reuse the same structure: list of subgroups with rationale."""
    pass


@register_corruption("A2")
class PropertyAmbiguityCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        self._clusters: list[PropertyClusterCandidate] = []
        self._hypernym_idx: dict[str, int] = defaultdict(int)

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list[PropertyClusterCandidate]:
        raw_candidates = find_property_ambiguity_candidates(schema)

        if self.llm is None:
            # Analysis-only mode
            return [
                PropertyClusterCandidate(
                    entity_label=c.entity_label,
                    properties=c.properties,
                )
                for c in raw_candidates
            ]

        clusters: list[PropertyClusterCandidate] = []

        for candidate in raw_candidates:
            if len(candidate.properties) < 2:
                continue

            # Ask LLM to find semantic overlap groups among properties
            subgroups = self._find_property_subgroups(
                candidate.entity_label, candidate.properties
            )
            for sg in subgroups:
                cluster = PropertyClusterCandidate(
                    entity_label=candidate.entity_label,
                    properties=sg,
                )
                # Pre-generate ambiguous terms
                self._pregenerate_hypernyms(cluster)
                if cluster.hypernyms:
                    clusters.append(cluster)

        self._clusters = clusters
        logger.info(
            "  A2: found %d property clusters across %d entity types",
            len(clusters),
            len(raw_candidates),
        )
        return clusters

    def _find_property_subgroups(
        self,
        entity_label: str,
        properties: list[str],
    ) -> list[list[str]]:
        """Ask LLM to find groups of semantically overlapping properties."""
        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find natural semantic overlaps between knowledge-graph "
                        "properties — cases where the same everyday word could "
                        "plausibly refer to multiple different properties."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Entity type '{entity_label}' has these properties:\n"
                        + "\n".join(f"- {p}" for p in properties)
                        + "\n\nFind all subgroups of 2 or more properties that "
                        "have natural semantic overlap — where the same everyday "
                        "word could plausibly refer to any of them.\n"
                        "Rules:\n"
                        "- Only group properties whose meanings genuinely overlap\n"
                        "- Do NOT group unrelated properties\n"
                        "- Include subgroups at different granularities\n"
                        "- Subgroups can overlap\n"
                        "- Only include the full set if ALL properties share overlap"
                    ),
                },
            ],
            response_model=PropertySubgroupResponse,
        )

        subgroups = []
        for sg in resp.subgroups:
            # Filter to properties that actually exist
            valid = [p for p in sg.relations if p in properties]
            if len(valid) >= 2:
                subgroups.append(valid)

        logger.info(
            "  A2 subgroups for %s: %d properties -> %d subgroups",
            entity_label,
            len(properties),
            len(subgroups),
        )
        return subgroups

    def _pregenerate_hypernyms(self, cluster: PropertyClusterCandidate) -> None:
        """Ask LLM for all plausible ambiguous terms for a property cluster."""
        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find everyday words that naturally cover multiple "
                        "knowledge-graph properties because they share similar meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"These properties belong to '{cluster.entity_label}':\n"
                        + "\n".join(f"- {p}" for p in cluster.properties)
                        + "\n\nList everyday words or short phrases that a real "
                        "person would use in a question, where the word naturally "
                        "covers all of these properties because they share "
                        "similar meaning.\n"
                        "Requirements:\n"
                        "- The word must be a natural synonym or near-synonym at "
                        "the intersection of these properties\n"
                        "- NOT a vague umbrella term\n"
                        "- Must sound natural in a question\n"
                        "- Return only words that genuinely work — empty is fine"
                    ),
                },
            ],
            response_model=HypernymListResponse,
        )
        cluster.hypernyms = resp.hypernyms
        logger.info(
            "  A2 hypernyms for %s %s: %s",
            cluster.entity_label,
            cluster.properties,
            cluster.hypernyms,
        )

    def _next_hypernym(self, cluster: PropertyClusterCandidate) -> str | None:
        if not cluster.hypernyms:
            return None
        key = f"{cluster.entity_label}:{','.join(sorted(cluster.properties))}"
        idx = self._hypernym_idx[key] % len(cluster.hypernyms)
        self._hypernym_idx[key] += 1
        return cluster.hypernyms[idx]

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list[PropertyClusterCandidate],
    ) -> list[tuple[Nl2CypherSample, PropertyClusterCandidate]]:
        """Match samples that query a property belonging to a cluster."""
        # Build lookup: (entity_label, property) -> clusters
        prop_to_clusters: dict[tuple[str, str], list[PropertyClusterCandidate]] = defaultdict(list)
        for cluster in candidates:
            for prop in cluster.properties:
                prop_to_clusters[(cluster.entity_label, prop)].append(cluster)

        pairs: list[tuple[Nl2CypherSample, PropertyClusterCandidate]] = []
        for sample in samples:
            parsed = parse_cypher(sample.gold_cypher)
            returned = get_returned_properties(sample.gold_cypher)
            seen: set[int] = set()
            for var, prop in returned:
                if prop == "name":
                    continue
                entity_label = parsed.node_labels.get(var)
                if entity_label:
                    for cluster in prop_to_clusters.get((entity_label, prop), []):
                        cid = id(cluster)
                        if cid not in seen:
                            pairs.append((sample, cluster))
                            seen.add(cid)

        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: PropertyClusterCandidate,
        graph_name: str,
    ) -> CorruptedSample | None:
        cluster = candidate

        # Find which property the original query accesses
        parsed = parse_cypher(sample.gold_cypher)
        returned = get_returned_properties(sample.gold_cypher)
        original_prop = None
        for var, prop in returned:
            if prop != "name" and parsed.node_labels.get(var) == cluster.entity_label:
                if prop in cluster.properties:
                    original_prop = prop
                    break

        if original_prop is None:
            return None

        # Pick next hypernym
        hypernym = self._next_hypernym(cluster)
        if hypernym is None:
            return None

        # Rewrite the question
        alternatives_str = "\n".join(f"- {p}" for p in cluster.properties)
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at crafting ambiguous questions for "
                        "a text-to-query benchmark. Your goal is to corrupt a "
                        "clear question into one that admits multiple exclusive "
                        "interpretations — each leading to a different, equally "
                        "valid answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {sample.nl_question}\n\n"
                        f"This question asks about the property '{original_prop}'. "
                        f"The following properties all have semantic overlap:\n"
                        f"{alternatives_str}\n\n"
                        f"Suggested ambiguous term: '{hypernym}'\n\n"
                        "Rewrite the question so that:\n"
                        "1. Multiple exclusive interpretations exist — each "
                        "corresponding to one of the properties above\n"
                        "2. Someone would need to ask 'which property do you mean?' "
                        "to answer\n"
                        "3. The question sounds natural\n"
                        "4. Use singular form\n"
                        "5. Keep entity names intact\n\n"
                        "You may use the suggested term or find a better one."
                    ),
                },
            ],
            response_model=RewrittenQuestionResponse,
        )

        # Generate valid Cypher for each property in the cluster
        valid_cyphers = []
        for prop in cluster.properties:
            variant = sample.gold_cypher.replace(f".{original_prop}", f".{prop}")
            valid_cyphers.append(variant)

        return CorruptedSample(
            corruption_id=f"A2-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.A2,
                corruption_category="ambiguity",
                original_element=original_prop,
                corrupted_element=hypernym,
                candidate_interpretations=cluster.properties,
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            valid_cyphers=valid_cyphers,
            expected_answer="AMBIGUOUS",
        )
