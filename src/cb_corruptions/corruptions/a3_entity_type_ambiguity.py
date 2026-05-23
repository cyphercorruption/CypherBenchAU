"""A3 - Entity Type Ambiguity: question uses a term that refers to 2+ entity types.

Only considers source types that share the SAME relation to the target,
so Cypher variants only need to swap the entity label (safe str.replace).

Pre-computes clusters and ambiguous terms during analyze().
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_entity_labels
from cb_corruptions.graph_analysis import (
    EntityTypeAmbiguityCandidate,
    find_entity_type_ambiguity_candidates,
)
from cb_corruptions.llm import LLM
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    HypernymListResponse,
    MergedQuestionResponse,
)

logger = logging.getLogger(__name__)


@register_corruption("A3")
class EntityTypeAmbiguityCorruption(BaseCorruption):

    def __init__(self, llm: LLM) -> None:
        super().__init__(llm)
        # Pre-generated hypernyms per candidate, keyed by sorted source labels
        self._hypernyms: dict[str, list[str]] = {}
        self._hypernym_idx: dict[str, int] = defaultdict(int)

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list[EntityTypeAmbiguityCandidate]:
        candidates = find_entity_type_ambiguity_candidates(schema)

        if self.llm is None:
            return candidates

        # Pre-generate ambiguous terms for each candidate
        for candidate in candidates:
            self._pregenerate_hypernyms(candidate)

        # Only keep candidates that have at least one hypernym
        candidates = [c for c in candidates if self._group_key(c) in self._hypernyms
                      and self._hypernyms[self._group_key(c)]]

        logger.info("  A3: %d candidates with hypernyms", len(candidates))
        return candidates

    @staticmethod
    def _group_key(candidate: EntityTypeAmbiguityCandidate) -> str:
        source_labels = sorted({src for src, _ in candidate.sources})
        return f"{candidate.target_label}:{','.join(source_labels)}"

    def _pregenerate_hypernyms(self, candidate: EntityTypeAmbiguityCandidate) -> None:
        key = self._group_key(candidate)
        if key in self._hypernyms:
            return

        source_labels = sorted({src for src, _ in candidate.sources})
        rel_label = candidate.sources[0][1]  # all share the same relation

        resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find everyday words that naturally cover multiple "
                        "entity types because they share similar meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"These entity types all connect to '{candidate.target_label}' "
                        f"via the '{rel_label}' relationship:\n"
                        + "\n".join(f"- {label}" for label in source_labels)
                        + "\n\nList everyday words that a real person would use "
                        "in a question, where the word naturally covers all of "
                        "these entity types because they share similar meaning.\n"
                        "Requirements:\n"
                        "- The word must be a natural synonym or near-synonym "
                        "at the intersection of these entity types\n"
                        "- NOT a vague umbrella term like 'thing' or 'item'\n"
                        "- Must sound natural in a question\n"
                        "- Return only words that genuinely work — empty is fine"
                    ),
                },
            ],
            response_model=HypernymListResponse,
        )

        self._hypernyms[key] = resp.hypernyms
        logger.info(
            "  A3 hypernyms for %s -> %s [%s]: %s",
            source_labels,
            candidate.target_label,
            rel_label,
            resp.hypernyms,
        )

    def _next_hypernym(self, candidate: EntityTypeAmbiguityCandidate) -> str | None:
        key = self._group_key(candidate)
        hypernyms = self._hypernyms.get(key, [])
        if not hypernyms:
            return None
        idx = self._hypernym_idx[key] % len(hypernyms)
        self._hypernym_idx[key] += 1
        return hypernyms[idx]

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list[EntityTypeAmbiguityCandidate],
    ) -> list[tuple[Nl2CypherSample, EntityTypeAmbiguityCandidate]]:
        """Match samples whose gold_cypher uses a source entity type."""
        source_to_candidate: dict[str, EntityTypeAmbiguityCandidate] = {}
        for candidate in candidates:
            for src_label, _ in candidate.sources:
                source_to_candidate[src_label] = candidate

        pairs = []
        for sample in samples:
            labels = get_entity_labels(sample.gold_cypher)
            for label in labels:
                if label in source_to_candidate:
                    pairs.append((sample, source_to_candidate[label]))
                    break

        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: EntityTypeAmbiguityCandidate,
        graph_name: str,
    ) -> CorruptedSample | None:
        labels = get_entity_labels(sample.gold_cypher)
        original_label = next(
            (l for l in labels if l in {src for src, _ in candidate.sources}),
            None,
        )
        if original_label is None:
            return None

        source_labels = sorted({src for src, _ in candidate.sources})

        hypernym = self._next_hypernym(candidate)
        if hypernym is None:
            return None

        # Rewrite the question
        rewrite_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You find natural semantic overlaps between entity types "
                        "and rewrite questions to exploit that overlap, making it "
                        "unclear which entity type is being asked about."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original question: {sample.nl_question}\n\n"
                        f"This question asks about '{original_label}'. However, "
                        f"these entity types all share the same relationship:\n"
                        + "\n".join(f"- {label}" for label in source_labels)
                        + f"\n\nSuggested ambiguous term: '{hypernym}'\n\n"
                        f"Rewrite the question replacing '{original_label}' with "
                        f"'{hypernym}' (or a natural variant) so a reader cannot "
                        f"tell which entity type is meant.\n\n"
                        "Rules:\n"
                        "1. Keep the question as close to the original as "
                        "possible — only change the entity type reference\n"
                        "2. Keep all entity names, filters, and details intact\n"
                        "3. The question must sound natural\n"
                        "4. Use singular form\n"
                        "5. Return the ambiguous term you used"
                    ),
                },
            ],
            response_model=MergedQuestionResponse,
        )

        # Generate valid Cypher for each source entity type
        # Safe: all sources share the same relation, only swap the label
        valid_cyphers = []
        for src_label, _ in candidate.sources:
            variant = sample.gold_cypher.replace(f":{original_label}", f":{src_label}")
            valid_cyphers.append(variant)

        return CorruptedSample(
            corruption_id=f"A3-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.A3,
                corruption_category="ambiguity",
                original_element=original_label,
                corrupted_element=rewrite_resp.ambiguous_term,
                candidate_interpretations=source_labels,
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=rewrite_resp.question,
            valid_cyphers=valid_cyphers,
            expected_answer="AMBIGUOUS",
        )
