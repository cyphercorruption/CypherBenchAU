"""U5 - Temporal Unanswerability: question asks for temporal info the graph doesn't track."""

from __future__ import annotations

import uuid

from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

from cb_corruptions.corruptions import BaseCorruption, register_corruption
from cb_corruptions.cypher_parser import get_relation_types
from cb_corruptions.graph_analysis import TemporalCandidate, find_temporal_candidates
from cb_corruptions.models import (
    CorruptedSample,
    CorruptionMetadata,
    CorruptionType,
    TemporalQuestionResponse,
)


@register_corruption("U5")
class TemporalUnanswerabilityCorruption(BaseCorruption):

    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list[TemporalCandidate]:
        return find_temporal_candidates(schema, graph_info)

    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list[TemporalCandidate],
    ) -> list[tuple[Nl2CypherSample, TemporalCandidate]]:
        """Match samples that use a non-temporal relation."""
        non_temporal_labels = {c.label for c in candidates}
        rel_to_candidate = {c.label: c for c in candidates}

        pairs = []
        for sample in samples:
            used_rels = get_relation_types(sample.gold_cypher)
            for rel in used_rels:
                if rel in non_temporal_labels:
                    pairs.append((sample, rel_to_candidate[rel]))
                    break

        return pairs

    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: TemporalCandidate,
        graph_name: str,
    ) -> CorruptedSample | None:
        rel = candidate.relation

        # Rewrite the question to ask "when" about this relation
        temporal_resp = self.llm.structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite natural language questions to ask about temporal "
                        "information (when, what year, what date) for a relationship "
                        "that has no temporal data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"In the '{graph_name}' knowledge graph, the relationship "
                        f"'{rel.label}' connects {rel.subj_label} to {rel.obj_label} "
                        f"but has NO temporal properties (no year, date, start_year, etc.).\n\n"
                        f"Original question: {sample.nl_question}\n"
                        f"Original Cypher: {sample.gold_cypher}\n\n"
                        "Rewrite the question to ask about WHEN this relationship "
                        "occurred or a specific year/date. The question should sound "
                        "natural but be unanswerable because the temporal info doesn't exist."
                    ),
                },
            ],
            response_model=TemporalQuestionResponse,
        )

        return CorruptedSample(
            corruption_id=f"U5-{graph_name}-{uuid.uuid4().hex[:8]}",
            corruption=CorruptionMetadata(
                corruption_type=CorruptionType.U5,
                corruption_category="unanswerability",
                original_element=rel.label,
                corrupted_element=f"{rel.label} (temporal query)",
                reason_unanswerable=(
                    f"Relation '{rel.label}' between {rel.subj_label} and {rel.obj_label} "
                    f"has no temporal properties. Properties: "
                    f"{list(rel.properties.keys()) if rel.properties else 'none'}"
                ),
            ),
            original_qid=sample.qid,
            original_graph=graph_name,
            original_nl_question=sample.nl_question,
            original_gold_cypher=sample.gold_cypher,
            corrupted_nl_question=temporal_resp.question,
            expected_answer="UNANSWERABLE",
        )
