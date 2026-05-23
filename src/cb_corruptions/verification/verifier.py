"""Core verification logic for corrupted samples."""

from __future__ import annotations

import logging
import re
from typing import Any

from neo4j.exceptions import ClientError, CypherSyntaxError

from cb_corruptions.cypher_substitutor import replace_label, replace_property, replace_relation
from cb_corruptions.models import AMBIGUITY_TYPES, CorruptedSample, CorruptionType
from cb_corruptions.schema_loader import load_graph_info
from cb_corruptions.verification.models import (
    AmbiguityVerdict,
    UnanswerabilityVerdict,
    VerificationResult,
    VerificationStatus,
)
from cb_corruptions.verification.neo4j_client import Neo4jClient
from cb_corruptions.verification.utils import result_fingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error classification patterns (Neo4j error messages)
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("missing_property", re.compile(r"Type mismatch:.*has no property", re.IGNORECASE)),
    ("missing_property", re.compile(r"property `[^`]+` does not exist", re.IGNORECASE)),
    ("missing_property", re.compile(r"Unknown property", re.IGNORECASE)),
    ("missing_label", re.compile(r"Label `?[^`]*`? does not exist", re.IGNORECASE)),
    ("missing_label", re.compile(r"Unknown label", re.IGNORECASE)),
    ("missing_relationship", re.compile(r"Type `?[^`]*`? does not exist", re.IGNORECASE)),
    ("missing_relationship", re.compile(r"Unknown relationship type", re.IGNORECASE)),
    ("syntax_error", re.compile(r"SyntaxError|Invalid input", re.IGNORECASE)),
]


def classify_error(error: Exception) -> str:
    """Classify a Neo4j error into a category based on its message."""
    msg = str(error)
    for category, pattern in _ERROR_PATTERNS:
        if pattern.search(msg):
            return category
    return "unknown"



class Verifier:
    """Verify corrupted samples by executing queries against Neo4j."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client
        self._graph_infos = load_graph_info()

    def verify(self, sample: CorruptedSample) -> VerificationResult:
        ctype = CorruptionType(sample.corruption.corruption_type)
        if ctype == CorruptionType.U5:
            verdict = self._verify_u5_temporal(sample)
        elif ctype in AMBIGUITY_TYPES:
            verdict = self._verify_ambiguity(sample)
        elif ctype == CorruptionType.U1:
            verdict = self._verify_u1_missing_property(sample)
        else:
            verdict = self._verify_unanswerability(sample)

        return VerificationResult(
            corruption_id=sample.corruption_id,
            corruption_type=ctype.value,
            graph=sample.original_graph,
            status=verdict.status,
            ambiguity=verdict if ctype in AMBIGUITY_TYPES else None,
            unanswerability=verdict if ctype not in AMBIGUITY_TYPES else None,
        )

    def verify_batch(self, samples: list[CorruptedSample]) -> list[VerificationResult]:
        from tqdm import tqdm
        return [self.verify(s) for s in tqdm(samples, desc="Verifying")]

    # ------------------------------------------------------------------
    # Ambiguity: execute each valid_cyphers variant, check results differ
    # ------------------------------------------------------------------

    def _verify_ambiguity(self, sample: CorruptedSample) -> AmbiguityVerdict:
        if len(sample.valid_cyphers) < 2:
            return AmbiguityVerdict(
                status=VerificationStatus.FAIL,
                results_per_cypher={},
                all_results_differ=False,
                detail="Less than 2 valid Cypher variants provided.",
            )

        results_per_cypher: dict[str, list[dict[str, Any]]] = {}
        for cypher in sample.valid_cyphers:
            try:
                rows = self._client.run_query(cypher, sample.original_graph)
                results_per_cypher[cypher] = rows
            except Exception as e:
                return AmbiguityVerdict(
                    status=VerificationStatus.ERROR,
                    results_per_cypher=results_per_cypher,
                    all_results_differ=False,
                    detail=f"Query execution failed: {e}\nQuery: {cypher}",
                )

        fingerprints = {
            cypher: result_fingerprint(rows)
            for cypher, rows in results_per_cypher.items()
        }
        cyphers = list(fingerprints.keys())
        same_pairs: list[tuple[str, str]] = []
        for i in range(len(cyphers)):
            for j in range(i + 1, len(cyphers)):
                if fingerprints[cyphers[i]] == fingerprints[cyphers[j]]:
                    same_pairs.append((cyphers[i], cyphers[j]))

        all_differ = len(same_pairs) == 0
        if all_differ:
            status = VerificationStatus.PASS
            detail = f"All {len(cyphers)} variants produce different results."
        else:
            status = VerificationStatus.FAIL
            detail = f"{len(same_pairs)} pair(s) returned identical results."

        return AmbiguityVerdict(
            status=status,
            results_per_cypher=results_per_cypher,
            all_results_differ=all_differ,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Unanswerability: build wrong query, confirm it errors or returns empty
    # ------------------------------------------------------------------

    def _verify_unanswerability(self, sample: CorruptedSample) -> UnanswerabilityVerdict:
        """Build the wrong Cypher (gold + corrupted element) and execute it.

        A valid unanswerable corruption should cause one of:
        - A specific Neo4j error (missing property/label/relationship)
        - Empty result set (corrupted element matches nothing)
        - Null/zero-only results (aggregations over missing data)
        """
        query = _build_wrong_query(sample)
        if query is None:
            return UnanswerabilityVerdict(
                status=VerificationStatus.ERROR,
                query_executed="",
                returned_rows=0,
                detail="Could not build verification query from sample.",
            )

        try:
            rows = self._client.run_query(query, sample.original_graph)
        except (CypherSyntaxError, ClientError) as e:
            category = classify_error(e)
            is_specific = category not in ("unknown", "syntax_error")
            return UnanswerabilityVerdict(
                status=VerificationStatus.PASS if is_specific else VerificationStatus.FAIL,
                query_executed=query,
                returned_rows=0,
                error_message=str(e),
                error_category=category,
                detail=f"Query failed with {category} error." if is_specific
                else f"Query failed with non-specific error: {category}.",
            )
        except Exception as e:
            return UnanswerabilityVerdict(
                status=VerificationStatus.ERROR,
                query_executed=query,
                returned_rows=0,
                error_message=str(e),
                detail=f"Unexpected error: {e}",
            )

        if len(rows) == 0:
            return UnanswerabilityVerdict(
                status=VerificationStatus.PASS,
                query_executed=query,
                returned_rows=0,
                detail="Query returned 0 rows (corrupted element matches nothing).",
            )

        if _all_null_or_zero(rows):
            return UnanswerabilityVerdict(
                status=VerificationStatus.PASS,
                query_executed=query,
                returned_rows=len(rows),
                detail="Query returned only null/zero values.",
            )

        return UnanswerabilityVerdict(
            status=VerificationStatus.FAIL,
            query_executed=query,
            returned_rows=len(rows),
            detail=f"Query returned {len(rows)} non-empty row(s) — corruption may not make the query unanswerable.",
        )


    def _verify_u1_missing_property(self, sample: CorruptedSample) -> UnanswerabilityVerdict:
        """U1: verify the fake property doesn't exist on any node in the graph.

        Runs a targeted query: MATCH (n) WHERE n.fake_prop IS NOT NULL RETURN count(n).
        If count is 0, the property truly doesn't exist → PASS.
        """
        fake_prop = sample.corruption.corrupted_element
        query = f"MATCH (n) WHERE n.`{fake_prop}` IS NOT NULL RETURN count(n) AS cnt LIMIT 1"

        try:
            rows = self._client.run_query(query, sample.original_graph)
        except Exception as e:
            return UnanswerabilityVerdict(
                status=VerificationStatus.ERROR,
                query_executed=query,
                returned_rows=0,
                error_message=str(e),
                detail=f"Unexpected error: {e}",
            )

        cnt = rows[0]["cnt"] if rows else 0
        if cnt == 0:
            return UnanswerabilityVerdict(
                status=VerificationStatus.PASS,
                query_executed=query,
                returned_rows=0,
                detail=f"Property '{fake_prop}' does not exist on any node — unanswerable.",
            )
        return UnanswerabilityVerdict(
            status=VerificationStatus.FAIL,
            query_executed=query,
            returned_rows=cnt,
            detail=f"Property '{fake_prop}' exists on {cnt} node(s) — not truly missing.",
        )

    def _verify_u5_temporal(self, sample: CorruptedSample) -> UnanswerabilityVerdict:
        """U5: verify the relation has no temporal properties in the schema.

        U5 corruptions ask temporal questions about relations that lack
        temporal data. The gold query is valid — the unanswerability is
        in the NL question / schema mismatch, not in the Cypher.
        """
        rel_label = sample.corruption.original_element
        graph_info = self._graph_infos.get(sample.original_graph)
        if graph_info is None:
            return UnanswerabilityVerdict(
                status=VerificationStatus.ERROR,
                query_executed="",
                returned_rows=0,
                detail=f"No graph_info found for '{sample.original_graph}'.",
            )

        for rel_info in graph_info.relations:
            if rel_info.label == rel_label:
                if not rel_info.is_time_sensitive:
                    return UnanswerabilityVerdict(
                        status=VerificationStatus.PASS,
                        query_executed="(schema check, no query executed)",
                        returned_rows=0,
                        detail=f"Relation '{rel_label}' is not time-sensitive in graph_info — temporal question is unanswerable.",
                    )
                else:
                    return UnanswerabilityVerdict(
                        status=VerificationStatus.FAIL,
                        query_executed="(schema check, no query executed)",
                        returned_rows=0,
                        detail=f"Relation '{rel_label}' IS time-sensitive — temporal question may be answerable.",
                    )

        return UnanswerabilityVerdict(
            status=VerificationStatus.ERROR,
            query_executed="",
            returned_rows=0,
            detail=f"Relation '{rel_label}' not found in graph_info.",
        )


def _build_wrong_query(sample: CorruptedSample) -> str | None:
    """Substitute the corrupted element into the gold Cypher to produce the wrong query."""
    ctype = CorruptionType(sample.corruption.corruption_type)
    original = sample.corruption.original_element
    corrupted = sample.corruption.corrupted_element
    query = sample.original_gold_cypher

    if ctype == CorruptionType.U4:
        return replace_property(query, original, corrupted)
    elif ctype == CorruptionType.U2:
        return replace_relation(query, original, corrupted)
    elif ctype == CorruptionType.U3:
        return replace_label(query, original, corrupted)
    # U5 is handled separately via schema check, not query execution.
    return None


def _all_null_or_zero(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for v in row.values():
            if v is not None and v != 0:
                return False
    return True
