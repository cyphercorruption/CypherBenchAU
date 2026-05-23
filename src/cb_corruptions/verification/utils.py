"""Shared result normalization and fingerprinting for Cypher query comparison."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def has_order_by(cypher: str) -> bool:
    """Return True if the Cypher query contains an ORDER BY clause."""
    return bool(re.search(r"\bORDER\s+BY\b", cypher, re.IGNORECASE))


def _normalize_value(v: Any) -> Any:
    """Recursively convert Neo4j driver types to plain Python types."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        if v == int(v):
            return int(v)
        return v
    if isinstance(v, int):
        return v
    if hasattr(v, "items"):
        return {k: _normalize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    return v


def _neo4j_json_default(v: Any) -> Any:
    """json.dumps fallback for Neo4j types not handled by _normalize_value."""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if type(v).__module__.startswith("neo4j.spatial"):
        return {k: _normalize_value(val) for k, val in sorted(vars(v).items())}
    return str(v)


def result_fingerprint(rows: list[dict[str, Any]], ordered: bool = False) -> str:
    """Return a canonical JSON string representing the query result set.

    When ordered=False (default), rows are sorted so that result-set equality
    is independent of Neo4j's non-deterministic row order.  When ordered=True,
    row order is preserved — use this when the Cypher query contains ORDER BY
    so that different orderings produce different fingerprints.

    Normalisation applied (see docs/superpowers/specs/2026-05-05-benchmarking-eval-design.md
    for the full rationale):
    - sort_keys=True    → eliminates dict key-insertion-order differences
    - whole-number floats → int  → eliminates int/float type mismatches
    - bool preserved as bool  → not collapsed into int
    - nan/inf → str()   → prevents json.dumps ValueError crash
    - Neo4j temporal types (.isoformat())  → prevents TypeError crash
    - Neo4j spatial types (vars dict)      → prevents TypeError crash
    """
    normalized = [
        {k: _normalize_value(v) for k, v in row.items()}
        for row in rows
    ]
    if not ordered:
        try:
            normalized = sorted(
                normalized,
                key=lambda r: json.dumps(r, sort_keys=True, default=_neo4j_json_default),
            )
        except (TypeError, ValueError):
            pass
    return json.dumps(normalized, sort_keys=True, default=_neo4j_json_default)
