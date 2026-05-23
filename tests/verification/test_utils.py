from __future__ import annotations

import math

import pytest

from cb_corruptions.verification.utils import has_order_by, result_fingerprint


# --- has_order_by ---

def test_has_order_by_detects_order_by():
    assert has_order_by("MATCH (n) RETURN n.name ORDER BY n.name ASC")

def test_has_order_by_false_without_clause():
    assert not has_order_by("MATCH (n) RETURN n.name")

def test_has_order_by_case_insensitive():
    assert has_order_by("MATCH (n) RETURN n.name order by n.name")

def test_has_order_by_with_whitespace():
    assert has_order_by("MATCH (n) RETURN n.name ORDER  BY n.name")


# --- result_fingerprint: basic ---

def test_fingerprint_basic_serialization():
    rows = [{"name": "Alice", "age": 30}]
    fp = result_fingerprint(rows)
    assert '"age": 30' in fp
    assert '"name": "Alice"' in fp

def test_fingerprint_sort_keys():
    # Same data, different key insertion order → same fingerprint
    rows1 = [{"name": "Alice", "age": 30}]
    rows2 = [{"age": 30, "name": "Alice"}]
    assert result_fingerprint(rows1) == result_fingerprint(rows2)

def test_fingerprint_rows_sorted_by_default():
    rows1 = [{"name": "Alice"}, {"name": "Bob"}]
    rows2 = [{"name": "Bob"}, {"name": "Alice"}]
    assert result_fingerprint(rows1) == result_fingerprint(rows2)

def test_fingerprint_ordered_preserves_row_order():
    rows1 = [{"name": "Alice"}, {"name": "Bob"}]
    rows2 = [{"name": "Bob"}, {"name": "Alice"}]
    assert result_fingerprint(rows1, ordered=True) != result_fingerprint(rows2, ordered=True)

def test_fingerprint_empty_rows():
    assert result_fingerprint([]) == "[]"


# --- result_fingerprint: numeric normalization ---

def test_fingerprint_int_and_whole_float_equal():
    rows_int = [{"count": 1}]
    rows_float = [{"count": 1.0}]
    assert result_fingerprint(rows_int) == result_fingerprint(rows_float)

def test_fingerprint_fractional_float_not_coerced():
    rows = [{"value": 1.5}]
    fp = result_fingerprint(rows)
    assert "1.5" in fp

def test_fingerprint_bool_not_coerced_to_int():
    rows_bool = [{"flag": True}]
    rows_int = [{"flag": 1}]
    # True serialises as JSON true, 1 as 1 — they must remain distinct
    assert result_fingerprint(rows_bool) != result_fingerprint(rows_int)

def test_fingerprint_nan_does_not_crash():
    rows = [{"value": float("nan")}]
    fp = result_fingerprint(rows)
    assert "nan" in fp

def test_fingerprint_infinity_does_not_crash():
    rows = [{"value": float("inf")}]
    fp = result_fingerprint(rows)
    assert "inf" in fp


# --- result_fingerprint: neo4j-like types ---

def test_fingerprint_temporal_type_via_isoformat():
    class FakeDateTime:
        def isoformat(self) -> str:
            return "2024-01-15T10:30:00"
    rows = [{"dt": FakeDateTime()}]
    fp = result_fingerprint(rows)
    assert "2024-01-15T10:30:00" in fp

def test_fingerprint_nested_dict_normalised():
    # Simulates a Neo4j Node returned as a mapping
    class FakeNode:
        def items(self):
            return {"name": "Alice", "age": 30}.items()
    rows = [{"node": FakeNode()}]
    fp = result_fingerprint(rows)
    assert "Alice" in fp
    assert "30" in fp

def test_fingerprint_list_valued_property_order_preserved():
    # List properties in Neo4j are ordered — inner order must not be sorted
    rows = [{"tags": ["b", "a", "c"]}]
    fp = result_fingerprint(rows)
    import json
    data = json.loads(fp)
    assert data[0]["tags"] == ["b", "a", "c"]
