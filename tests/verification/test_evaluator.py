from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cb_corruptions.verification.evaluator import evaluate
from cb_corruptions.verification.models import EvalResult
from cb_corruptions.verification.neo4j_client import Neo4jClient


def _client(return_map: dict[str, list]) -> MagicMock:
    """Build a mock Neo4jClient whose run_query returns rows by query string."""
    client = MagicMock(spec=Neo4jClient)
    def run_query(cypher: str, graph_name: str) -> list:
        return return_map[cypher]
    client.run_query.side_effect = run_query
    return client


# --- basic correctness ---

def test_evaluate_returns_eval_result():
    rows = [{"name": "Alice"}]
    client = _client({"PRED": rows, "TARGET": rows})
    result = evaluate(["PRED"], ["TARGET"], "g", client)
    assert isinstance(result, EvalResult)


def test_evaluate_perfect_match():
    rows = [{"name": "Alice"}]
    client = _client({"PRED": rows, "TARGET": rows})
    result = evaluate(["PRED"], ["TARGET"], "g", client)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.n_matched == 1


def test_evaluate_no_match():
    client = _client({
        "PRED": [{"name": "Alice"}],
        "TARGET": [{"name": "Bob"}],
    })
    result = evaluate(["PRED"], ["TARGET"], "g", client)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.n_matched == 0


def test_evaluate_partial_precision():
    rows_a = [{"name": "Alice"}]
    rows_b = [{"name": "Bob"}]
    client = _client({"PRED1": rows_a, "PRED2": rows_b, "TARGET": rows_a})
    result = evaluate(["PRED1", "PRED2"], ["TARGET"], "g", client)
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_evaluate_partial_recall():
    rows_a = [{"name": "Alice"}]
    rows_b = [{"name": "Bob"}]
    client = _client({"PRED": rows_a, "T1": rows_a, "T2": rows_b})
    result = evaluate(["PRED"], ["T1", "T2"], "g", client)
    assert result.precision == 1.0
    assert result.recall == 0.5


# --- counts ---

def test_evaluate_counts_raw_and_unique():
    rows = [{"name": "Alice"}]
    client = _client({"PRED": rows, "TARGET": rows})
    result = evaluate(["PRED", "PRED"], ["TARGET"], "g", client)
    assert result.n_predictions == 2
    assert result.n_pred_unique == 1
    assert result.n_targets == 1
    assert result.n_target_unique == 1


# --- failed predictions ---

def test_evaluate_failed_prediction_skipped():
    rows = [{"name": "Alice"}]

    def run_query(cypher: str, graph_name: str) -> list:
        if cypher == "BAD":
            raise RuntimeError("syntax error")
        return rows

    client = MagicMock(spec=Neo4jClient)
    client.run_query.side_effect = run_query

    result = evaluate(["BAD", "GOOD"], ["GOOD"], "g", client)
    assert "BAD" in result.failed_predictions
    assert result.n_predictions == 2
    assert result.precision == 1.0


def test_evaluate_all_predictions_fail_gives_zero():
    def run_query(cypher: str, graph_name: str) -> list:
        if cypher == "TARGET":
            return [{"name": "Alice"}]
        raise RuntimeError("bad")

    client = MagicMock(spec=Neo4jClient)
    client.run_query.side_effect = run_query

    result = evaluate(["BAD"], ["TARGET"], "g", client)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.n_pred_unique == 0


# --- failed targets ---

def test_evaluate_failed_target_raises():
    client = MagicMock(spec=Neo4jClient)
    client.run_query.side_effect = RuntimeError("bad target")
    with pytest.raises(RuntimeError, match="bad target"):
        evaluate(["PRED"], ["TARGET"], "g", client)


# --- edge cases ---

def test_evaluate_empty_predictions():
    client = _client({"TARGET": [{"name": "Alice"}]})
    result = evaluate([], ["TARGET"], "g", client)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.n_predictions == 0


def test_evaluate_empty_targets_raises():
    client = MagicMock(spec=Neo4jClient)
    with pytest.raises(ValueError):
        evaluate(["PRED"], [], "g", client)


# --- ORDER BY ---

def test_evaluate_different_order_by_not_matched():
    def run_query(cypher: str, graph_name: str) -> list:
        if "ASC" in cypher:
            return [{"name": "Alice"}, {"name": "Bob"}]
        return [{"name": "Bob"}, {"name": "Alice"}]

    client = MagicMock(spec=Neo4jClient)
    client.run_query.side_effect = run_query

    result = evaluate(
        ["MATCH (n) RETURN n.name ORDER BY n.name ASC"],
        ["MATCH (n) RETURN n.name ORDER BY n.name DESC"],
        "g",
        client,
    )
    assert result.n_matched == 0


def test_evaluate_same_order_by_matched():
    rows = [{"name": "Alice"}, {"name": "Bob"}]

    client = MagicMock(spec=Neo4jClient)
    client.run_query.return_value = rows

    result = evaluate(
        ["MATCH (n) RETURN n.name ORDER BY n.name ASC"],
        ["MATCH (m) RETURN m.name ORDER BY m.name ASC"],
        "g",
        client,
    )
    assert result.n_matched == 1


# --- graph_name forwarded ---

def test_evaluate_passes_graph_name_to_client():
    rows = [{"x": 1}]
    client = MagicMock(spec=Neo4jClient)
    client.run_query.return_value = rows
    evaluate(["PRED"], ["TARGET"], "my_graph", client)
    for call in client.run_query.call_args_list:
        assert call.args[1] == "my_graph"
