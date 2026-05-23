"""Precision and recall evaluation for Cypher query lists."""

from __future__ import annotations

from cb_corruptions.verification.models import EvalResult
from cb_corruptions.verification.neo4j_client import Neo4jClient
from cb_corruptions.verification.utils import has_order_by, result_fingerprint


def evaluate(
    predictions: list[str],
    targets: list[str],
    graph_name: str,
    client: Neo4jClient,
) -> EvalResult:
    """Compute precision and recall for a single benchmark sample.

    Two Cypher queries are equal iff their executed result sets are identical
    after normalization.  Set-based deduplication is applied to both lists.

    Args:
        predictions: Cypher queries predicted by the model for this sample.
        targets: Gold Cypher queries for this sample.
        graph_name: Neo4j database to run all queries against.
        client: Connected Neo4jClient instance.

    Raises:
        ValueError: If targets is empty.
        Exception: If any target query fails to execute.
    """
    if not targets:
        raise ValueError("targets must not be empty")

    target_fps: set[str] = set()
    for query in targets:
        rows = client.run_query(query, graph_name)
        target_fps.add(result_fingerprint(rows, ordered=has_order_by(query)))

    pred_fps: set[str] = set()
    failed_predictions: list[str] = []
    for query in predictions:
        try:
            rows = client.run_query(query, graph_name)
            pred_fps.add(result_fingerprint(rows, ordered=has_order_by(query)))
        except Exception:
            failed_predictions.append(query)

    matched = pred_fps & target_fps
    n_matched = len(matched)
    n_pred_unique = len(pred_fps)
    n_target_unique = len(target_fps)

    precision = n_matched / n_pred_unique if n_pred_unique > 0 else 0.0
    recall = n_matched / n_target_unique if n_target_unique > 0 else 0.0

    return EvalResult(
        precision=precision,
        recall=recall,
        n_predictions=len(predictions),
        n_targets=len(targets),
        n_pred_unique=n_pred_unique,
        n_target_unique=n_target_unique,
        n_matched=n_matched,
        failed_predictions=failed_predictions,
    )
