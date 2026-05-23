#!/usr/bin/env python3
"""Compute every paper-grade metric for the CypherBench-Corruptions benchmark.

Reads:
  - output/benchmark_v4/<model>/<mode>/*.json   (predictions + telemetry)
  - output/curated_200_v4/full_<graph>.json     (gold + corruption metadata)
  - output/query_cache/<graph>.jsonl            (cached query results)
  - Neo4j                                       (only for cyphers missing in cache)

Writes (to --out-dir, default output/paper_tables_v4_final/):
  - per_model_mode.csv             : main 18-row table (P/R/F1 value-only,
                                     U_acc, cost, latency, …)
  - per_corruption_type.csv        : (model, mode, corruption_type) breakdown
  - per_corrupted_element.csv      : finest grain — corrupted_element
  - shape_distribution.csv         : single / multi / empty per cell
  - confidence_intervals.csv       : bootstrap 95% CI on F1 (A*) and U_acc
  - mcnemar.csv                    : pairwise McNemar on per-sample success
  - per_graph_difficulty.csv       : avg F1 per (graph, model) and per graph
  - all_per_sample.jsonl           : raw per-sample evaluation (debugging)

Usage:
    uv run python scripts/evaluation/compute_all_metrics.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import sys
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb_corruptions.verification.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=False)

AMBIGUITY_TYPES = {"A1", "A2", "A3", "A4", "A5"}
UNANSWERABLE_TYPES = {"U1", "U2", "U3", "U4", "U5"}

ALL_GRAPHS = [
    "art", "biology", "company", "fictional_character", "flight_accident",
    "geography", "movie", "nba", "politics", "soccer", "terrorist_attack",
]


# ---------------------------------------------------------------------------
# Value-set fingerprint (ignores all column names / aliases)
# ---------------------------------------------------------------------------

def _row_value_bag(row: dict) -> tuple:
    return tuple(sorted(("null" if v is None else str(v)) for v in row.values()))


def _value_signature(rows: list[dict] | None) -> frozenset | None:
    if rows is None:
        return None
    return frozenset(_row_value_bag(r) for r in rows)


def _cypher_hash(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

class QueryCache:
    """Cache of executed (graph, cypher) → rows | error."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        # (graph, hash) -> (rows_or_None, error_or_None)
        self.entries: dict[tuple[str, str], tuple[list | None, str | None]] = {}
        self._load()

    def _load(self) -> None:
        for f in self.cache_dir.glob("*.jsonl"):
            if f.stem == "cache_index":
                continue
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                self.entries[(f.stem, r["hash"])] = (r.get("rows"), r.get("error"))

    def has(self, graph: str, cypher: str) -> bool:
        return (graph, _cypher_hash(cypher)) in self.entries

    def get(self, graph: str, cypher: str) -> tuple[list | None, str | None] | None:
        return self.entries.get((graph, _cypher_hash(cypher)))

    def append(self, graph: str, cypher: str,
               rows: list | None, error: str | None) -> None:
        h = _cypher_hash(cypher)
        self.entries[(graph, h)] = (rows, error)
        with open(self.cache_dir / f"{graph}.jsonl", "a") as f:
            f.write(json.dumps({"hash": h, "cypher": cypher,
                                "rows": rows, "error": error},
                               default=str) + "\n")


# ---------------------------------------------------------------------------
# Refresh cache for missing cyphers
# ---------------------------------------------------------------------------

def _collect_unique_cyphers(
    benchmark_root: Path,
    sample_meta: dict,
) -> dict[str, set[str]]:
    """Return {graph: {cypher, ...}} for all predicted + gold cyphers."""
    by_graph: dict[str, set[str]] = defaultdict(set)
    for model_dir in benchmark_root.iterdir():
        if not model_dir.is_dir():
            continue
        for mode_dir in model_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            for f in mode_dir.glob("*.json"):
                for it in json.load(open(f)):
                    g = it["graph"]
                    for c in it.get("predicted_cyphers") or []:
                        by_graph[g].add(c)
                    for c in it.get("gold_valid_cyphers") or []:
                        by_graph[g].add(c)
    return by_graph


def _refresh_cache(
    by_graph: dict[str, set[str]],
    cache: QueryCache,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    db_map: dict[str, str],
    timeout_s: float,
) -> None:
    """Execute cyphers missing from the cache and append results."""
    missing: list[tuple[str, str]] = []
    for g, cyphers in by_graph.items():
        for c in cyphers:
            if not cache.has(g, c):
                missing.append((g, c))
    if not missing:
        logger.info("Cache fully populated — nothing to refresh")
        return

    logger.info("Refreshing cache: %d missing (graph, cypher) pairs", len(missing))
    os.environ.setdefault("NEO4J_DATABASE_MAP", json.dumps(db_map))
    client = Neo4jClient(uri=neo4j_uri, username=neo4j_user, password=neo4j_password)
    try:
        for g, c in tqdm(missing, desc="neo4j", unit="q"):
            try:
                rows = client.run_query(c, g, timeout_s=timeout_s)
                cache.append(g, c, rows, None)
            except Exception as e:
                cache.append(g, c, None, str(e))
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _evaluate_sample(
    record: dict,
    sample_meta: dict,
    cache: QueryCache,
) -> dict:
    """Evaluate one benchmark record. Returns a flat dict of metrics."""
    cid = record["corruption_id"]
    ctype = record["corruption_type"]
    graph = record["graph"]
    preds = record.get("predicted_cyphers") or []
    golds = record.get("gold_valid_cyphers") or []
    is_A = ctype in AMBIGUITY_TYPES
    is_U = ctype in UNANSWERABLE_TYPES
    err = record.get("error")

    meta = sample_meta.get(cid, {})
    elem = meta.get("corrupted_element")

    # --- execution / signatures (only meaningful for A*) ---
    pred_sigs: list[frozenset] = []
    n_pred_failed = 0
    n_pred_empty = 0
    for q in preds:
        cell = cache.get(graph, q)
        if cell is None or cell[1] is not None or cell[0] is None:
            n_pred_failed += 1
            continue
        sig = _value_signature(cell[0])
        pred_sigs.append(sig)
        if not sig:
            n_pred_empty += 1

    gold_sigs: list[frozenset] = []
    for q in golds:
        cell = cache.get(graph, q)
        if cell is None or cell[1] is not None or cell[0] is None:
            continue
        sig = _value_signature(cell[0])
        if sig:
            gold_sigs.append(sig)

    # value-only set comparisons (empty sigs already excluded above for gold,
    # so an empty pred sig cannot accidentally match an empty gold)
    pred_set = {s for s in pred_sigs if s}
    gold_set = set(gold_sigs)

    strict = bool(pred_set & gold_set)
    # forward = at least one pred contains a gold (gold ⊆ pred), model verbose
    forward = any(g.issubset(p) for p in pred_set for g in gold_set)
    # inverse = at least one pred is contained by a gold (pred ⊆ gold), partial
    inverse = any(p.issubset(g) for p in pred_set for g in gold_set)
    either = strict or forward or inverse

    n_pred_unique = len(pred_set)
    n_gold_unique = len(gold_set)
    matched = len(pred_set & gold_set)
    precision = matched / n_pred_unique if n_pred_unique else None
    recall = matched / n_gold_unique if n_gold_unique else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))

    # --- abstention (U*) ---
    u_correct = (len(preds) == 0) if is_U else None

    # --- output shape ---
    if err:
        shape = "error"
    elif len(preds) == 0:
        shape = "empty"
    elif len(preds) == 1:
        shape = "single"
    else:
        shape = "multi"

    # cost / telemetry
    cost = record.get("cost_openrouter")
    pt = record.get("prompt_tokens")
    ct = record.get("completion_tokens")
    rt = record.get("reasoning_tokens")
    lat = record.get("latency_ms")

    return {
        "corruption_id": cid,
        "corruption_type": ctype,
        "corrupted_element": elem,
        "graph": graph,
        "model": record["model"],
        "mode": record["mode"],
        "is_A": is_A,
        "is_U": is_U,
        "n_preds": len(preds),
        "n_golds": len(golds),
        "n_pred_unique": n_pred_unique,
        "n_gold_unique": n_gold_unique,
        "n_pred_failed_execution": n_pred_failed,
        "n_pred_empty_result": n_pred_empty,
        "matched": matched,
        "precision_A": precision,
        "recall_A": recall,
        "f1_A": f1,
        "strict_value": strict,
        "forward_value": forward,
        "inverse_value": inverse,
        "either_value": either,
        "u_correct": u_correct,
        "shape": shape,
        "error": err,
        "cost": cost,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "latency_ms": lat,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_cell(samples: list[dict]) -> dict:
    """Aggregate metrics for one (model, mode) cell (or sub-cell)."""
    n_total = len(samples)
    A_samples = [s for s in samples if s["is_A"]]
    U_samples = [s for s in samples if s["is_U"]]
    n_A = len(A_samples)
    n_U = len(U_samples)

    # A* metrics — macro avg over A* samples
    p_vals = [s["precision_A"] for s in A_samples if s["precision_A"] is not None]
    r_vals = [s["recall_A"]    for s in A_samples if s["recall_A"]    is not None]
    f_vals = [s["f1_A"]         for s in A_samples if s["f1_A"]        is not None]

    macro_precision = statistics.mean(p_vals) if p_vals else 0.0
    macro_recall    = statistics.mean(r_vals) if r_vals else 0.0
    macro_f1        = statistics.mean(f_vals) if f_vals else 0.0

    # Micro avg (treats every prediction equally rather than every sample)
    sum_matched = sum(s["matched"] for s in A_samples)
    sum_pred_u  = sum(s["n_pred_unique"] for s in A_samples)
    sum_gold_u  = sum(s["n_gold_unique"] for s in A_samples)
    micro_precision = sum_matched / sum_pred_u if sum_pred_u else 0.0
    micro_recall    = sum_matched / sum_gold_u if sum_gold_u else 0.0
    micro_f1 = (2 * micro_precision * micro_recall /
                (micro_precision + micro_recall)
                if (micro_precision + micro_recall) else 0.0)

    strict_rate  = (sum(1 for s in A_samples if s["strict_value"])  / n_A) if n_A else 0.0
    forward_rate = (sum(1 for s in A_samples if s["forward_value"]) / n_A) if n_A else 0.0
    inverse_rate = (sum(1 for s in A_samples if s["inverse_value"]) / n_A) if n_A else 0.0
    either_rate  = (sum(1 for s in A_samples if s["either_value"])  / n_A) if n_A else 0.0

    # Multi-rate on A*: model emitted ≥2 predictions
    multi_rate_A = (sum(1 for s in A_samples if s["n_preds"] >= 2) / n_A) if n_A else 0.0
    # False-abstention rate on A*: model returned [] for an A* sample
    false_abst_A = (sum(1 for s in A_samples if s["n_preds"] == 0 and not s["error"]) / n_A) if n_A else 0.0

    # U_acc — both numerator and denominator must come from U* samples
    u_acc = (sum(1 for s in U_samples if s["u_correct"]) / n_U) if n_U else 0.0
    # Over-eager rate on U*
    over_eager_U = (sum(1 for s in U_samples if s["n_preds"] > 0 and not s["error"]) / n_U) if n_U else 0.0

    # Combined score = 0.5*U_acc + 0.5*A_either
    combined = 0.5 * u_acc + 0.5 * either_rate

    # Execution / robustness
    n_err = sum(1 for s in samples if s["error"])
    err_rate = n_err / n_total if n_total else 0.0
    # crash rate on PREDICTED queries (over all preds emitted)
    sum_preds_emitted = sum(s["n_preds"] for s in samples)
    sum_preds_failed = sum(s["n_pred_failed_execution"] for s in samples)
    crash_rate = sum_preds_failed / sum_preds_emitted if sum_preds_emitted else 0.0
    sum_preds_empty  = sum(s["n_pred_empty_result"] for s in samples)
    empty_result_rate = sum_preds_empty / sum_preds_emitted if sum_preds_emitted else 0.0

    # Output shape distribution
    shape_ct = Counter(s["shape"] for s in samples)
    shape_pct = {k: shape_ct[k] / n_total for k in ("single", "multi", "empty", "error")}

    # Cost + tokens
    costs = [s["cost"] for s in samples if s["cost"] is not None]
    cost_total = sum(costs)
    cost_mean = cost_total / len(costs) if costs else 0.0
    # cost per "correct answer" — count = (U correct) + (A either)
    n_correct = (sum(1 for s in U_samples if s["u_correct"])
                 + sum(1 for s in A_samples if s["either_value"]))
    cost_per_correct = cost_total / n_correct if n_correct else float("inf")

    lats = [s["latency_ms"] for s in samples if s["latency_ms"]]
    p50 = statistics.median(lats) if lats else 0.0
    p95 = (sorted(lats)[int(0.95 * (len(lats) - 1))] if lats else 0.0)
    avg_lat = statistics.mean(lats) if lats else 0.0

    pt = sum(s["prompt_tokens"] or 0 for s in samples)
    ct = sum(s["completion_tokens"] or 0 for s in samples)
    rt = sum(s["reasoning_tokens"] or 0 for s in samples)

    return {
        "n_total": n_total, "n_A": n_A, "n_U": n_U,
        # A* macro
        "macro_precision_A": macro_precision,
        "macro_recall_A": macro_recall,
        "macro_f1_A": macro_f1,
        # A* micro
        "micro_precision_A": micro_precision,
        "micro_recall_A": micro_recall,
        "micro_f1_A": micro_f1,
        # value-only flags
        "strict_value": strict_rate,
        "forward_value": forward_rate,
        "inverse_value": inverse_rate,
        "either_value": either_rate,
        "multi_rate_A": multi_rate_A,
        "false_abst_A": false_abst_A,
        # U*
        "u_acc": u_acc,
        "over_eager_U": over_eager_U,
        # combined
        "combined": combined,
        # robustness
        "err_rate": err_rate,
        "crash_rate": crash_rate,
        "empty_result_rate": empty_result_rate,
        # shape
        "shape_single": shape_pct["single"],
        "shape_multi": shape_pct["multi"],
        "shape_empty": shape_pct["empty"],
        "shape_error": shape_pct["error"],
        # cost / tokens
        "cost_total": cost_total,
        "cost_mean": cost_mean,
        "cost_per_correct": cost_per_correct,
        "n_correct": n_correct,
        "tokens_prompt": pt,
        "tokens_completion": ct,
        "tokens_reasoning": rt,
        # latency
        "latency_mean_ms": avg_lat,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_mean_ci(values: list[float], n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap (mean, lo, hi) at (1-alpha) CI on a list of per-sample values."""
    import random
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (sum(values) / n, lo, hi)


def _mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """McNemar exact p-value for paired binary outcomes (a vs b).

    Returns two-sided p-value via binomial test on discordant pairs.
    """
    b10 = sum(1 for x, y in zip(a, b) if x and not y)  # a right, b wrong
    b01 = sum(1 for x, y in zip(a, b) if y and not x)  # b right, a wrong
    n = b10 + b01
    if n == 0:
        return 1.0
    # two-sided binomial test, p=0.5
    from math import comb
    k = min(b10, b01)
    p = 2 * sum(comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(p, 1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def run(
    benchmark_dir: Annotated[Path, typer.Option()] = Path("output/benchmark_v4"),
    input_dir:     Annotated[Path, typer.Option()] = Path("output/curated_200_v4"),
    cache_dir:     Annotated[Path, typer.Option()] = Path("output/query_cache"),
    out_dir:       Annotated[Path, typer.Option()] = Path("output/paper_tables_v4_final"),
    skip_refresh:  Annotated[bool, typer.Option("--skip-refresh",
                    help="Skip Neo4j cache refresh (assumes all preds in cache)")] = False,
    neo4j_uri:     Annotated[str, typer.Option()] = os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user:    Annotated[str, typer.Option()] = os.getenv("NEO4J_USERNAME", "neo4j"),
    neo4j_pwd:     Annotated[str, typer.Option()] = os.getenv("NEO4J_PASSWORD", "password"),
    timeout_s:     Annotated[float, typer.Option()] = 60.0,
    n_boot:        Annotated[int, typer.Option(help="Bootstrap iterations")] = 2000,
    verbose:       Annotated[bool, typer.Option("-v")] = False,
) -> None:
    """Compute all metrics, refresh cache for missing cyphers, write CSVs."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load sample metadata (corruption_type, corrupted_element, graph)
    sample_meta: dict[str, dict] = {}
    for g in ALL_GRAPHS:
        fp = input_dir / f"full_{g}.json"
        if not fp.exists():
            continue
        for s in json.load(open(fp)):
            sample_meta[s["corruption_id"]] = {
                "corruption_type": s["corruption"]["corruption_type"],
                "corrupted_element": s["corruption"]["corrupted_element"],
                "original_element": s["corruption"]["original_element"],
                "graph": s["original_graph"],
            }
    typer.echo(f"Loaded metadata for {len(sample_meta)} corruption_ids")

    # 2. Load cache
    cache = QueryCache(cache_dir)
    typer.echo(f"Cache has {len(cache.entries)} (graph, hash) entries")

    # 3. Identify and execute missing cyphers
    if not skip_refresh:
        by_graph = _collect_unique_cyphers(benchmark_dir, sample_meta)
        n_unique = sum(len(v) for v in by_graph.values())
        typer.echo(f"Unique cyphers across all benchmarks: {n_unique}")
        db_map_env = os.environ.get("NEO4J_DATABASE_MAP")
        db_map = json.loads(db_map_env) if db_map_env else {g: "neo4j" for g in ALL_GRAPHS}
        _refresh_cache(by_graph, cache, neo4j_uri, neo4j_user, neo4j_pwd,
                       db_map, timeout_s)

    # 4. Evaluate every sample
    per_sample: list[dict] = []
    n_files = 0
    for model_dir in sorted(benchmark_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for mode_dir in sorted(model_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for f in sorted(mode_dir.glob("*.json")):
                n_files += 1
                for it in json.load(open(f)):
                    per_sample.append(_evaluate_sample(it, sample_meta, cache))
    typer.echo(f"Evaluated {len(per_sample)} samples from {n_files} files")

    # Dump per-sample JSONL for debugging
    with open(out_dir / "all_per_sample.jsonl", "w") as fh:
        for s in per_sample:
            d = {k: (list(v) if isinstance(v, frozenset) else v) for k, v in s.items()}
            fh.write(json.dumps(d, default=str) + "\n")

    # 5. Aggregate per (model, mode)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in per_sample:
        by_cell[(s["model"], s["mode"])].append(s)

    rows_main = []
    for (model, mode), ss in sorted(by_cell.items()):
        agg = _agg_cell(ss)
        rows_main.append({"model": model, "mode": mode, **agg})
    _write_csv(out_dir / "per_model_mode.csv", rows_main)

    # 6. Per (model, mode, corruption_type)
    rows_type = []
    by_type: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in per_sample:
        by_type[(s["model"], s["mode"], s["corruption_type"])].append(s)
    for (model, mode, ctype), ss in sorted(by_type.items()):
        agg = _agg_cell(ss)
        rows_type.append({"model": model, "mode": mode, "corruption_type": ctype, **agg})
    _write_csv(out_dir / "per_corruption_type.csv", rows_type)

    # 7. Per (model, mode, corrupted_element)
    rows_elem = []
    by_elem: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for s in per_sample:
        by_elem[(s["model"], s["mode"], s["corruption_type"],
                 s["corrupted_element"] or "")].append(s)
    for (model, mode, ctype, elem), ss in sorted(by_elem.items()):
        agg = _agg_cell(ss)
        rows_elem.append({"model": model, "mode": mode,
                          "corruption_type": ctype,
                          "corrupted_element": elem, **agg})
    _write_csv(out_dir / "per_corrupted_element.csv", rows_elem)

    # 8. Per (model, mode, graph)
    rows_graph = []
    by_graph_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in per_sample:
        by_graph_cell[(s["model"], s["mode"], s["graph"])].append(s)
    for (model, mode, g), ss in sorted(by_graph_cell.items()):
        agg = _agg_cell(ss)
        rows_graph.append({"model": model, "mode": mode, "graph": g, **agg})
    _write_csv(out_dir / "per_graph_difficulty.csv", rows_graph)

    # 9. Bootstrap CIs on per (model, mode) macro-F1 (A*) and U_acc
    rows_ci = []
    for (model, mode), ss in sorted(by_cell.items()):
        A = [s for s in ss if s["is_A"]]
        U = [s for s in ss if s["is_U"]]
        f1_vals = [s["f1_A"] for s in A if s["f1_A"] is not None]
        u_vals = [1.0 if s["u_correct"] else 0.0 for s in U]
        combined_vals = [s["either_value"] for s in A] + [s["u_correct"] for s in U]
        combined_vals = [1.0 if v else 0.0 for v in combined_vals]
        f1_mean, f1_lo, f1_hi = _bootstrap_mean_ci(f1_vals, n_boot=n_boot)
        u_mean,  u_lo,  u_hi  = _bootstrap_mean_ci(u_vals,  n_boot=n_boot)
        c_mean,  c_lo,  c_hi  = _bootstrap_mean_ci(combined_vals, n_boot=n_boot)
        rows_ci.append({
            "model": model, "mode": mode,
            "f1_A_mean": f1_mean, "f1_A_ci_lo": f1_lo, "f1_A_ci_hi": f1_hi,
            "u_acc_mean": u_mean, "u_acc_ci_lo": u_lo, "u_acc_ci_hi": u_hi,
            "combined_mean": c_mean, "combined_ci_lo": c_lo, "combined_ci_hi": c_hi,
        })
    _write_csv(out_dir / "confidence_intervals.csv", rows_ci)

    # 10. Pairwise McNemar on combined-success (per-sample 0/1) — within same mode
    # For each pair (model_a, model_b) at the same mode, paired on corruption_id
    # the "success" indicator is: U_correct for U*, either_value for A*
    def _success_vec(samples):
        by_cid = {}
        for s in samples:
            ok = s["u_correct"] if s["is_U"] else s["either_value"]
            by_cid[s["corruption_id"]] = bool(ok)
        return by_cid

    rows_mc = []
    cells = sorted(by_cell.keys())
    for i, k1 in enumerate(cells):
        for k2 in cells[i + 1:]:
            if k1[1] != k2[1]:  # only same mode
                continue
            sv1 = _success_vec(by_cell[k1])
            sv2 = _success_vec(by_cell[k2])
            common = sorted(set(sv1) & set(sv2))
            a = [sv1[c] for c in common]
            b = [sv2[c] for c in common]
            p = _mcnemar_p(a, b)
            wins_a = sum(1 for x, y in zip(a, b) if x and not y)
            wins_b = sum(1 for x, y in zip(a, b) if y and not x)
            ties   = sum(1 for x, y in zip(a, b) if x == y)
            rows_mc.append({
                "mode": k1[1],
                "model_a": k1[0], "model_b": k2[0],
                "n_compared": len(common),
                "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                "mcnemar_p": p,
            })
    _write_csv(out_dir / "mcnemar.csv", rows_mc)

    # 11. Naive vs aware delta per model
    rows_delta = []
    by_model = defaultdict(dict)
    for r in rows_main:
        by_model[r["model"]][r["mode"]] = r
    for model, modes in sorted(by_model.items()):
        if "naive" not in modes or "aware" not in modes:
            continue
        n = modes["naive"]; a = modes["aware"]
        rows_delta.append({
            "model": model,
            "delta_either_value":  a["either_value"]  - n["either_value"],
            "delta_u_acc":         a["u_acc"]         - n["u_acc"],
            "delta_combined":      a["combined"]      - n["combined"],
            "delta_macro_f1":      a["macro_f1_A"]    - n["macro_f1_A"],
            "delta_false_abst_A":  a["false_abst_A"]  - n["false_abst_A"],
            "delta_over_eager_U":  a["over_eager_U"]  - n["over_eager_U"],
            "delta_cost_total":    a["cost_total"]    - n["cost_total"],
        })
    _write_csv(out_dir / "naive_vs_aware_delta.csv", rows_delta)

    typer.echo(f"\nAll tables written to {out_dir}/")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    # Stable column order from first row
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    app()
