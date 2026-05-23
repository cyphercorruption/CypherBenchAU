#!/usr/bin/env python3
"""Build the v4 evaluation set: merge v3 evaluations (minus bad samples) with
fresh evaluations on the 43 new samples.

Workflow:
  1. Collect all unique predicted_cyphers for the 43 new samples from
     output/benchmark_v4_new/. Identify which are missing from
     output/query_cache/ and execute them on Neo4j (graph-major).
  2. From the cache, compute strict + soft + value-only metrics for each
     (model, mode, new_sample).
  3. Build the v4 outputs:
     - output/evaluation_v4/<model>/<mode>.jsonl
     - output/soft_eval_v4/<model>/<mode>.jsonl
     - output/value_eval_v4/<model>/<mode>.jsonl
     by taking v3 records (minus bad_ids) + the new ones.

Usage:
    uv run python scripts/evaluation/eval_v4.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_graph import ALL_GRAPHS, load_graph  # noqa: E402

from cb_corruptions.verification import evaluate
from cb_corruptions.verification.neo4j_client import Neo4jClient
from cb_corruptions.verification.utils import has_order_by, result_fingerprint

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=False)

GRAPH_ORDER = [
    "terrorist_attack", "flight_accident", "nba", "fictional_character",
    "company", "soccer", "geography", "movie", "politics", "art", "biology",
]


def _default_db_map() -> str:
    return json.dumps({g: "neo4j" for g in ALL_GRAPHS})


def _h(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_dir: Path) -> dict[tuple[str, str], tuple]:
    cache: dict[tuple[str, str], tuple] = {}
    for f in cache_dir.glob("*.jsonl"):
        if f.stem == "cache_index":
            continue
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cache[(f.stem, r["hash"])] = (r.get("rows"), r.get("error"))
    return cache


def _append_to_cache(cache_dir: Path, graph: str, hsh: str, cypher: str,
                     rows: list | None, error: str | None) -> None:
    with open(cache_dir / f"{graph}.jsonl", "a") as f:
        f.write(json.dumps({"hash": hsh, "cypher": cypher,
                            "rows": rows, "error": error},
                           default=str) + "\n")


# ----- metric helpers (mirrors of the originals, reading from cache) -----

def _row_to_value_bag(row: dict) -> tuple:
    return tuple(sorted(("null" if v is None else str(v)) for v in row.values()))


def _result_signature_set(rows: list[dict]) -> frozenset:
    return frozenset(_row_to_value_bag(r) for r in rows)


def _strip_alias_keys(rows):
    return [
        {k.split(".", 1)[1] if "." in k else k: v for k, v in row.items()}
        for row in rows
    ]


def _row_contains(super_row, sub_row) -> bool:
    for k, v in sub_row.items():
        if k not in super_row:
            return False
        if str(super_row[k]) != str(v):
            return False
    return True


def _result_contains(super_rows, sub_rows) -> bool:
    if not sub_rows:
        return False
    for sr in sub_rows:
        if not any(_row_contains(pr, sr) for pr in super_rows):
            return False
    return True


def _strict_eval_for_sample(record, graph, cache):
    """Mirrors verification/evaluator.py evaluate() but reads from cache."""
    targets = record.get("gold_valid_cyphers") or []
    preds = record.get("predicted_cyphers") or []
    if not targets:
        return None  # only A* eligible

    target_fps = set()
    for q in targets:
        cell = cache.get((graph, _h(q)))
        if cell is None or cell[1] is not None:
            continue
        target_fps.add(result_fingerprint(cell[0], ordered=has_order_by(q)))

    pred_fps = set()
    failed = []
    for q in preds:
        cell = cache.get((graph, _h(q)))
        if cell is None or cell[1] is not None:
            failed.append(q)
            continue
        pred_fps.add(result_fingerprint(cell[0], ordered=has_order_by(q)))

    matched = pred_fps & target_fps
    n_matched = len(matched)
    n_pred_unique = len(pred_fps)
    n_target_unique = len(target_fps)
    precision = n_matched / n_pred_unique if n_pred_unique else 0.0
    recall = n_matched / n_target_unique if n_target_unique else 0.0
    return {
        "precision": precision, "recall": recall,
        "n_predictions": len(preds), "n_targets": len(targets),
        "n_pred_unique": n_pred_unique, "n_target_unique": n_target_unique,
        "n_matched": n_matched, "failed_predictions": failed,
    }


def _soft_eval_for_sample(record, graph, cache):
    """Mirrors check_soft_match_batch (with alias-stripping)."""
    targets = record.get("gold_valid_cyphers") or []
    preds = record.get("predicted_cyphers") or []
    gold_rows = []
    for q in targets:
        cell = cache.get((graph, _h(q)))
        if cell and cell[1] is None and cell[0] is not None:
            gold_rows.append(cell[0])
    pred_rows = []
    n_failed_pred = 0
    for q in preds:
        cell = cache.get((graph, _h(q)))
        if cell and cell[1] is None and cell[0] is not None:
            pred_rows.append(cell[0])
        else:
            n_failed_pred += 1

    # strict-with-alias
    strict = False
    if pred_rows and gold_rows:
        # Use result_fingerprint with alias keys (default)
        gold_fps = {result_fingerprint(r) for r in gold_rows}
        pred_fps = {result_fingerprint(r) for r in pred_rows}
        strict = bool(gold_fps & pred_fps)

    gs = [_strip_alias_keys(r) for r in gold_rows]
    ps = [_strip_alias_keys(r) for r in pred_rows]
    forward = any(_result_contains(p, g) for p in ps if p for g in gs if g)
    inverse = any(_result_contains(g, p) for p in ps if p for g in gs if g)

    return {
        "n_pred_queries": len(preds), "n_gold_queries": len(targets),
        "n_failed_pred_queries": n_failed_pred,
        "n_failed_gold_queries": sum(1 for r in gold_rows if r is None),
        "strict_match": strict, "forward_match": forward,
        "inverse_match": inverse, "either_match": strict or forward or inverse,
    }


def _value_eval_for_sample(record, graph, cache):
    """Mirrors value_only_eval (no key info, with empty exclusion)."""
    targets = record.get("gold_valid_cyphers") or []
    preds = record.get("predicted_cyphers") or []
    gold_sigs = []
    for q in targets:
        cell = cache.get((graph, _h(q)))
        if cell and cell[1] is None and cell[0] is not None:
            gold_sigs.append(_result_signature_set(cell[0]))
    pred_sigs = []
    n_failed_pred = 0
    for q in preds:
        cell = cache.get((graph, _h(q)))
        if cell and cell[1] is None and cell[0] is not None:
            pred_sigs.append(_result_signature_set(cell[0]))
        else:
            n_failed_pred += 1
    # exclude empties
    gold_ne = [s for s in gold_sigs if s]
    pred_ne = [s for s in pred_sigs if s]

    strict = forward = inverse = False
    for ps in pred_ne:
        for gs in gold_ne:
            if ps == gs:
                strict = True
            if gs.issubset(ps):
                forward = True
            if ps.issubset(gs):
                inverse = True
    return {
        "n_pred": len(preds), "n_pred_valid": len(pred_sigs),
        "n_pred_nonempty": len(pred_ne),
        "n_gold": len(targets), "n_gold_nonempty": len(gold_ne),
        "n_failed_pred_queries": n_failed_pred,
        "strict_value": strict, "forward_value": forward,
        "inverse_value": inverse, "either_value": strict or forward or inverse,
    }


def _unanswerable_eval(record):
    preds = record.get("predicted_cyphers") or []
    return {"kind": "unanswerable", "predicted_correctly": len(preds) == 0,
            "n_predictions": len(preds)}


@app.command()
def run(
    benchmark_dir: Annotated[Path, typer.Option()] = Path("output/benchmark_v4"),
    new_bench_dir: Annotated[Path, typer.Option()] = Path("output/benchmark_v4_new"),
    cache_dir: Annotated[Path, typer.Option()] = Path("output/query_cache"),
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    timeout_s: Annotated[float, typer.Option()] = 60.0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    # 1) Collect new pred cyphers needing execution
    new_ids = set(open("output/curated_200_v4/_new_corruption_ids.txt").read().split())
    bad_ids = set(json.load(open("output/curated_200_v4/_replacement_plan.json"))["bad_ids_dropped"])
    typer.echo(f"new ids: {len(new_ids)}, bad ids to drop from v3: {len(bad_ids)}")

    cache = _load_cache(cache_dir)
    todo: dict[str, dict[str, str]] = defaultdict(dict)
    new_records_count = 0
    for f in new_bench_dir.glob("*/*/*.json"):
        for r in json.loads(f.read_text()):
            new_records_count += 1
            g = r["graph"]
            for q in r.get("predicted_cyphers") or []:
                if (g, _h(q)) not in cache:
                    todo[g][_h(q)] = q
    typer.echo(f"new benchmark records: {new_records_count}")
    typer.echo(f"pred cyphers to execute: {sum(len(v) for v in todo.values())}")

    # 2) Execute missing pred cyphers
    if todo:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        client = Neo4jClient(uri=uri, username=username, password=password)
        try:
            driver.verify_connectivity()
        except Exception as e:
            typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
            raise typer.Exit(1)
        for g in GRAPH_ORDER:
            items = todo.get(g)
            if not items:
                continue
            typer.echo(f"\n[{g}] loading + executing {len(items)} cyphers...")
            load_graph(driver.session, g, database="neo4j")
            for i, (hsh, q) in enumerate(items.items(), 1):
                try:
                    rows = client.run_query(q, g, timeout_s=timeout_s)
                    cache[(g, hsh)] = (rows, None)
                    _append_to_cache(cache_dir, g, hsh, q, rows, None)
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    cache[(g, hsh)] = (None, err)
                    _append_to_cache(cache_dir, g, hsh, q, None, err)
                if i % 50 == 0 or i == len(items):
                    typer.echo(f"  {g}: {i}/{len(items)}")
        driver.close()
        client.close()

    # 3) For each (model, mode), compute v4 eval files
    eval_v3_dir = Path("output/evaluation_v3")
    soft_v3_dir = Path("output/soft_eval_v3")
    value_v3_dir = Path("output/value_eval_v3")
    eval_v4_dir = Path("output/evaluation_v4")
    soft_v4_dir = Path("output/soft_eval_v4")
    value_v4_dir = Path("output/value_eval_v4")

    pairs = [(d.name, m) for d in benchmark_dir.iterdir() if d.is_dir() and 'broken' not in d.name
             for m in ("naive", "aware") if (d / m).is_dir()]
    typer.echo(f"\nProcessing {len(pairs)} (model, mode) pairs...")

    for model, mode in pairs:
        # Load benchmark_v4 records (= v3 minus bad + new)
        bench_records: dict[str, dict] = {}
        for f in (benchmark_dir / model / mode).glob("*.json"):
            for r in json.loads(f.read_text()):
                bench_records[r["corruption_id"]] = r

        # Existing v3 evals (we'll keep those for unchanged sample IDs)
        def _load_jsonl(path):
            if not path.exists(): return {}
            return {json.loads(l)["corruption_id"]: json.loads(l) for l in path.open()}

        old_eval = _load_jsonl(eval_v3_dir / model / f"{mode}.jsonl")
        old_soft = _load_jsonl(soft_v3_dir / model / f"{mode}.jsonl")
        old_value = _load_jsonl(value_v3_dir / model / f"{mode}.jsonl")

        # Build v4 outputs
        out_eval = []
        out_soft = []
        out_value = []
        for cid, rec in bench_records.items():
            graph = rec["graph"]
            ct = rec["corruption_type"]
            if cid in new_ids:
                # Compute fresh from cache
                if ct.startswith("A"):
                    s = _strict_eval_for_sample(rec, graph, cache)
                    if s is not None:
                        out_eval.append({
                            "corruption_id": cid, "corruption_type": ct,
                            "model": rec["model"], "mode": rec["mode"],
                            "graph_name": graph, "kind": "ambiguous",
                            "predictions": rec.get("predicted_cyphers") or [],
                            "targets": rec.get("gold_valid_cyphers") or [],
                            **s,
                        })
                    soft = _soft_eval_for_sample(rec, graph, cache)
                    out_soft.append({
                        "corruption_id": cid, "corruption_type": ct,
                        "model": rec["model"], "mode": rec["mode"],
                        "graph_name": graph, **soft,
                    })
                    val = _value_eval_for_sample(rec, graph, cache)
                    out_value.append({
                        "corruption_id": cid, "corruption_type": ct,
                        "model": rec["model"], "mode": rec["mode"],
                        "graph_name": graph, **val,
                    })
                else:  # U*
                    una = _unanswerable_eval(rec)
                    out_eval.append({
                        "corruption_id": cid, "corruption_type": ct,
                        "model": rec["model"], "mode": rec["mode"],
                        "graph_name": graph,
                        "predictions": rec.get("predicted_cyphers") or [],
                        "targets": rec.get("gold_valid_cyphers") or [],
                        **una,
                    })
            else:
                # Copy from v3
                if cid in old_eval:
                    out_eval.append(old_eval[cid])
                if cid in old_soft:
                    out_soft.append(old_soft[cid])
                if cid in old_value:
                    out_value.append(old_value[cid])

        # Write
        for out_dir, items in (
            (eval_v4_dir / model, out_eval),
            (soft_v4_dir / model, out_soft),
            (value_v4_dir / model, out_value),
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{mode}.jsonl", "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        typer.echo(f"  {model}/{mode}: eval={len(out_eval)} soft={len(out_soft)} value={len(out_value)}")

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
