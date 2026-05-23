#!/usr/bin/env python3
"""Compute four match levels (strict / forward / inverse / either) for every
A* sample across all (model, mode) benchmark runs.

Match levels — for a sample with gold cyphers G and predicted cyphers P:

  - **strict**:  some p ∈ P and some g ∈ G have identical result-set
                 fingerprints (with full Cypher aliases on keys); this is
                 what the existing evaluator computes.
  - **forward (gold ⊆ pred)**:  alias-stripped, some pred result contains
                 some gold result as a subset (every gold row appears as a
                 sub-dict of some pred row). Catches "verbose" predictions
                 that return more columns / rows than the gold.
  - **inverse (pred ⊆ gold)**:  alias-stripped, some pred result is a
                 (non-empty) subset of some gold result. Catches partial
                 predictions where the model picked one valid answer.
  - **either**:  strict OR forward OR inverse.

U* samples are skipped (they have no gold targets — they use the binary
"predicted_correctly" metric in evaluation_v3).

The pipeline is graph-major (each of the 11 graphs is loaded into Neo4j
exactly once and shared across all 18 (model, mode) pairs), and uses the
60-second tx-timeout patched in `Neo4jClient`.

Output: one JSONL per (model, mode) at
``output/soft_eval_v3/<model>/<mode>.jsonl``. Resumable via corruption_id.

Usage:
    uv run python scripts/evaluation/check_soft_match_batch.py
    uv run python scripts/evaluation/check_soft_match_batch.py --graphs nba art
    uv run python scripts/evaluation/check_soft_match_batch.py --only-model anthropic_claude-sonnet-4.6
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_graph import ALL_GRAPHS, load_graph  # noqa: E402

from cb_corruptions.verification.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=False)

GRAPH_ORDER = [
    "terrorist_attack", "flight_accident", "nba", "fictional_character",
    "company", "soccer", "geography", "movie", "politics", "art", "biology",
]


def _default_db_map() -> str:
    return json.dumps({g: "neo4j" for g in ALL_GRAPHS})


def _discover_runs(benchmark_dir: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for model_dir in sorted(benchmark_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if "broken" in model_dir.name:
            continue
        for mode in ("naive", "aware"):
            if (model_dir / mode).is_dir():
                pairs.append((model_dir.name, mode))
    return pairs


def _already_done(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["corruption_id"])
    return done


def _strip_alias_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k.split(".", 1)[1] if "." in k else k: v for k, v in row.items()}
        for row in rows
    ]


def _row_fp_keep_alias(rows: list[dict[str, Any]]) -> str:
    """Canonical JSON fingerprint preserving full alias keys (`n.name`)."""
    norm = [
        {k: (None if v is None else str(v)) for k, v in row.items()}
        for row in rows
    ]
    return json.dumps(sorted(json.dumps(r, sort_keys=True) for r in norm))


def _row_contains(super_row: dict[str, Any], sub_row: dict[str, Any]) -> bool:
    """Every (k,v) in sub_row appears in super_row (after stringifying)."""
    for k, v in sub_row.items():
        if k not in super_row:
            return False
        if str(super_row[k]) != str(v):
            return False
    return True


def _result_contains(super_rows: list[dict], sub_rows: list[dict]) -> bool:
    """For every sub_row, there exists a super_row that contains it."""
    if not sub_rows:
        return False  # vacuous match excluded by design
    for sr in sub_rows:
        if not any(_row_contains(pr, sr) for pr in super_rows):
            return False
    return True


def _run_query(client: Neo4jClient, cypher: str, graph: str) -> Optional[list[dict]]:
    try:
        return client.run_query(cypher, graph)
    except Exception:
        return None


def _evaluate_soft(
    record: dict, graph: str, client: Neo4jClient
) -> dict:
    targets = record.get("gold_valid_cyphers") or []
    preds = record.get("predicted_cyphers") or []

    gold_results = [_run_query(client, q, graph) for q in targets]
    pred_results = [_run_query(client, q, graph) for q in preds]

    n_failed_gold = sum(1 for r in gold_results if r is None)
    n_failed_pred = sum(1 for r in pred_results if r is None)

    gold_valid = [r for r in gold_results if r is not None]
    pred_valid = [r for r in pred_results if r is not None]

    # STRICT: identical fingerprints (with aliases)
    strict = False
    if pred_valid and gold_valid:
        gold_fps = {_row_fp_keep_alias(g) for g in gold_valid}
        pred_fps = {_row_fp_keep_alias(p) for p in pred_valid}
        strict = bool(gold_fps & pred_fps)

    gold_s = [_strip_alias_keys(r) for r in gold_valid]
    pred_s = [_strip_alias_keys(r) for r in pred_valid]

    # FORWARD: gold ⊆ pred (model is verbose / superset)
    forward = any(
        _result_contains(super_rows=pr, sub_rows=gr)
        for pr in pred_s if pr
        for gr in gold_s if gr
    )

    # INVERSE: pred ⊆ gold (model is partial / subset). Exclude empty pred to
    # avoid trivial vacuous matches.
    inverse = any(
        _result_contains(super_rows=gr, sub_rows=pr)
        for pr in pred_s if pr
        for gr in gold_s if gr
    )

    return {
        "corruption_id": record["corruption_id"],
        "corruption_type": record["corruption_type"],
        "model": record["model"],
        "mode": record["mode"],
        "graph_name": graph,
        "n_pred_queries": len(preds),
        "n_gold_queries": len(targets),
        "n_failed_pred_queries": n_failed_pred,
        "n_failed_gold_queries": n_failed_gold,
        "strict_match": strict,
        "forward_match": forward,
        "inverse_match": inverse,
        "either_match": strict or forward or inverse,
    }


@app.command()
def run(
    benchmark_dir: Annotated[
        Path, typer.Option("--benchmark-dir")
    ] = Path("output/benchmark_v3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("output/soft_eval_v3"),
    only_graphs: Annotated[
        Optional[list[str]], typer.Option("--graphs")
    ] = None,
    only_model: Annotated[Optional[str], typer.Option("--only-model")] = None,
    only_mode: Annotated[Optional[str], typer.Option("--only-mode")] = None,
    workers: Annotated[int, typer.Option(help="Parallel evaluations per group")] = 4,
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Compute strict / forward / inverse / either match levels for every A* sample."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    graphs = only_graphs or GRAPH_ORDER
    pairs = _discover_runs(benchmark_dir)
    if only_model:
        pairs = [(m, md) for m, md in pairs if m == only_model]
    if only_mode:
        pairs = [(m, md) for m, md in pairs if md == only_mode]
    if not pairs:
        typer.echo("No (model, mode) pairs.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Graphs: {graphs}")
    typer.echo(f"(model, mode) pairs: {len(pairs)}")

    done_by_pair: dict[tuple[str, str], set[str]] = {}
    for m, md in pairs:
        out = output_dir / m / f"{md}.jsonl"
        done_by_pair[(m, md)] = _already_done(out)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    # Gather A* work per graph (skip U*)
    work_by_graph: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for m, md in pairs:
        done = done_by_pair[(m, md)]
        for g in graphs:
            path = benchmark_dir / m / md / f"{g}.json"
            if not path.exists():
                continue
            for rec in json.loads(path.read_text()):
                if not rec["corruption_type"].startswith("A"):
                    continue
                if rec["corruption_id"] in done:
                    continue
                work_by_graph[g].append((m, md, rec))

    total = sum(len(v) for v in work_by_graph.values())
    typer.echo(f"Total A* samples to evaluate: {total}")

    for g in graphs:
        items = work_by_graph.get(g, [])
        if not items:
            typer.echo(f"\n[{g}] skip (no pending samples)")
            continue
        typer.echo(f"\n[{g}] loading + evaluating {len(items)} A* samples...")
        load_graph(driver.session, g, database="neo4j")

        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m, md, rec in items:
            by_pair[(m, md)].append(rec)

        for (m, md), recs in by_pair.items():
            out_path = output_dir / m / f"{md}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_evaluate_soft, r, g, client) for r in recs]
                    for fut in tqdm(
                        as_completed(futures), total=len(recs),
                        desc=f"{g}/{m}/{md}",
                    ):
                        f.write(json.dumps(fut.result()) + "\n")
                        f.flush()

    driver.close()
    client.close()
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
