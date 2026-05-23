#!/usr/bin/env python3
"""Graph-major batch evaluation across many (model, mode) benchmark runs.

For each graph (loaded into Neo4j exactly once), this script evaluates
the A* samples for every discovered (model, mode) pair against that graph,
then moves on to the next graph. U* samples are scored at the end without
touching Neo4j (binary metric: correct iff predicted_cyphers is empty).

This avoids the (model, mode) x graph quadratic loading cost that comes from
running `scripts/run_evaluation.py` once per (model, mode): with N=18 pairs
and G=11 graphs, the naive layout incurs N*G=198 graph loads, while this
script only does G=11.

Output is the same per-(model, mode) JSONL under
``output/evaluation_v3/<model>/<mode>.jsonl``, and is resumable: any sample
whose corruption_id already appears in the output is skipped.

Usage:
    uv run python scripts/evaluation/run_evaluation_batch.py
    uv run python scripts/evaluation/run_evaluation_batch.py --graphs nba art
    uv run python scripts/evaluation/run_evaluation_batch.py --only-model anthropic_claude-sonnet-4.6
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import typer
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_graph import ALL_GRAPHS, load_graph  # noqa: E402

from cb_corruptions.verification import evaluate
from cb_corruptions.verification.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=False)


# Smallest-first so we can spot issues early in long batch runs.
GRAPH_ORDER = [
    "terrorist_attack", "flight_accident", "nba", "fictional_character",
    "company", "soccer", "geography", "movie", "politics", "art", "biology",
]


def _default_db_map() -> str:
    return json.dumps({g: "neo4j" for g in ALL_GRAPHS})


def _discover_runs(benchmark_dir: Path) -> list[tuple[str, str]]:
    """Return all (model_slug, mode) pairs found under benchmark_dir."""
    pairs: list[tuple[str, str]] = []
    for model_dir in sorted(benchmark_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if "broken" in model_dir.name:  # skip backup dirs
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


def _load_graph_records(
    benchmark_dir: Path, model_slug: str, mode: str, graph: str
) -> list[dict]:
    path = benchmark_dir / model_slug / mode / f"{graph}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _eval_ambiguous_row(record: dict, graph: str, client: Neo4jClient) -> dict:
    base = {
        "corruption_id": record["corruption_id"],
        "corruption_type": record["corruption_type"],
        "model": record["model"],
        "mode": record["mode"],
        "graph_name": graph,
        "kind": "ambiguous",
        "predictions": record.get("predicted_cyphers") or [],
        "targets": record.get("gold_valid_cyphers") or [],
    }
    try:
        result = evaluate(base["predictions"], base["targets"], graph, client)
        return {**base, **result.model_dump()}
    except Exception as e:
        logger.error("evaluate() failed for %s: %s", record["corruption_id"], e)
        return {**base, "kind": "ambiguous_error", "evaluator_error": str(e)}


def _score_unanswerable_row(record: dict) -> dict:
    preds = record.get("predicted_cyphers") or []
    return {
        "corruption_id": record["corruption_id"],
        "corruption_type": record["corruption_type"],
        "model": record["model"],
        "mode": record["mode"],
        "graph_name": record["graph"],
        "kind": "unanswerable",
        "predictions": preds,
        "targets": record.get("gold_valid_cyphers") or [],
        "predicted_correctly": len(preds) == 0,
        "n_predictions": len(preds),
    }


@app.command()
def run(
    benchmark_dir: Annotated[
        Path, typer.Option("--benchmark-dir", help="Root with <model>/<mode>/<graph>.json")
    ] = Path("output/benchmark_v3"),
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where to write <model>/<mode>.jsonl")
    ] = Path("output/evaluation_v3"),
    only_graphs: Annotated[
        Optional[list[str]], typer.Option("--graphs", help="Subset of graphs (default: all 11)")
    ] = None,
    only_model: Annotated[
        Optional[str], typer.Option("--only-model", help="Restrict to a single model slug")
    ] = None,
    only_mode: Annotated[
        Optional[str], typer.Option("--only-mode", help="naive or aware")
    ] = None,
    workers: Annotated[
        int, typer.Option(help="Parallel evaluate() calls per (graph, model, mode)")
    ] = 4,
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Graph-major batch evaluation across all discovered (model, mode) runs."""
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
        typer.echo("No (model, mode) pairs to evaluate.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Graphs: {graphs}")
    typer.echo(f"(model, mode) pairs: {len(pairs)}")
    for m, md in pairs:
        typer.echo(f"  {m}/{md}")

    # Preload "already done" per (model, mode) so we can skip cleanly.
    done_by_pair: dict[tuple[str, str], set[str]] = {}
    for m, md in pairs:
        out = output_dir / m / f"{md}.jsonl"
        done_by_pair[(m, md)] = _already_done(out)
        if done_by_pair[(m, md)]:
            typer.echo(f"  resume: {m}/{md} → {len(done_by_pair[(m, md)])} already done")

    # Connect Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    # Helper: collect A* work per graph across all pairs before loading.
    a_work: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    u_work_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m, md in pairs:
        done = done_by_pair[(m, md)]
        for g in graphs:
            for rec in _load_graph_records(benchmark_dir, m, md, g):
                if rec["corruption_id"] in done:
                    continue
                if rec["corruption_type"].startswith("A"):
                    a_work[g].append((m, md, rec))
                else:
                    u_work_by_pair[(m, md)].append(rec)

    a_total = sum(len(v) for v in a_work.values())
    u_total = sum(len(v) for v in u_work_by_pair.values())
    typer.echo(f"\nWork: {a_total} A* (across {len(a_work)} graphs), {u_total} U*")

    # Process A* graph by graph
    for g in graphs:
        items = a_work.get(g, [])
        if not items:
            typer.echo(f"\n[{g}] skip (no pending A* samples)")
            continue
        typer.echo(f"\n[{g}] loading + evaluating {len(items)} A* samples...")
        load_graph(driver.session, g, database="neo4j")

        # Open all output files for append, grouped by (m, md)
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m, md, rec in items:
            by_pair[(m, md)].append(rec)

        for (m, md), recs in by_pair.items():
            out_path = output_dir / m / f"{md}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_eval_ambiguous_row, r, g, client) for r in recs]
                    for fut in tqdm(
                        as_completed(futures), total=len(recs), desc=f"{g}/{m}/{md}"
                    ):
                        f.write(json.dumps(fut.result()) + "\n")
                        f.flush()

    # Process U* (no Neo4j)
    if u_total:
        typer.echo(f"\n[U*] scoring {u_total} unanswerable samples (binary)")
        for (m, md), recs in u_work_by_pair.items():
            out_path = output_dir / m / f"{md}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                for rec in recs:
                    f.write(json.dumps(_score_unanswerable_row(rec)) + "\n")

    driver.close()
    client.close()
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
