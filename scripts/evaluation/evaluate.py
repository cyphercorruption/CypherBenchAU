#!/usr/bin/env python3
"""Batch precision & recall evaluator for Cypher query predictions.

Usage:
    uv run python scripts/evaluation/evaluate.py -i predictions.json -o results.jsonl
    uv run python scripts/evaluation/evaluate.py -i predictions.json -o results.jsonl --workers 8

Neo4j Community Edition note:
    Community Edition only supports a single database ("neo4j"). Set the
    NEO4J_DATABASE_MAP env var to route all graph names to it, e.g.:
        export NEO4J_DATABASE_MAP='{"nba":"neo4j","geography":"neo4j"}'
    See also: scripts/verify_all.sh for a working example.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from neo4j import GraphDatabase
from tqdm import tqdm

# Allow importing load_graph from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_graph import load_graph  # noqa: E402

from cb_corruptions.verification import evaluate
from cb_corruptions.verification.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

app = typer.Typer(name="evaluate", no_args_is_help=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in an existing JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@app.command()
def run(
    input_file: Annotated[Path, typer.Option("-i", "--input", help="Input JSON file with predictions and targets")],
    output_file: Annotated[Path, typer.Option("-o", "--output", help="Output JSONL file")],
    uri: Annotated[str, typer.Option(help="Neo4j bolt URI")] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option(help="Neo4j username")] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option(help="Neo4j password")] = os.environ.get("NEO4J_PASSWORD", "password"),
    database: Annotated[str, typer.Option(help="Neo4j target database")] = os.environ.get("NEO4J_DATABASE", "neo4j"),
    workers: Annotated[int, typer.Option(help="Max parallel evaluate() calls per graph group")] = 4,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
) -> None:
    """Evaluate Cypher query predictions against targets using precision & recall."""
    _setup_logging(verbose)

    if not input_file.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    with open(input_file) as f:
        samples: list[dict] = json.load(f)

    typer.echo(f"Loaded {len(samples)} samples from {input_file}")

    # Recovery: if output exists, skip already-processed samples
    already_done = 0
    if output_file.exists():
        already_done = _count_lines(output_file)
        typer.echo(f"Resuming: skipping {already_done} already-processed samples (output file exists)")
    samples = samples[already_done:]

    if not samples:
        typer.echo("All samples already processed. Nothing to do.")
        raise typer.Exit(0)

    typer.echo(f"{len(samples)} samples remaining to evaluate")

    # Group remaining samples by graph_name (preserves within-group insertion order)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["graph_name"]].append(s)

    typer.echo(f"Graphs to process: {', '.join(groups.keys())}")

    # Connect raw driver for load_graph() and Neo4jClient for evaluate()
    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if already_done > 0 else "w"

    typer.echo(f"Output: {output_file} (mode={file_mode})")

    per_graph_stats: dict[str, dict] = {}

    with open(output_file, file_mode) as out_f:
        for graph_name, group in groups.items():
            typer.echo(f"\n[{graph_name}] Loading graph into Neo4j ({len(group)} samples to evaluate)...")
            load_graph(driver.session, graph_name, database=database)

            typer.echo(f"[{graph_name}] Evaluating with {workers} workers...")
            precs: list[float] = []
            recs: list[float] = []

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_sample = {
                    pool.submit(
                        evaluate,
                        s["predictions"],
                        s["targets"],
                        graph_name,
                        client,
                    ): s
                    for s in group
                }
                for future in tqdm(as_completed(future_to_sample), total=len(group), desc=graph_name):
                    sample = future_to_sample[future]
                    result = future.result()  # raises immediately if evaluate() raised
                    merged = {**sample, **result.model_dump()}
                    out_f.write(json.dumps(merged) + "\n")
                    out_f.flush()
                    precs.append(result.precision)
                    recs.append(result.recall)

            mean_prec = sum(precs) / len(precs) if precs else 0.0
            mean_rec = sum(recs) / len(recs) if recs else 0.0
            per_graph_stats[graph_name] = {"n": len(group), "prec": mean_prec, "rec": mean_rec}
            typer.echo(f"[{graph_name}] done — prec={mean_prec:.3f} rec={mean_rec:.3f}")

    driver.close()
    client.close()

    # Summary table
    total_n = sum(v["n"] for v in per_graph_stats.values())
    total_prec = (
        sum(v["prec"] * v["n"] for v in per_graph_stats.values()) / total_n if total_n else 0.0
    )
    total_rec = (
        sum(v["rec"] * v["n"] for v in per_graph_stats.values()) / total_n if total_n else 0.0
    )

    typer.echo(f"\n{'Graph':<22} {'N':>5} {'Prec':>8} {'Rec':>8}")
    typer.echo("-" * 48)
    for g, stats in per_graph_stats.items():
        typer.echo(f"{g:<22} {stats['n']:>5} {stats['prec']:>8.3f} {stats['rec']:>8.3f}")
    typer.echo("-" * 48)
    typer.echo(f"{'TOTAL':<22} {total_n:>5} {total_prec:>8.3f} {total_rec:>8.3f}")

    typer.echo(
        "\nWARNING: output line order is non-deterministic due to parallel execution. "
        "Use the sample_id field (or equivalent passthrough field) to match results "
        "back to input samples."
    )


if __name__ == "__main__":
    app()
