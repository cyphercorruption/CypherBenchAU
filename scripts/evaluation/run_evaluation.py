#!/usr/bin/env python3
"""End-to-end evaluation of a single (model, mode) benchmark run.

For each record in the input dir:
  - corruption_type starts with 'A' → execute predictions vs targets on
    Neo4j via cb_corruptions.verification.evaluate() → precision/recall
  - corruption_type starts with 'U' → binary score: correct iff predicted
    cypher list is empty (the question is unanswerable by design)

Output is one JSONL line per sample under output/evaluation_v3/<model>/<mode>.jsonl
(auto-derived from input path). Resumable: samples whose corruption_id is
already in the output file are skipped.

Usage:
    uv run python scripts/evaluation/run_evaluation.py -i output/benchmark_v3/<model>/<mode>
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from load_graph import ALL_GRAPHS, load_graph  # noqa: E402

from cb_corruptions.verification import evaluate
from cb_corruptions.verification.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


def _default_db_map() -> str:
    return json.dumps({g: "neo4j" for g in ALL_GRAPHS})


def _load_records(input_dir: Path) -> list[dict]:
    records: list[dict] = []
    for f in sorted(input_dir.glob("*.json")):
        records.extend(json.loads(f.read_text()))
    return records


def _already_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["corruption_id"])
    return done


def _derive_output_path(input_dir: Path) -> Path:
    """output/benchmark_v3/<model>/<mode>/ → output/evaluation_v3/<model>/<mode>.jsonl"""
    parts = input_dir.resolve().parts
    try:
        i = parts.index("benchmark_v3")
    except ValueError as exc:
        raise typer.BadParameter(
            "Cannot auto-derive output path: 'benchmark_v3' not in input path; "
            "pass --output explicitly."
        ) from exc
    model_slug, mode = parts[i + 1], parts[i + 2]
    return Path("output/evaluation_v3") / model_slug / f"{mode}.jsonl"


def _score_unanswerable(record: dict) -> dict:
    """Binary metric: predicted_correctly iff the model returned no Cypher."""
    n_pred = len(record.get("predicted_cyphers") or [])
    return {
        "kind": "unanswerable",
        "predicted_correctly": n_pred == 0,
        "n_predictions": n_pred,
    }


def _eval_ambiguous(record: dict, graph_name: str, client: Neo4jClient) -> dict:
    """Run evaluate() and shape the output row. Wraps evaluator failures."""
    base = {
        "corruption_id": record["corruption_id"],
        "corruption_type": record["corruption_type"],
        "model": record["model"],
        "mode": record["mode"],
        "graph_name": graph_name,
        "kind": "ambiguous",
        "predictions": record.get("predicted_cyphers") or [],
        "targets": record.get("gold_valid_cyphers") or [],
    }
    try:
        result = evaluate(
            base["predictions"], base["targets"], graph_name, client
        )
        return {**base, **result.model_dump()}
    except Exception as e:
        logger.error("evaluate() failed for %s: %s", record["corruption_id"], e)
        return {**base, "kind": "ambiguous_error", "evaluator_error": str(e)}


def _print_summary(output_file: Path) -> None:
    by_ct: dict[str, list[dict]] = defaultdict(list)
    with open(output_file) as f:
        for line in f:
            r = json.loads(line)
            by_ct[r["corruption_type"]].append(r)

    typer.echo(f"\n{'Type':<6} {'N':>5}  {'metric':<28}")
    typer.echo("-" * 44)
    a_prec_sum = a_rec_sum = a_n = 0
    u_correct = u_n = 0
    for ct in sorted(by_ct.keys()):
        rows = by_ct[ct]
        n = len(rows)
        if ct.startswith("A"):
            prec = sum(r.get("precision", 0.0) for r in rows) / n
            rec = sum(r.get("recall", 0.0) for r in rows) / n
            typer.echo(f"{ct:<6} {n:>5}  prec={prec:.3f}  rec={rec:.3f}")
            a_prec_sum += prec * n
            a_rec_sum += rec * n
            a_n += n
        else:
            correct = sum(1 for r in rows if r.get("predicted_correctly"))
            typer.echo(f"{ct:<6} {n:>5}  acc={correct / n:.3f}")
            u_correct += correct
            u_n += n
    typer.echo("-" * 44)
    if a_n:
        typer.echo(
            f"{'A* tot':<6} {a_n:>5}  prec={a_prec_sum / a_n:.3f}  rec={a_rec_sum / a_n:.3f}"
        )
    if u_n:
        typer.echo(f"{'U* tot':<6} {u_n:>5}  acc={u_correct / u_n:.3f}")


@app.command()
def run(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i", "--input-dir",
            help="A single (model, mode) benchmark dir, e.g. output/benchmark_v3/<model>/<mode>",
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o", "--output",
            help="Output JSONL. Auto-derived from input-dir if omitted.",
        ),
    ] = None,
    uri: Annotated[str, typer.Option(help="Neo4j bolt URI")] = os.environ.get(
        "NEO4J_URI", "bolt://localhost:7687"
    ),
    username: Annotated[str, typer.Option(help="Neo4j username")] = os.environ.get(
        "NEO4J_USERNAME", "neo4j"
    ),
    password: Annotated[str, typer.Option(help="Neo4j password")] = os.environ.get(
        "NEO4J_PASSWORD", "password"
    ),
    workers: Annotated[
        int, typer.Option(help="Parallel evaluate() calls per graph group")
    ] = 4,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Evaluate one (model, mode) run end-to-end."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    # Auto-populate the Community-edition DB map: every graph → "neo4j".
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    if not input_dir.is_dir():
        typer.echo(f"Input dir not found: {input_dir}", err=True)
        raise typer.Exit(1)

    if output_file is None:
        output_file = _derive_output_path(input_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Input:  {input_dir}")
    typer.echo(f"Output: {output_file}")

    records = _load_records(input_dir)
    done = _already_done(output_file)
    if done:
        typer.echo(f"Resuming: {len(done)} samples already in output, skipping")
    remaining = [r for r in records if r["corruption_id"] not in done]
    if not remaining:
        typer.echo("All samples already evaluated.")
        _print_summary(output_file)
        return
    typer.echo(f"To evaluate: {len(remaining)} / {len(records)}")

    amb_by_graph: dict[str, list[dict]] = defaultdict(list)
    unanswerable: list[dict] = []
    for r in remaining:
        if r["corruption_type"].startswith("A"):
            amb_by_graph[r["graph"]].append(r)
        else:
            unanswerable.append(r)
    typer.echo(
        f"  ambiguous A* across {len(amb_by_graph)} graphs: "
        f"{sum(len(v) for v in amb_by_graph.values())}"
    )
    typer.echo(f"  unanswerable U*: {len(unanswerable)}")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    with open(output_file, "a") as out_f:
        for graph_name in sorted(amb_by_graph):
            group = amb_by_graph[graph_name]
            typer.echo(
                f"\n[{graph_name}] loading + evaluating {len(group)} A* samples..."
            )
            load_graph(driver.session, graph_name, database="neo4j")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_eval_ambiguous, r, graph_name, client): r
                    for r in group
                }
                for fut in tqdm(
                    as_completed(futures), total=len(group), desc=graph_name
                ):
                    line = fut.result()
                    out_f.write(json.dumps(line) + "\n")
                    out_f.flush()

        if unanswerable:
            typer.echo(f"\n[U*] scoring {len(unanswerable)} samples (binary, no Neo4j)")
            for rec in unanswerable:
                line = {
                    "corruption_id": rec["corruption_id"],
                    "corruption_type": rec["corruption_type"],
                    "model": rec["model"],
                    "mode": rec["mode"],
                    "graph_name": rec["graph"],
                    "predictions": rec.get("predicted_cyphers") or [],
                    "targets": rec.get("gold_valid_cyphers") or [],
                    **_score_unanswerable(rec),
                }
                out_f.write(json.dumps(line) + "\n")

    driver.close()
    client.close()

    _print_summary(output_file)


if __name__ == "__main__":
    app()
