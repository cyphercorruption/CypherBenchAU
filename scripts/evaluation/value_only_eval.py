#!/usr/bin/env python3
"""Value-only evaluation: match by raw result values, ignoring column names.

For each A* sample, executes gold and predicted Cypher on Neo4j and compares
the result sets at the value level only — column names and aliases are
stripped entirely.

A "row" is represented as a frozen multiset of stringified values; the whole
result set is represented as a frozenset of those rows. Two queries match
iff their representations are equal (strict_value) or one is a subset of the
other (forward / inverse / either).

Output: ``output/value_eval_v3/<model>/<mode>.jsonl`` (resumable via
corruption_id).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

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


def _row_to_value_bag(row: dict) -> tuple:
    """A row's signature, ignoring column names. Sorted tuple of stringified
    values (a multiset — preserves duplicates within the row)."""
    return tuple(sorted(("null" if v is None else str(v)) for v in row.values()))


def _result_signature(rows: list[dict]) -> frozenset:
    """Whole result-set signature: frozenset of row-value-bags.

    Dedups identical rows (DISTINCT-by-content), discards row order.
    """
    return frozenset(_row_to_value_bag(r) for r in rows)


def _result_signature_multiset(rows: list[dict]) -> Counter:
    """Multiset variant: preserves duplicate rows."""
    return Counter(_row_to_value_bag(r) for r in rows)


def _is_subset_multiset(sub: Counter, sup: Counter) -> bool:
    for k, v in sub.items():
        if sup.get(k, 0) < v:
            return False
    return True


def _run_query(client: Neo4jClient, cypher: str, graph: str) -> Optional[list[dict]]:
    try:
        return client.run_query(cypher, graph)
    except Exception:
        return None


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


def _evaluate_value(
    record: dict, graph: str, gold_sigs: list,        # list of (set, multiset) tuples
    gold_sigs_nonempty: list, client: Neo4jClient,
) -> dict:
    preds = record.get("predicted_cyphers") or []
    pred_results = [_run_query(client, q, graph) for q in preds]
    pred_valid = [r for r in pred_results if r is not None]
    n_failed_pred = sum(1 for r in pred_results if r is None)

    pred_sigs = [(_result_signature(r), _result_signature_multiset(r)) for r in pred_valid]
    pred_sigs_nonempty = [s for s in pred_sigs if s[0]]  # drop empty result sets

    # All comparisons exclude empty-result fingerprints on BOTH sides
    strict = False
    forward = False
    inverse = False
    for ps_set, ps_mset in pred_sigs_nonempty:
        for gs_set, gs_mset in gold_sigs_nonempty:
            if ps_set == gs_set:
                strict = True
            if gs_set.issubset(ps_set):  # gold ⊆ pred (model is verbose)
                forward = True
            if ps_set.issubset(gs_set):  # pred ⊆ gold (model is partial)
                inverse = True

    return {
        "corruption_id": record["corruption_id"],
        "corruption_type": record["corruption_type"],
        "model": record["model"],
        "mode": record["mode"],
        "graph_name": graph,
        "n_pred": len(preds),
        "n_pred_valid": len(pred_valid),
        "n_pred_nonempty": len(pred_sigs_nonempty),
        "n_gold": len(gold_sigs),
        "n_gold_nonempty": len(gold_sigs_nonempty),
        "n_failed_pred_queries": n_failed_pred,
        "strict_value": strict,
        "forward_value": forward,
        "inverse_value": inverse,
        "either_value": strict or forward or inverse,
    }


@app.command()
def run(
    benchmark_dir: Annotated[Path, typer.Option()] = Path("output/benchmark_v3"),
    output_dir: Annotated[Path, typer.Option()] = Path("output/value_eval_v3"),
    only_graphs: Annotated[Optional[list[str]], typer.Option("--graphs")] = None,
    only_model: Annotated[Optional[str], typer.Option("--only-model")] = None,
    only_mode: Annotated[Optional[str], typer.Option("--only-mode")] = None,
    workers: Annotated[int, typer.Option()] = 4,
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
) -> None:
    """Value-only match across all (model, mode) A* samples."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    graphs = only_graphs or GRAPH_ORDER
    pairs = _discover_runs(benchmark_dir)
    if only_model:
        pairs = [(m, md) for m, md in pairs if m == only_model]
    if only_mode:
        pairs = [(m, md) for m, md in pairs if md == only_mode]
    typer.echo(f"Graphs: {graphs}")
    typer.echo(f"(model, mode) pairs: {len(pairs)}")

    done_by_pair = {(m, md): _already_done(output_dir / m / f"{md}.jsonl") for m, md in pairs}

    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    # Gather work
    work_by_graph: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    gold_by_cid: dict[str, list[str]] = {}
    graph_by_cid: dict[str, str] = {}
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
                gold_by_cid[rec["corruption_id"]] = rec["gold_valid_cyphers"]
                graph_by_cid[rec["corruption_id"]] = g
                work_by_graph[g].append((m, md, rec))

    typer.echo(f"Total A* (model,mode) evaluations to do: {sum(len(v) for v in work_by_graph.values())}")

    for g in graphs:
        items = work_by_graph.get(g, [])
        if not items:
            typer.echo(f"\n[{g}] skip (no pending)")
            continue
        typer.echo(f"\n[{g}] loading + evaluating {len(items)} entries...")
        load_graph(driver.session, g, database="neo4j")

        # Precompute gold sigs once per cid for this graph
        unique_cids = {it[2]["corruption_id"] for it in items}
        gold_cache: dict[str, tuple] = {}
        for cid in unique_cids:
            golds = gold_by_cid[cid]
            sigs = []
            for q in golds:
                rows = _run_query(client, q, g)
                if rows is not None:
                    sigs.append((_result_signature(rows), _result_signature_multiset(rows)))
            sigs_nonempty = [s for s in sigs if s[0]]
            gold_cache[cid] = (sigs, sigs_nonempty)

        # Group by (model, mode) for batched writing
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m, md, rec in items:
            by_pair[(m, md)].append(rec)

        for (m, md), recs in by_pair.items():
            out_path = output_dir / m / f"{md}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            _evaluate_value, rec, g,
                            gold_cache[rec["corruption_id"]][0],
                            gold_cache[rec["corruption_id"]][1],
                            client,
                        ) for rec in recs
                    ]
                    for fut in tqdm(as_completed(futures), total=len(recs), desc=f"{g}/{m}/{md}"):
                        f.write(json.dumps(fut.result()) + "\n")
                        f.flush()

    driver.close()
    client.close()
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
