#!/usr/bin/env python3
"""Execute every unique (graph, cypher) once and persist the raw rows.

This is run ONCE. From then on, any metric variation (strict/lenient/value-only/
key-based/whatever else) is a pure data-processing step over the cache —
no more re-running anything on Neo4j.

Cache layout::

    output/query_cache/
    └── <graph>.jsonl              # one line per cypher
                                   # {"hash": ..., "cypher": ..., "rows": [...], "error": null}

For convenience also produces an index ``cache_index.json`` that maps every
seen (graph, cypher) to its hash.

Resumable: if a cypher already has a line in the per-graph file, it is
skipped.

Usage:
    uv run python scripts/evaluation/cache_query_results.py
    uv run python scripts/evaluation/cache_query_results.py --graphs nba
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Optional

import typer
from neo4j import GraphDatabase

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


def _hash(cypher: str) -> str:
    return hashlib.sha256(cypher.encode("utf-8")).hexdigest()[:16]


def _collect_unique_queries(benchmark_dir: Path) -> dict[str, dict[str, str]]:
    """Return {graph: {hash: cypher}} across all benchmark records.

    Includes every gold_valid_cyphers and every predicted_cyphers entry. The
    `original_gold_cypher` is also indexed so that downstream consumers can
    look it up if needed.
    """
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for f in sorted(benchmark_dir.rglob("*.json")):
        parts = f.parts
        if "broken" in str(f):
            continue
        for rec in json.loads(f.read_text()):
            g = rec["graph"]
            for q in (rec.get("gold_valid_cyphers") or []):
                if q:
                    out[g][_hash(q)] = q
            for q in (rec.get("predicted_cyphers") or []):
                if q:
                    out[g][_hash(q)] = q
            ogc = rec.get("original_gold_cypher")
            if ogc:
                out[g][_hash(ogc)] = ogc
    return out


def _load_existing_cache(cache_dir: Path) -> dict[str, set[str]]:
    """Return {graph: set(hash)} of cypher hashes already cached."""
    done: dict[str, set[str]] = defaultdict(set)
    for f in cache_dir.glob("*.jsonl"):
        graph = f.stem
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                done[graph].add(json.loads(line)["hash"])
            except Exception:
                continue
    return done


@app.command()
def run(
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path("output/benchmark_v3"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("output/query_cache"),
    only_graphs: Annotated[Optional[list[str]], typer.Option("--graphs")] = None,
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    timeout_s: Annotated[float, typer.Option(help="Per-query tx timeout")] = 60.0,
) -> None:
    """Execute every unique (graph, cypher) once and persist results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())
    cache_dir.mkdir(parents=True, exist_ok=True)

    queries = _collect_unique_queries(benchmark_dir)
    typer.echo(f"Unique cyphers per graph:")
    for g in sorted(queries):
        typer.echo(f"  {g}: {len(queries[g])}")

    graphs = only_graphs or GRAPH_ORDER
    already = _load_existing_cache(cache_dir)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = Neo4jClient(uri=uri, username=username, password=password)
    try:
        driver.verify_connectivity()
    except Exception as e:
        typer.echo(f"Cannot connect to Neo4j: {e}", err=True)
        raise typer.Exit(1)

    grand_total_ok = grand_total_err = 0
    for g in graphs:
        gqs = queries.get(g, {})
        pending = {h: q for h, q in gqs.items() if h not in already.get(g, set())}
        if not pending:
            typer.echo(f"\n[{g}] all cached ({len(gqs)} cyphers); skip")
            continue
        typer.echo(f"\n[{g}] {len(pending)}/{len(gqs)} to execute (rest cached). Loading graph...")
        load_graph(driver.session, g, database="neo4j")

        out_path = cache_dir / f"{g}.jsonl"
        with open(out_path, "a") as f:
            ok = err = 0
            for i, (h, q) in enumerate(pending.items(), 1):
                try:
                    rows = client.run_query(q, g, timeout_s=timeout_s)
                    line = {"hash": h, "cypher": q, "rows": rows, "error": None}
                    ok += 1
                except Exception as e:
                    line = {"hash": h, "cypher": q, "rows": None,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    err += 1
                f.write(json.dumps(line, default=str) + "\n")
                if i % 50 == 0 or i == len(pending):
                    typer.echo(f"  {g}: {i}/{len(pending)}  (ok={ok}, err={err})")
                f.flush()
        grand_total_ok += ok
        grand_total_err += err

    # Refresh global index (graph + hash → cypher)
    idx = {}
    for f in cache_dir.glob("*.jsonl"):
        graph = f.stem
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[f"{graph}|{r['hash']}"] = {"graph": graph, "cypher": r["cypher"]}
    (cache_dir / "cache_index.json").write_text(json.dumps(idx, indent=2))

    typer.echo(f"\nGrand total executed this run: ok={grand_total_ok} err={grand_total_err}")
    typer.echo(f"Cache index: {cache_dir / 'cache_index.json'}")
    driver.close()
    client.close()


if __name__ == "__main__":
    app()
