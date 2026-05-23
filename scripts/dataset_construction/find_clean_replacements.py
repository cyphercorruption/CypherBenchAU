#!/usr/bin/env python3
"""Find clean replacement A* samples from the pool, executing their gold
cyphers on Neo4j if not already cached.

A "clean" candidate is an LLM-verified A* sample (verdict='pass' in
output/final_output/verification/) whose every gold_valid_cypher returns
≥1 row on the graph.

This extends ``output/query_cache/<graph>.jsonl`` in place (idempotent: any
cypher already cached is skipped). At the end it writes
``output/paper_tables/_replacement_candidates.json`` listing every clean
candidate not in curated_200_v3, grouped by corruption_type and graph.

Usage:
    uv run python scripts/dataset_construction/find_clean_replacements.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections import Counter, defaultdict
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


def _h(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_dir: Path) -> dict[tuple[str, str], tuple]:
    cache: dict[tuple[str, str], tuple] = {}
    for f in cache_dir.glob("*.jsonl"):
        graph = f.stem
        if graph == "cache_index":
            continue
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cache[(graph, r["hash"])] = (r.get("rows"), r.get("error"))
    return cache


def _curated_ids() -> set[str]:
    ids = set()
    for f in Path("output/curated_200_v3").glob("full_*.json"):
        for s in json.loads(f.read_text()):
            ids.add(s["corruption_id"])
    return ids


def _verified_pool_ids() -> set[str]:
    """LLM-verified A* corruption_ids from the original pool."""
    ids = set()
    for f in Path("output/final_output/verification").glob("llm_verified_*.json"):
        for r in json.loads(f.read_text()):
            if r.get("verdict") == "pass":
                cid = r["corruption_id"]
                if cid.split("-")[0] in {"A1", "A2", "A3", "A5"}:
                    ids.add(cid)
    return ids


def _candidate_samples(only_types: Optional[set[str]] = None) -> list[dict]:
    """Return pool A* samples that are LLM-verified AND not yet in curated_200_v3."""
    curated = _curated_ids()
    verified = _verified_pool_ids()
    out = []
    for f in sorted(Path("output/final_output").glob("full_*.json")):
        graph = f.stem.removeprefix("full_")
        for s in json.loads(f.read_text()):
            cid = s["corruption_id"]
            ct = s["corruption"]["corruption_type"]
            if not ct.startswith("A"):
                continue
            if cid in curated:
                continue
            if cid not in verified:
                continue
            if only_types and ct not in only_types:
                continue
            out.append({
                "corruption_id": cid,
                "corruption_type": ct,
                "graph": graph,
                "gold_cyphers": s.get("valid_cyphers") or [],
                "sample": s,
            })
    return out


@app.command()
def run(
    only_types: Annotated[
        Optional[list[str]], typer.Option("--types", help="Restrict to corruption types (default A1 A3 A5)")
    ] = None,
    cache_dir: Annotated[Path, typer.Option()] = Path("output/query_cache"),
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    timeout_s: Annotated[float, typer.Option()] = 60.0,
) -> None:
    """Execute pool gold cyphers (if not cached) and report clean replacement candidates."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    types = set(only_types or ["A1", "A3", "A5"])
    cands = _candidate_samples(types)
    typer.echo(f"Candidates (LLM-verified, not in curated): {len(cands)}")
    typer.echo(f"  by type: {Counter(c['corruption_type'] for c in cands)}")
    typer.echo(f"  by graph: {Counter(c['graph'] for c in cands)}")

    cache = _load_cache(cache_dir)
    # Find cyphers we still need to run
    todo: dict[str, dict[str, str]] = defaultdict(dict)  # graph -> hash -> cypher
    for c in cands:
        for q in c["gold_cyphers"]:
            if (c["graph"], _h(q)) not in cache:
                todo[c["graph"]][_h(q)] = q
    typer.echo(f"\nCyphers to execute (not in cache): {sum(len(v) for v in todo.values())}")
    for g in GRAPH_ORDER:
        if todo.get(g):
            typer.echo(f"  {g}: {len(todo[g])}")

    # Execute and append to cache
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
            typer.echo(f"\n[{g}] loading + executing {len(items)} new cyphers...")
            load_graph(driver.session, g, database="neo4j")
            out_path = cache_dir / f"{g}.jsonl"
            with open(out_path, "a") as f:
                for hsh, q in items.items():
                    try:
                        rows = client.run_query(q, g, timeout_s=timeout_s)
                        line = {"hash": hsh, "cypher": q, "rows": rows, "error": None}
                        cache[(g, hsh)] = (rows, None)
                    except Exception as e:
                        line = {"hash": hsh, "cypher": q, "rows": None,
                                "error": f"{type(e).__name__}: {str(e)[:200]}"}
                        cache[(g, hsh)] = (None, line["error"])
                    f.write(json.dumps(line, default=str) + "\n")
                f.flush()

        driver.close()
        client.close()

    # Classify candidates
    clean = []
    for c in cands:
        golds = c["gold_cyphers"]
        if not golds:
            continue
        results = [cache.get((c["graph"], _h(q))) for q in golds]
        if any(r is None for r in results):
            continue
        if any(r[1] is not None for r in results):  # any error
            continue
        if any(len(r[0]) == 0 for r in results):  # any empty
            continue
        clean.append(c)

    typer.echo(f"\nClean replacement candidates: {len(clean)}")
    by_ct_graph = defaultdict(lambda: defaultdict(list))
    for c in clean:
        by_ct_graph[c["corruption_type"]][c["graph"]].append(c["corruption_id"])
    for ct in sorted(by_ct_graph):
        n = sum(len(v) for v in by_ct_graph[ct].values())
        typer.echo(f"  {ct}: {n}")
        for g, ids in sorted(by_ct_graph[ct].items()):
            typer.echo(f"    {g}: {len(ids)}")

    # Save full details (without 'sample' to keep it small)
    out_path = Path("output/paper_tables/_replacement_candidates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = [{k: v for k, v in c.items() if k != "sample"} for c in clean]
    out_path.write_text(json.dumps(out, indent=2))
    typer.echo(f"\nSaved: {out_path}")


if __name__ == "__main__":
    app()
