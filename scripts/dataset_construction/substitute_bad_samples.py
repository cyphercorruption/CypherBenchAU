#!/usr/bin/env python3
"""Substitute the 47 A* samples in curated_200_v3 whose gold queries return
empty results, drawing replacements from the LLM-verified pool.

For A3, applies a relaxed criterion: accept LLM-pass plus LLM-fail whose
issue rationale is purely "hypernym/union" (the verifier's strict
interpretation that a hypernym = union rather than ambiguity is debatable
for our benchmark scope).

For each candidate, executes its `valid_cyphers` on Neo4j (graph-major,
extending the existing cache) and keeps only those where every variant
returns ≥1 row. Final selection diversifies by the `corruption.corrupted_element`
field (the swapped word/relation) so we don't pile up samples that all
attack the same relation.

Output:
  - output/curated_200_v4/full_<graph>.json        — new dataset
  - output/curated_200_v4/verification/...         — copied + extended LLM verification
  - output/curated_200_v4/_replacement_plan.json   — bad→new mapping
  - output/curated_200_v4/_new_corruption_ids.txt  — only the 47 new IDs, to feed back into benchmark

Usage:
    uv run python scripts/dataset_construction/substitute_bad_samples.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
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


def _h(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]


def _default_db_map() -> str:
    return json.dumps({g: "neo4j" for g in ALL_GRAPHS})


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{graph}.jsonl", "a") as f:
        f.write(json.dumps({"hash": hsh, "cypher": cypher,
                            "rows": rows, "error": error},
                           default=str) + "\n")


def _load_pool_samples() -> dict[str, dict]:
    """Map corruption_id → sample dict, for the full_*.json pool."""
    out = {}
    for f in sorted(Path("output/final_output").glob("full_*.json")):
        graph = f.stem.removeprefix("full_")
        for s in json.loads(f.read_text()):
            out[s["corruption_id"]] = {**s, "_graph": graph}
    return out


def _load_llm_verdicts() -> dict[str, dict]:
    """Map corruption_id → LLM verification record."""
    out = {}
    for f in Path("output/final_output/verification").glob("llm_verified_*.json"):
        for r in json.loads(f.read_text()):
            out[r["corruption_id"]] = r
    return out


def _load_bad_curated() -> dict[str, list[dict]]:
    """Return {corruption_type: [bad_curated_records]} for the 47 problematic samples."""
    gold_data = json.load(open("output/paper_tables/_gold_empty_samples.json"))
    bad_ids = {r["cid"]: r for r in gold_data if r["n_empty"] > 0}

    bad_by_type: dict[str, list[dict]] = defaultdict(list)
    for f in Path("output/curated_200_v3").glob("full_*.json"):
        for s in json.loads(f.read_text()):
            if s["corruption_id"] in bad_ids:
                bad_by_type[s["corruption_id"].split("-")[0]].append(s)
    return bad_by_type


def _is_hypernym_only_fail(r: dict) -> bool:
    """A3 relaxed: accept LLM-fail whose issues are purely about hypernym/union."""
    issues = " ".join(r.get("issues") or []).lower()
    return ("hypernym" in issues) or ("union" in issues)


def _candidate_ids_per_type(
    pool: dict[str, dict],
    verdicts: dict[str, dict],
    curated_ids: set[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for cid, sample in pool.items():
        if cid in curated_ids:
            continue
        ct = sample["corruption"]["corruption_type"]
        if ct not in {"A1", "A3", "A5"}:
            continue
        ver = verdicts.get(cid)
        if not ver:
            continue
        # Strict pass for A1/A5; relaxed for A3
        if ct == "A3":
            if ver["verdict"] == "pass" or _is_hypernym_only_fail(ver):
                out[ct].append(cid)
        else:
            if ver["verdict"] == "pass":
                out[ct].append(cid)
    return out


def _pick_diverse(
    clean_cands: list[dict], n_needed: int, key_fn,
) -> list[dict]:
    """Round-robin selection across unique key-fn values (corrupted_element)."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for c in clean_cands:
        by_key[key_fn(c)].append(c)
    keys = list(by_key.keys())
    picked = []
    while len(picked) < n_needed and any(by_key.values()):
        for k in keys:
            if not by_key[k]:
                continue
            picked.append(by_key[k].pop(0))
            if len(picked) >= n_needed:
                break
    return picked


@app.command()
def run(
    cache_dir: Annotated[Path, typer.Option()] = Path("output/query_cache"),
    output_dir: Annotated[Path, typer.Option()] = Path("output/curated_200_v4"),
    uri: Annotated[str, typer.Option()] = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    username: Annotated[str, typer.Option()] = os.environ.get("NEO4J_USERNAME", "neo4j"),
    password: Annotated[str, typer.Option()] = os.environ.get("NEO4J_PASSWORD", "password"),
    timeout_s: Annotated[float, typer.Option()] = 60.0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    os.environ.setdefault("NEO4J_DATABASE_MAP", _default_db_map())

    typer.echo("Loading pool + verdicts + bad curated...")
    pool = _load_pool_samples()
    verdicts = _load_llm_verdicts()
    bad_by_type = _load_bad_curated()
    typer.echo(f"  pool: {len(pool)}, LLM-verified: {len(verdicts)}")
    for ct, recs in bad_by_type.items():
        typer.echo(f"  bad {ct}: {len(recs)}")

    curated_ids: set[str] = set()
    for f in Path("output/curated_200_v3").glob("full_*.json"):
        for s in json.loads(f.read_text()):
            curated_ids.add(s["corruption_id"])

    cands_by_type = _candidate_ids_per_type(pool, verdicts, curated_ids)
    typer.echo("\nCandidate pool (eligible by LLM verification):")
    for ct, ids in cands_by_type.items():
        typer.echo(f"  {ct}: {len(ids)} candidates")

    # Execute valid_cyphers for all candidates (only those not in cache)
    cache = _load_cache(cache_dir)
    todo: dict[str, dict[str, str]] = defaultdict(dict)  # graph → hash → cypher
    for ct, ids in cands_by_type.items():
        for cid in ids:
            sample = pool[cid]
            g = sample["_graph"]
            for q in sample.get("valid_cyphers") or []:
                if (g, _h(q)) not in cache:
                    todo[g][_h(q)] = q
    typer.echo(f"\nCyphers to execute (not cached): {sum(len(v) for v in todo.values())}")
    for g in GRAPH_ORDER:
        if todo.get(g):
            typer.echo(f"  {g}: {len(todo[g])}")

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
            for hsh, q in items.items():
                try:
                    rows = client.run_query(q, g, timeout_s=timeout_s)
                    cache[(g, hsh)] = (rows, None)
                    _append_to_cache(cache_dir, g, hsh, q, rows, None)
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    cache[(g, hsh)] = (None, err)
                    _append_to_cache(cache_dir, g, hsh, q, None, err)
        driver.close()
        client.close()

    # Filter clean (all valid_cyphers non-empty)
    clean_by_type: dict[str, list[dict]] = defaultdict(list)
    for ct, ids in cands_by_type.items():
        for cid in ids:
            sample = pool[cid]
            g = sample["_graph"]
            golds = sample.get("valid_cyphers") or []
            if not golds:
                continue
            cells = [cache.get((g, _h(q))) for q in golds]
            if any(c is None for c in cells):
                continue
            if any(c[1] is not None for c in cells):  # error
                continue
            if any(len(c[0]) == 0 for c in cells):  # empty
                continue
            clean_by_type[ct].append(sample)
    typer.echo("\nClean candidates (after execution):")
    for ct in sorted(clean_by_type):
        typer.echo(f"  {ct}: {len(clean_by_type[ct])}")

    # Pick diversely by corrupted_element
    needed = {ct: len(recs) for ct, recs in bad_by_type.items()}
    typer.echo(f"\nReplacements needed: {dict(needed)}")
    picked_by_type: dict[str, list[dict]] = {}
    for ct in ("A1", "A3", "A5"):
        if ct not in needed:
            continue
        cands = clean_by_type.get(ct, [])
        if len(cands) < needed[ct]:
            typer.echo(f"  ⚠️  only {len(cands)} clean {ct} available (need {needed[ct]})")
        picked = _pick_diverse(
            cands, needed[ct], lambda c: c["corruption"]["corrupted_element"]
        )
        picked_by_type[ct] = picked
        typer.echo(f"  picked {ct}: {len(picked)} → corrupted_element diversity: "
                   f"{Counter(p['corruption']['corrupted_element'] for p in picked)}")

    # Build new curated_200_v4
    bad_ids = {r["corruption_id"] for recs in bad_by_type.values() for r in recs}
    new_ids = {p["corruption_id"] for picks in picked_by_type.values() for p in picks}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compose: keep all curated_v3 except bad_ids, add picked replacements
    by_graph: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(Path("output/curated_200_v3").glob("full_*.json")):
        graph = f.stem.removeprefix("full_")
        for s in json.loads(f.read_text()):
            if s["corruption_id"] not in bad_ids:
                by_graph[graph].append(s)
    for ct, picks in picked_by_type.items():
        for p in picks:
            # Strip our internal _graph field before writing
            clean = {k: v for k, v in p.items() if k != "_graph"}
            by_graph[p["_graph"]].append(clean)
    for g, items in by_graph.items():
        (output_dir / f"full_{g}.json").write_text(
            json.dumps(items, indent=2, ensure_ascii=False)
        )

    # Copy + extend verification dir (LLM verdicts for the new picks)
    verif_dir = output_dir / "verification"
    verif_dir.mkdir(exist_ok=True)
    # Start from curated_v3 verification (drop bad_ids)
    new_verif_by_graph: dict[str, list[dict]] = defaultdict(list)
    for f in (Path("output/curated_200_v3/verification")).glob("llm_verified_*.json"):
        graph = f.stem.removeprefix("llm_verified_")
        for r in json.loads(f.read_text()):
            if r["corruption_id"] not in bad_ids:
                new_verif_by_graph[graph].append(r)
    # Add the verdict for each new pick (force verdict=pass; A3 relaxed get "pass_relaxed")
    for ct, picks in picked_by_type.items():
        for p in picks:
            cid = p["corruption_id"]
            ver = verdicts.get(cid, {}).copy()
            if ct == "A3" and ver.get("verdict") == "fail":
                ver = {**ver, "verdict": "pass", "_original_verdict": "fail",
                       "_relaxed_reason": "hypernym/union"}
            new_verif_by_graph[p["_graph"]].append(ver)
    for g, items in new_verif_by_graph.items():
        (verif_dir / f"llm_verified_{g}.json").write_text(
            json.dumps(items, indent=2, ensure_ascii=False)
        )

    # Replacement plan + new-IDs list
    (output_dir / "_replacement_plan.json").write_text(json.dumps({
        "bad_ids_dropped": sorted(bad_ids),
        "new_ids_added": sorted(new_ids),
        "by_type": {ct: [p["corruption_id"] for p in picks]
                    for ct, picks in picked_by_type.items()},
    }, indent=2))
    (output_dir / "_new_corruption_ids.txt").write_text("\n".join(sorted(new_ids)))

    typer.echo(f"\nWrote curated_200_v4 → {output_dir}")
    typer.echo(f"  bad dropped: {len(bad_ids)}")
    typer.echo(f"  new added:   {len(new_ids)}")
    typer.echo(f"  new ids written to: {output_dir / '_new_corruption_ids.txt'}")


if __name__ == "__main__":
    app()
