"""Curate a 200-sample evaluation subset: 100 ambiguous + 100 unanswerable.

Allocation:
  A1=A2=A3=A5 = 25  (100 ambiguous)
  U1..U5      = 20  (100 unanswerable)

Source pool:
  output/final_output/full_<graph>.json
  output/final_output/verification/llm_verified_<graph>.json

Output:
  output/curated_200_v3/full_<graph>.json     (subset, original schema)
  output/curated_200_v3/verification/...      (matching verifier records)
  output/curated_200_v3/metadata.json
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path("output/final_output")
OUT = Path("output/curated_200_v3")

# Manual whitelist of A3 samples judged to have TRUE entity-level ambiguity
# (not hypernym/union semantics). 5 solid + 2 borderline = 7.
A3_TRUE_AMBIGUITY = {
    "A3-art-2ee224f8",          # "art piece The Kiss" — Painting (Klimt) AND Sculpture (Rodin)
    "A3-art-6a9a900d",          # "masterpiece The Cathedral" — Sculpture (Rodin) AND Painting (Monet)
    "A3-geography-2942a61c",    # "waterways part of Tobol basin" — schema-permissive River+Lake
    "A3-geography-672a0ade",    # "body of water Nerda" — schema-permissive
    "A3-geography-6b763b04",    # "Tayma water body" — schema-permissive
    "A3-art-82471014",          # "land art" — domain-biased but ambiguous
    "A3-geography-9f0dea9a",    # "water body Vyazovka/Tura flow into" — schema-permissive
}

# A1/A2 samples to exclude based on manual review:
#   - bug-cypher pairs (identical valid_cyphers, no real distinction)
#   - hypernym instead of exclusive ambiguity
#   - awkward English
A1_EXCLUDE = {
    "A1-fictional_character-a45d47e3",  # "parent" — hypernym, not exclusive
    "A1-fictional_character-ffb7c3e0",  # "relative" — too broad
}
A2_EXCLUDE = {
    "A2-art-1b69c511",   # buggy: both cyphers RETURN n.name (identical)
    "A2-art-532f4c6b",   # buggy: both cyphers RETURN n (identical)
    "A2-art-b7bcc338",   # "What is the birth of..." — broken English
}

ALLOC = {
    "A1": 25, "A2": 25, "A3": 25, "A5": 25,
    "U1": 20, "U2": 20, "U3": 20, "U4": 20, "U5": 20,
}


def load_pool() -> tuple[dict[str, dict], dict[str, dict]]:
    """Load all source samples + their verification records, keyed by corruption_id."""
    samples: dict[str, dict] = {}
    for f in sorted((ROOT).glob("full_*.json")):
        graph = f.stem.removeprefix("full_")
        for s in json.loads(f.read_text()):
            s["_graph"] = graph
            samples[s["corruption_id"]] = s

    ver: dict[str, dict] = {}
    for f in sorted((ROOT / "verification").glob("llm_verified_*.json")):
        for r in json.loads(f.read_text()):
            ver[r["corruption_id"]] = r
    return samples, ver


def signature(sample: dict) -> tuple[str, str]:
    """Cluster key: (original_element, corrupted_element)."""
    c = sample["corruption"]
    return (
        str(c.get("original_element") or c.get("reason_unanswerable") or ""),
        str(c.get("corrupted_element") or ""),
    )


def quality_key(sample: dict, ver: dict) -> tuple:
    """Sort key (descending preference).

    Rank: pass first, then high effectiveness, then high naturalness,
    then no-issues, then deterministic hash for stable tie-breaking.
    """
    v = ver.get(sample["corruption_id"], {})
    is_pass = v.get("verdict") == "pass"
    eff = v.get("effectiveness") or 0
    nat = v.get("naturalness") or 0
    no_issues = 0 if (v.get("issues") or []) else 1
    h = int(hashlib.sha1(sample["corruption_id"].encode()).hexdigest()[:8], 16)
    return (-int(is_pass), -eff, -nat, -no_issues, h)


def round_robin_pick(
    candidates: list[dict],
    target: int,
    ver: dict,
) -> list[dict]:
    """Cluster by signature; iteratively take best-of-cluster until target reached."""
    if len(candidates) <= target:
        return list(candidates)

    by_sig: dict[tuple, list[dict]] = collections.defaultdict(list)
    for s in candidates:
        by_sig[signature(s)].append(s)
    for sig in by_sig:
        by_sig[sig].sort(key=lambda s: quality_key(s, ver))

    selected: list[dict] = []
    seen_ids: set[str] = set()
    while len(selected) < target:
        progress = False
        # sort clusters by best-remaining quality so the round-robin picks
        # globally good samples first across clusters
        sigs_sorted = sorted(
            (sig for sig in by_sig if by_sig[sig]),
            key=lambda sig: quality_key(by_sig[sig][0], ver),
        )
        for sig in sigs_sorted:
            if not by_sig[sig]:
                continue
            s = by_sig[sig].pop(0)
            if s["corruption_id"] in seen_ids:
                continue
            selected.append(s)
            seen_ids.add(s["corruption_id"])
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    return selected


def select_a3(samples: dict, ver: dict) -> list[dict]:
    """A3 = 7 manually-whitelisted (true ambiguity) + 18 hypernym-fail (diverse)."""
    tier1 = [samples[cid] for cid in A3_TRUE_AMBIGUITY if cid in samples]
    tier1_ids = {s["corruption_id"] for s in tier1}

    # Tier 2 candidates: A3 fail with naturalness >= 4, not in tier1
    candidates = []
    for s in samples.values():
        if s["corruption"]["corruption_type"] != "A3":
            continue
        if s["corruption_id"] in tier1_ids:
            continue
        v = ver.get(s["corruption_id"], {})
        if v.get("verdict") == "fail" and (v.get("naturalness") or 0) >= 4:
            candidates.append(s)

    tier2 = round_robin_pick(candidates, 18, ver)
    return tier1 + tier2


def select_ambiguous(ctype: str, target: int, samples: dict, ver: dict, exclude: set) -> list[dict]:
    """A1/A2/A5: pass-verified, non-empty valid_cyphers (A1/A2), diverse cluster pick."""
    candidates = []
    for s in samples.values():
        if s["corruption"]["corruption_type"] != ctype:
            continue
        if s["corruption_id"] in exclude:
            continue
        v = ver.get(s["corruption_id"], {})
        if v.get("verdict") != "pass":
            continue
        if ctype in ("A1", "A2") and not s.get("valid_cyphers"):
            continue
        candidates.append(s)
    return round_robin_pick(candidates, target, ver)


def select_unanswerable(ctype: str, target: int, samples: dict, ver: dict) -> list[dict]:
    """U1..U5: pass-verified, diverse cluster pick (signature = original_element if present
    else reason_unanswerable, vs corrupted_element)."""
    candidates = []
    for s in samples.values():
        if s["corruption"]["corruption_type"] != ctype:
            continue
        v = ver.get(s["corruption_id"], {})
        if v.get("verdict") != "pass":
            continue
        candidates.append(s)
    return round_robin_pick(candidates, target, ver)


def main() -> None:
    samples, ver = load_pool()
    print(f"Source pool: {len(samples)} samples, {len(ver)} verification records")

    selection: list[dict] = []
    selection += select_ambiguous("A1", ALLOC["A1"], samples, ver, A1_EXCLUDE)
    selection += select_ambiguous("A2", ALLOC["A2"], samples, ver, A2_EXCLUDE)
    selection += select_a3(samples, ver)
    selection += select_ambiguous("A5", ALLOC["A5"], samples, ver, set())
    for u in ("U1", "U2", "U3", "U4", "U5"):
        selection += select_unanswerable(u, ALLOC[u], samples, ver)

    # Validate completeness of every selected sample
    REQUIRED = {
        "corruption_id", "corruption", "original_qid", "original_graph",
        "original_nl_question", "original_gold_cypher", "corrupted_nl_question",
        "valid_cyphers", "expected_answer",
    }
    for s in selection:
        missing = REQUIRED - set(s.keys())
        if missing:
            raise RuntimeError(f"Sample {s['corruption_id']} missing fields: {missing}")
        ctype = s["corruption"]["corruption_type"]
        if ctype.startswith("A") and not s["valid_cyphers"]:
            raise RuntimeError(f"Ambiguous sample {s['corruption_id']} has empty valid_cyphers")
        if ctype.startswith("A") and s["expected_answer"] != "AMBIGUOUS":
            raise RuntimeError(f"Ambiguous sample {s['corruption_id']} expected_answer != AMBIGUOUS")
        if ctype.startswith("U") and s["expected_answer"] != "UNANSWERABLE":
            raise RuntimeError(f"Unanswerable sample {s['corruption_id']} expected_answer != UNANSWERABLE")

    # Group by graph and write
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification").mkdir(exist_ok=True)
    by_graph: dict[str, list[dict]] = collections.defaultdict(list)
    by_graph_ver: dict[str, list[dict]] = collections.defaultdict(list)
    for s in selection:
        g = s["_graph"]
        clean = {k: v for k, v in s.items() if not k.startswith("_")}
        by_graph[g].append(clean)
        if s["corruption_id"] in ver:
            by_graph_ver[g].append(ver[s["corruption_id"]])

    for g, items in by_graph.items():
        (OUT / f"full_{g}.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
        (OUT / "verification" / f"llm_verified_{g}.json").write_text(
            json.dumps(by_graph_ver[g], indent=2, ensure_ascii=False)
        )

    # Stats
    type_counts = collections.Counter(s["corruption"]["corruption_type"] for s in selection)
    graph_counts = collections.Counter(s["_graph"] for s in selection)
    diversity = {}
    for ctype in sorted(type_counts):
        sigs = collections.Counter(signature(s) for s in selection if s["corruption"]["corruption_type"] == ctype)
        diversity[ctype] = {"unique_signatures": len(sigs), "samples": type_counts[ctype]}

    metadata = {
        "total_samples": len(selection),
        "allocation": dict(ALLOC),
        "samples_per_type": dict(type_counts),
        "samples_per_graph": dict(graph_counts),
        "diversity": diversity,
        "rules": {
            "A1": "pass-verified + non-empty valid_cyphers; manually excluded 2 hypernym samples (parent, relative)",
            "A2": "pass-verified + non-empty valid_cyphers; manually excluded 2 buggy-cypher pairs and 1 awkward English",
            "A3": "7 manually-whitelisted true-ambiguity samples + 18 hypernym-fail samples picked by signature diversity",
            "A5": "pass-verified; diverse cluster pick (limited by ~9-15 unique signatures in source)",
            "U1-U5": "pass-verified; diverse cluster pick on corruption signature",
            "ranking_within_cluster": "verdict=pass DESC, effectiveness DESC, naturalness DESC, no-issues DESC, hash",
        },
        "a3_true_ambiguity_whitelist": sorted(A3_TRUE_AMBIGUITY),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nWrote {len(selection)} samples to {OUT}")
    print(f"  per type: {dict(type_counts)}")
    print(f"  per graph: {dict(graph_counts)}")
    print(f"  diversity: {diversity}")


if __name__ == "__main__":
    main()
