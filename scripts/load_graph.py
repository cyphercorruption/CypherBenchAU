#!/usr/bin/env python3
"""Load a CypherBench SimpleKG JSON file into Neo4j.

Usage:
    python scripts/load_graph.py nba
    python scripts/load_graph.py nba movie soccer
    python scripts/load_graph.py --all

Requires: neo4j Python driver (pip install neo4j)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMPLEKG_DIR = PROJECT_ROOT / "benchmark" / "graphs" / "simplekg"

ALL_GRAPHS = [
    "art", "biology", "company", "fictional_character", "flight_accident",
    "geography", "movie", "nba", "politics", "soccer", "terrorist_attack",
]

BATCH_SIZE = 500


def load_graph(tx_func, graph_name: str, database: str) -> None:
    """Load a single graph from its SimpleKG JSON into Neo4j."""
    path = SIMPLEKG_DIR / f"{graph_name}_simplekg.json"
    if not path.exists():
        logger.error("File not found: %s", path)
        return

    logger.info("Loading %s from %s ...", graph_name, path.name)
    with open(path) as f:
        data = json.load(f)

    entities = data["entities"]
    relations = data["relations"]
    logger.info("  %d entities, %d relations", len(entities), len(relations))

    # -- Clear existing data in this database (in batches to avoid OOM) --
    with tx_func(database=database) as session:
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            deleted = result.single()["deleted"]
            if deleted == 0:
                break
            logger.info("  Deleted %d nodes...", deleted)
        logger.info("  Cleared existing data")

    # -- Create indexes for fast lookups --
    labels = {e["label"] for e in entities}
    with tx_func(database=database) as session:
        for label in labels:
            session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.eid)"
            )
        logger.info("  Created %d indexes", len(labels))

    # -- Load entities in batches --
    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i : i + BATCH_SIZE]
        with tx_func(database=database) as session:
            session.run(
                """
                UNWIND $batch AS e
                CALL apoc.create.node([e.label], apoc.map.merge(
                    {eid: e.eid, name: e.name},
                    e.properties
                )) YIELD node
                RETURN count(node)
                """,
                batch=batch,
            )
        if (i + BATCH_SIZE) % 2000 == 0 or i + BATCH_SIZE >= len(entities):
            logger.info("  Entities: %d / %d", min(i + BATCH_SIZE, len(entities)), len(entities))

    # -- Pre-process relations: extract labels from IDs for index-backed lookups --
    # IDs are like "Person#Q123" — the part before '#' is the label
    eid_to_label = {e["eid"]: e["label"] for e in entities}
    for r in relations:
        r["subj_label"] = eid_to_label.get(r["subj_id"], "")
        r["obj_label"] = eid_to_label.get(r["obj_id"], "")

    # Group relations by (subj_label, obj_label) so each batch uses a single
    # label-aware MATCH that hits the indexes
    from collections import defaultdict
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in relations:
        groups[(r["subj_label"], r["obj_label"])].append(r)

    loaded = 0
    for (subj_label, obj_label), group_rels in groups.items():
        for i in range(0, len(group_rels), BATCH_SIZE):
            batch = group_rels[i : i + BATCH_SIZE]
            with tx_func(database=database) as session:
                session.run(
                    f"""
                    UNWIND $batch AS r
                    MATCH (a:`{subj_label}` {{eid: r.subj_id}})
                    MATCH (b:`{obj_label}` {{eid: r.obj_id}})
                    CALL apoc.create.relationship(a, r.label, r.properties, b)
                    YIELD rel
                    RETURN count(rel)
                    """,
                    batch=batch,
                )
            loaded += len(batch)
            if loaded % 10000 < BATCH_SIZE:
                logger.info("  Relations: %d / %d", loaded, len(relations))

    logger.info("  Done loading %s", graph_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CypherBench graphs into Neo4j")
    parser.add_argument("graphs", nargs="*", help="Graph names to load")
    parser.add_argument("--all", action="store_true", help="Load all 11 graphs")
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument(
        "--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Target database (Community edition only has 'neo4j')",
    )
    args = parser.parse_args()

    graphs = ALL_GRAPHS if args.all else args.graphs
    if not graphs:
        parser.error("Specify graph names or use --all")

    for g in graphs:
        if g not in ALL_GRAPHS:
            parser.error(f"Unknown graph: {g}. Valid: {', '.join(ALL_GRAPHS)}")

    driver = GraphDatabase.driver(args.uri, auth=(args.username, args.password))
    try:
        driver.verify_connectivity()
        logger.info("Connected to %s", args.uri)
    except Exception as e:
        logger.error("Cannot connect to Neo4j: %s", e)
        sys.exit(1)

    for graph_name in graphs:
        load_graph(driver.session, graph_name, database=args.database)

    driver.close()
    logger.info("All done.")


if __name__ == "__main__":
    main()
