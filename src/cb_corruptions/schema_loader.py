"""Load graph schemas and benchmark samples from local data files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cb_corruptions.schema import (
    DataType,
    EntitySchema,
    GraphInfo,
    Nl2CypherSample,
    PropertyGraphSchema,
    RelationSchema,
)

logger = logging.getLogger(__name__)

# Project root: two levels up from this file (src/cb_corruptions/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All 11 CypherBench graphs
ALL_GRAPHS = [
    "art",
    "biology",
    "company",
    "fictional_character",
    "flight_accident",
    "geography",
    "movie",
    "nba",
    "politics",
    "soccer",
    "terrorist_attack",
]


def load_schema(graph_name: str) -> PropertyGraphSchema:
    """Load a PropertyGraphSchema from a simplekg JSON file.

    The simplekg file has a top-level "schema" key containing entity and
    relation type definitions with string-encoded data types.
    """
    path = PROJECT_ROOT / "benchmark" / "graphs" / "simplekg" / f"{graph_name}_simplekg.json"
    if not path.exists():
        raise FileNotFoundError(f"SimpleKG file not found: {path}")

    with open(path) as f:
        raw_schema = json.load(f)["schema"]

    entities = [
        EntitySchema(
            label=e["label"],
            properties={"name": DataType.STR}
            | {k: DataType.from_simplekg_type(v) for k, v in e.get("properties", {}).items()},
        )
        for e in raw_schema["entities"]
    ]

    relations = [
        RelationSchema(
            label=r["label"],
            subj_label=r["subj_label"],
            obj_label=r["obj_label"],
            properties={k: DataType.from_simplekg_type(v) for k, v in r.get("properties", {}).items()},
        )
        for r in raw_schema["relations"]
    ]

    return PropertyGraphSchema(name=graph_name, entities=entities, relations=relations)


def load_all_schemas(graphs: list[str]) -> dict[str, PropertyGraphSchema]:
    """Load schemas for the requested graphs. Skip missing files with a warning."""
    schemas: dict[str, PropertyGraphSchema] = {}
    for graph in graphs:
        try:
            schemas[graph] = load_schema(graph)
        except FileNotFoundError:
            logger.warning("Schema not found for graph '%s', skipping", graph)
    return schemas


def load_graph_info() -> dict[str, GraphInfo]:
    """Load relation metadata for all graphs from graph_info.json."""
    path = PROJECT_ROOT / "graph_info.json"
    with open(path) as f:
        raw = json.load(f)
    return {name: GraphInfo(**info) for name, info in raw.items()}


def load_benchmark(benchmark_path: str) -> list[Nl2CypherSample]:
    """Load benchmark samples from a JSON file or directory.

    If benchmark_path points to a directory, loads all .json files in it.
    If it points to a file, loads that single file.
    """
    path = PROJECT_ROOT / benchmark_path
    if path.is_dir():
        samples: list[Nl2CypherSample] = []
        for f in sorted(path.glob("*.json")):
            with open(f) as fh:
                samples.extend(Nl2CypherSample(**item) for item in json.load(fh))
        return samples
    with open(path) as f:
        return [Nl2CypherSample(**item) for item in json.load(f)]
