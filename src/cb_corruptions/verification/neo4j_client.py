"""Thin Neo4j driver wrapper. One connection, switch database per graph."""

from __future__ import annotations

import logging
import os
from typing import Any

from neo4j import GraphDatabase, Result

logger = logging.getLogger(__name__)

# Suppress noisy Neo4j deprecation warnings
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# CypherBench database naming: each graph is a separate Neo4j database.
# Override with NEO4J_DATABASE_MAP='{"nba":"neo4j"}' if your instance differs.
_DEFAULT_DB_MAP: dict[str, str] = {}


def _load_db_map() -> dict[str, str]:
    raw = os.environ.get("NEO4J_DATABASE_MAP", "")
    if not raw:
        return _DEFAULT_DB_MAP
    import json

    return json.loads(raw)


class Neo4jClient:
    """Manages a single Neo4j driver and runs queries against per-graph databases."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._db_map = _load_db_map()

        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        logger.info("Neo4j driver created for %s", self._uri)

    def verify_connectivity(self) -> None:
        """Test that the Neo4j instance is reachable. Raises on failure."""
        self._driver.verify_connectivity()

    def _database_for(self, graph_name: str) -> str:
        return self._db_map.get(graph_name, graph_name)

    def run_query(
        self, cypher: str, graph_name: str, timeout_s: float = 60.0
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return rows as dicts.

        Raises on syntax errors, missing labels/properties, etc., and aborts
        with a transient error if the query exceeds ``timeout_s`` seconds —
        critical for benchmark evaluation, where pathological model
        predictions (e.g. variable-length paths with no bound) can spiral
        into effectively-infinite execution on large graphs.
        """
        database = self._database_for(graph_name)
        with self._driver.session(database=database) as session:
            with session.begin_transaction(timeout=timeout_s) as tx:
                result: Result = tx.run(cypher)
                return [dict(record) for record in result]

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
