"""Cypher query element substitution for generating corrupted queries."""

from __future__ import annotations

import re


def replace_property(query: str, old_prop: str, new_prop: str) -> str:
    """Replace a property name in a Cypher query (dotted access and map syntax)."""
    # n.old_prop -> n.new_prop
    query = re.sub(
        rf"(\w+)\.{re.escape(old_prop)}\b",
        rf"\1.{new_prop}",
        query,
    )
    # {old_prop: ...} -> {new_prop: ...}
    query = re.sub(
        rf"\b{re.escape(old_prop)}\s*:",
        f"{new_prop}:",
        query,
    )
    return query


def replace_relation(query: str, old_rel: str, new_rel: str) -> str:
    """Replace a relationship type in a Cypher query."""
    return re.sub(
        rf":{re.escape(old_rel)}\b",
        f":{new_rel}",
        query,
    )


def replace_label(query: str, old_label: str, new_label: str) -> str:
    """Replace an entity label in a Cypher query."""
    return re.sub(
        rf":{re.escape(old_label)}\b",
        f":{new_label}",
        query,
    )
