"""Regex-based parser for CypherBench gold_cypher strings.

CypherBench gold queries follow predictable template patterns, e.g.:
  MATCH (n:Movie {name: 'Inception'})-[r0:directedBy]->(m0:Person) RETURN m0.name
  MATCH (n:Player)-[r0:playsFor]->(m0:Team) WHERE r0.start_year <= 2020 RETURN n.name

This module extracts structured info without a full Cypher parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedCypher:
    """Structured info extracted from a gold_cypher string."""

    # Entity labels: {"n": "Movie", "m0": "Person", ...}
    node_labels: dict[str, str] = field(default_factory=dict)

    # Relation types: {"r0": "directedBy", ...}
    relation_types: dict[str, str] = field(default_factory=dict)

    # Properties accessed in RETURN or WHERE: ["n.name", "m0.height_cm", "r0.start_year"]
    property_accesses: list[str] = field(default_factory=list)

    # Properties used in WHERE filters specifically: ["r0.start_year", "n.runtime_minute"]
    where_properties: list[str] = field(default_factory=list)

    # Whether the query has a WHERE clause
    has_where: bool = False

    # Raw cypher
    raw: str = ""


def parse_cypher(cypher: str) -> ParsedCypher:
    """Extract labels, relations, and property accesses from a gold_cypher string."""
    result = ParsedCypher(raw=cypher)

    # Extract node labels: (var:Label) or (var:Label {prop: value})
    for var, label in re.findall(r"\((\w+):(\w+)", cypher):
        result.node_labels[var] = label

    # Extract relation types: [var:RelType]
    for var, rel_type in re.findall(r"\[(\w+):(\w+)\]", cypher):
        result.relation_types[var] = rel_type

    # Extract all property accesses: var.property
    result.property_accesses = re.findall(r"\b(\w+\.\w+)\b", cypher)

    # Check for WHERE clause and extract properties used in filters
    where_match = re.search(r"\bWHERE\b(.+?)(?:\bRETURN\b|\bWITH\b|$)", cypher, re.IGNORECASE)
    if where_match:
        result.has_where = True
        where_clause = where_match.group(1)
        result.where_properties = re.findall(r"\b(\w+\.\w+)\b", where_clause)

    return result


def get_entity_labels(cypher: str) -> set[str]:
    """Extract all entity type labels from a Cypher query."""
    return set(re.findall(r"\(\w+:(\w+)", cypher))


def get_relation_types(cypher: str) -> set[str]:
    """Extract all relation type labels from a Cypher query."""
    return set(re.findall(r"\[\w+:(\w+)\]", cypher))


def get_returned_properties(cypher: str) -> list[tuple[str, str]]:
    """Extract (var, property) pairs from the RETURN clause.

    E.g. "RETURN n.name, n.height_cm" -> [("n", "name"), ("n", "height_cm")]
    """
    return_match = re.search(r"\bRETURN\b(.+)", cypher, re.IGNORECASE)
    if not return_match:
        return []
    return_clause = return_match.group(1)
    return re.findall(r"\b(\w+)\.(\w+)\b", return_clause)


def get_where_properties(cypher: str) -> list[tuple[str, str]]:
    """Extract (var, property) pairs from the WHERE clause."""
    where_match = re.search(r"\bWHERE\b(.+?)(?:\bRETURN\b|\bWITH\b|$)", cypher, re.IGNORECASE)
    if not where_match:
        return []
    where_clause = where_match.group(1)
    return re.findall(r"\b(\w+)\.(\w+)\b", where_clause)


def has_optional_match(cypher: str) -> bool:
    """Check if the query contains an OPTIONAL MATCH clause."""
    return bool(re.search(r"\bOPTIONAL\s+MATCH\b", cypher, re.IGNORECASE))


def has_union(cypher: str) -> bool:
    """Check if the query contains a UNION clause."""
    return bool(re.search(r"\bUNION\b", cypher, re.IGNORECASE))


def is_in_optional_match(cypher: str, element: str) -> bool:
    """Check if an element (relation type or label) appears inside an OPTIONAL MATCH clause."""
    for m in re.finditer(r"\bOPTIONAL\s+MATCH\b(.+?)(?:\bWITH\b|\bWHERE\b|\bRETURN\b|$)", cypher, re.IGNORECASE):
        if element in m.group(1):
            return True
    return False


def is_in_union_branch(cypher: str, element: str) -> bool:
    """Check if an element only appears in one branch of a UNION query.

    If the element is only in one branch, corrupting it leaves the other
    branch intact — the query still returns results.
    """
    if not has_union(cypher):
        return False
    # Split on UNION (handling CALL { ... UNION ... })
    branches = re.split(r"\bUNION\b", cypher, flags=re.IGNORECASE)
    branches_with_element = [b for b in branches if element in b]
    return len(branches_with_element) < len(branches)


def is_return_only_property(cypher: str, prop: str) -> bool:
    """Check if a property only appears in non-filtering positions (RETURN/ORDER BY).

    Replacing such a property with a fake one just returns null or changes
    sort order — the query still produces rows. Only properties in WHERE
    or MATCH filters are structurally essential for filtering.
    """
    # Check if property appears in WHERE
    where_match = re.search(r"\bWHERE\b(.+?)(?:\bRETURN\b|\bWITH\b|$)", cypher, re.IGNORECASE)
    if where_match and prop in where_match.group(1):
        return False

    # Check if property appears in a MATCH filter (e.g., {prop: value})
    match_filter = re.search(rf"\{{\s*{re.escape(prop)}\s*:", cypher)
    if match_filter:
        return False

    return True


def is_filtering_property(cypher: str, prop: str) -> bool:
    """Check if a property appears in a filtering position (WHERE or MATCH filter).

    Properties in these positions are structurally essential — replacing them
    causes the query to return 0 rows.
    """
    return not is_return_only_property(cypher, prop)
