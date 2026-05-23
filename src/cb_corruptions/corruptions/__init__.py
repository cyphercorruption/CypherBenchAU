"""Corruption generator registry and base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cb_corruptions.schema import GraphInfo, Nl2CypherSample, PropertyGraphSchema

    from cb_corruptions.llm import LLM
    from cb_corruptions.models import CorruptedSample

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CORRUPTION_REGISTRY: dict[str, type[BaseCorruption]] = {}


def register_corruption(name: str):
    """Decorator to register a corruption class by its type code (e.g. 'A1')."""

    def decorator(cls: type[BaseCorruption]) -> type[BaseCorruption]:
        CORRUPTION_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseCorruption(ABC):
    """Common interface for all corruption generators."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.schema: PropertyGraphSchema | None = None

    def set_schema(self, schema: PropertyGraphSchema) -> None:
        """Store the current graph schema. Called by the pipeline before each graph."""
        self.schema = schema

    @abstractmethod
    def analyze(
        self,
        schema: PropertyGraphSchema,
        graph_info: GraphInfo,
    ) -> list:
        """Return corruption candidates from static schema analysis."""

    @abstractmethod
    def select_samples(
        self,
        samples: list[Nl2CypherSample],
        candidates: list,
    ) -> list[tuple[Nl2CypherSample, object]]:
        """Match benchmark samples to candidates."""

    @abstractmethod
    def corrupt(
        self,
        sample: Nl2CypherSample,
        candidate: object,
        graph_name: str,
    ) -> CorruptedSample | None:
        """Produce a corrupted sample using the LLM. Return None on failure."""


# ---------------------------------------------------------------------------
# Import all corruption modules to trigger registration
# ---------------------------------------------------------------------------

from cb_corruptions.corruptions import (  # noqa: E402, F401
    a1_relation_ambiguity,
    a2_property_ambiguity,
    a3_entity_type_ambiguity,
    a5_direction_ambiguity,
    u_unanswerable,
    u5_temporal_unanswerability,
)
