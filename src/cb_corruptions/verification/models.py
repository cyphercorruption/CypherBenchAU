"""Data models for verification results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class AmbiguityVerdict(BaseModel):
    """Result of verifying an ambiguity corruption."""

    status: VerificationStatus
    results_per_cypher: dict[str, list[dict[str, Any]]]
    all_results_differ: bool
    detail: str


class UnanswerabilityVerdict(BaseModel):
    """Result of verifying an unanswerability corruption."""

    status: VerificationStatus
    query_executed: str
    returned_rows: int
    error_message: str | None = None
    error_category: str | None = None  # e.g. "missing_property", "missing_label", ...
    detail: str


class VerificationResult(BaseModel):
    """Verification outcome for a single corrupted sample."""

    corruption_id: str
    corruption_type: str
    graph: str
    status: VerificationStatus
    ambiguity: AmbiguityVerdict | None = None
    unanswerability: UnanswerabilityVerdict | None = None


class EvalResult(BaseModel):
    """Precision & recall outcome for a single benchmark sample."""

    precision: float
    recall: float
    n_predictions: int
    n_targets: int
    n_pred_unique: int
    n_target_unique: int
    n_matched: int
    failed_predictions: list[str] = []
