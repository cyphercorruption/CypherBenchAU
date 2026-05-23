"""Verification module: execute corrupted queries against Neo4j to validate correctness."""

from cb_corruptions.verification.evaluator import evaluate
from cb_corruptions.verification.models import (
    AmbiguityVerdict,
    EvalResult,
    UnanswerabilityVerdict,
    VerificationResult,
    VerificationStatus,
)
from cb_corruptions.verification.verifier import Verifier

__all__ = [
    "AmbiguityVerdict",
    "EvalResult",
    "UnanswerabilityVerdict",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    "evaluate",
]
