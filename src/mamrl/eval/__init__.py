"""Evaluation, Gate-0 Oracle audit, and distribution-aware metrics."""

from .metrics import summarize_method, summarize_paired_contrast
from .oracle import OracleAuditResult, audit_task_suite, exact_two_turn_audit
from .protocol import EpisodeOutcome, evaluate_suite

__all__ = [
    "EpisodeOutcome",
    "OracleAuditResult",
    "audit_task_suite",
    "evaluate_suite",
    "exact_two_turn_audit",
    "summarize_method",
    "summarize_paired_contrast",
]
