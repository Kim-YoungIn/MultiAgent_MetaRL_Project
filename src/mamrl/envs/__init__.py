"""Multi-agent environments and task generators."""

from .active_probe_game import (
    ActiveProbeGame,
    ActiveProbeSpec,
    ROLE_EXPLOIT_0,
    ROLE_EXPLOIT_1,
    ROLE_PROBE,
    ROLE_SAFE,
    generate_task_suite,
)
from .base import Observation, StepResult

__all__ = [
    "ActiveProbeGame",
    "ActiveProbeSpec",
    "Observation",
    "ROLE_EXPLOIT_0",
    "ROLE_EXPLOIT_1",
    "ROLE_PROBE",
    "ROLE_SAFE",
    "StepResult",
    "generate_task_suite",
]
