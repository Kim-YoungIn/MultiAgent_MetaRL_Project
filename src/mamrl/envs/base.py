"""Base data structures for simultaneous two-player repeated games."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Publicly observable single-agent observation (no hidden-task leakage)."""

    prev_self_action: int  # -1 at turn 0
    prev_opp_action: int   # -1 at turn 0
    prev_self_reward: float
    is_start: bool
    turn: int


@dataclass(frozen=True)
class StepResult:
    """Joint transition outcome for one turn of a two-player game."""

    focal_action: int
    opp_action: int
    focal_reward: float
    opp_reward: float
    focal_obs: Observation
    opp_obs: Observation
    turn: int
