"""Shared input checks. Booleans are never actions, seeds, or rewards."""

import math
from numbers import Real


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return value


def validate_n(value: object) -> int:
    value = _integer(value, "n")
    if value < 2:
        raise ValueError("n must be at least 2")
    return value


def validate_seed(value: object) -> int | None:
    if value is None:
        return None
    value = _integer(value, "seed")
    if value < 0:
        raise ValueError("seed must be nonnegative")
    return value


def validate_action(value: object, n: int) -> int:
    value = _integer(value, "action")
    if not 0 <= value < n:
        raise ValueError(f"action must be in [0, {n})")
    return value


def positive_integer(value: object, name: str) -> int:
    value = _integer(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def finite_reward(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reward/coefficient must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("reward/coefficient must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("reward/coefficient must be finite")
    return result
