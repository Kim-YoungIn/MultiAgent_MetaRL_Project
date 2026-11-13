"""Immutable reward tensors with global player/action axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .common import finite_reward, validate_n


@dataclass(frozen=True, slots=True, init=False)
class RewardTensor:
    """A dense, immutable ``(2, n, n, n, n)`` reward tensor.

    Values are copied from a flat row-major iterable. The final axis varies
    fastest; all action axes are in global player order.
    """

    n: int
    values: tuple[float, ...]

    def __init__(self, n: int, values: Iterable[float]) -> None:
        n = validate_n(n)
        try:
            copied = tuple(values)
        except TypeError as exc:
            raise TypeError("values must be a flat iterable of finite real rewards") from exc
        expected = 2 * n**4
        if len(copied) != expected:
            raise ValueError(f"expected {expected} flat reward values, got {len(copied)}")
        copied = tuple(finite_reward(value) for value in copied)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "values", copied)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (2, self.n, self.n, self.n, self.n)

    def __getitem__(self, index: tuple[int, int, int, int, int]) -> float:
        if not isinstance(index, tuple):
            raise TypeError("tensor index must be a tuple of five integers")
        if len(index) != 5:
            raise IndexError("tensor index must contain exactly five axes")
        offset = 0
        for axis, (component, size) in enumerate(zip(index, self.shape)):
            if isinstance(component, bool) or not isinstance(component, int):
                raise TypeError(f"tensor index for axis {axis} must be an integer")
            if not 0 <= component < size:
                raise IndexError(f"tensor index for axis {axis} out of range")
            offset = offset * size + component
        return self.values[offset]
