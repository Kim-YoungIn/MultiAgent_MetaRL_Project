"""A small, compositional logical rule generator and deterministic compiler.

Rules describe reward functions, not policies. Sampling happens only when a
specification is generated, so evaluating or compiling a specification never
changes the game or consumes random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from typing import Iterable

from .common import finite_reward, validate_action, validate_n, validate_seed
from .tensor import RewardTensor


_KINDS = frozenset(
    {"coordination", "repeat_own", "copy_opponent", "action_bonus", "cyclic_zero_sum",
     "probe_repeat", "probe_copy"}
)
_FAMILIES = frozenset({"zero_sum", "coordination", "memory_one", "mixed"})


@dataclass(frozen=True, slots=True)
class RuleTerm:
    """One additive condition, applied in each player's local perspective."""

    kind: str
    coefficient: float = 1.0
    action: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise TypeError("rule term kind must be a string")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown rule term kind: {self.kind!r}")
        object.__setattr__(self, "coefficient", finite_reward(self.coefficient))
        if self.kind == "action_bonus":
            if isinstance(self.action, bool) or not isinstance(self.action, int):
                raise TypeError("action_bonus requires an integer action")
            if self.action < 0:
                raise ValueError("action_bonus action must be non-negative")
        elif self.action is not None:
            raise ValueError("action is only used by action_bonus")


@dataclass(frozen=True, slots=True, init=False)
class RuleSpec:
    """A fully specified additive rule for a given action count.

    Term iterables are copied so the rule remains fixed even if a caller later
    changes the list used at construction.
    """

    n: int
    terms: tuple[RuleTerm, ...]

    def __init__(self, n: int, terms: Iterable[RuleTerm]) -> None:
        n = validate_n(n)
        try:
            copied = tuple(terms)
        except TypeError as exc:
            raise TypeError("terms must be an iterable of RuleTerm") from exc
        for term in copied:
            if not isinstance(term, RuleTerm):
                raise TypeError("each rule term must be a RuleTerm")
            if term.kind == "action_bonus" and term.action >= n:
                raise ValueError("action_bonus action must be less than n")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "terms", copied)


def generate_rule(n: int = 3, seed: int | None = None, family: str = "mixed") -> RuleSpec:
    """Generate a reproducible rule using local randomness.

    Coefficient magnitudes are sampled from {0,1,2,3,4}. In the mixed family
    their signs are also sampled, enabling penalties. The mixed family selects
    a nonempty combination of distinct primitive conditions and independently
    chooses each coefficient (and action target, where relevant).
    """

    n = validate_n(n)
    seed = validate_seed(seed)
    if not isinstance(family, str):
        raise TypeError("rule family must be a string")
    if family not in _FAMILIES:
        raise ValueError(f"unknown rule family: {family!r}")
    rng = random.Random(seed)
    if family == "zero_sum":
        terms = [RuleTerm("cyclic_zero_sum", rng.randrange(5))]
    elif family == "coordination":
        terms = [RuleTerm("coordination", rng.randrange(5))]
    elif family == "memory_one":
        terms = [RuleTerm("repeat_own", rng.randrange(5)),
                 RuleTerm("copy_opponent", rng.randrange(5))]
    else:
        # Fixed ordering here matters: a set iteration order varies across processes.
        kinds = ("coordination", "repeat_own", "copy_opponent", "action_bonus", "cyclic_zero_sum")
        selected = rng.sample(kinds, rng.randint(1, len(kinds)))
        terms = []
        for kind in selected:
            coefficient = rng.randrange(5) * rng.choice((-1, 1))
            action = rng.randrange(n) if kind == "action_bonus" else None
            terms.append(RuleTerm(kind, coefficient, action))
    return RuleSpec(n, terms)


def _joint_actions(values: Iterable[int], n: int, name: str) -> tuple[int, int]:
    try:
        copied = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a pair of actions") from exc
    if len(copied) != 2:
        raise ValueError(f"{name} must contain exactly two actions")
    return (validate_action(copied[0], n), validate_action(copied[1], n))


def evaluate_rule(
    spec: RuleSpec, previous: Iterable[int], actions: Iterable[int]
) -> tuple[float, float]:
    """Evaluate the fully specified rule for one previous/current joint action."""

    if not isinstance(spec, RuleSpec):
        raise TypeError("spec must be a RuleSpec")
    previous = _joint_actions(previous, spec.n, "previous")
    actions = _joint_actions(actions, spec.n, "actions")
    rewards = [0.0, 0.0]
    for player in (0, 1):
        opponent = 1 - player
        own = actions[player]
        other = actions[opponent]
        for term in spec.terms:
            coefficient = term.coefficient
            if term.kind == "coordination" and own == other:
                rewards[player] += coefficient
            elif term.kind == "repeat_own" and own == previous[player]:
                rewards[player] += coefficient
            elif term.kind == "copy_opponent" and own == previous[opponent]:
                rewards[player] += coefficient
            elif term.kind == "action_bonus" and own == term.action:
                rewards[player] += coefficient
            elif term.kind == "cyclic_zero_sum":
                delta = (own - other) % spec.n
                if 0 < 2 * delta < spec.n:
                    rewards[player] += coefficient
                elif 2 * delta > spec.n:
                    rewards[player] -= coefficient
            elif term.kind in ("probe_repeat", "probe_copy"):
                reference = previous[player] if term.kind == "probe_repeat" else previous[opponent]
                other_reference = previous[opponent] if term.kind == "probe_repeat" else previous[player]
                if own == reference:
                    rewards[player] += coefficient
                elif own == other_reference:
                    rewards[player] -= 3 * coefficient
                else:
                    rewards[player] -= (.25 if term.kind == "probe_repeat" else .5) * coefficient
        rewards[player] = finite_reward(rewards[player])
    return (rewards[0], rewards[1])


def compile_rule(spec: RuleSpec) -> RewardTensor:
    """Compile a fixed rule into global-player, row-major reward axes."""

    # Import lazily to keep the original rule API independent of the new grammar.
    from .logical_rules import LogicalRule, compile_logical_rule
    if isinstance(spec, LogicalRule):
        return compile_logical_rule(spec)
    from .relational_rules import RelationalRule, compile_relational_rule
    if isinstance(spec, RelationalRule):
        return compile_relational_rule(spec)
    if not isinstance(spec, RuleSpec):
        raise TypeError("spec must be a RuleSpec, LogicalRule or RelationalRule")
    n = spec.n
    values = []
    for player in (0, 1):
        for prev_0, prev_1, now_0, now_1 in product(range(n), repeat=4):
            rewards = evaluate_rule(spec, (prev_0, prev_1), (now_0, now_1))
            values.append(rewards[player])
    return RewardTensor(n, values)


def matrix_to_tensor(payoffs: Iterable[Iterable[Iterable[float]]]) -> RewardTensor:
    """Replicate static ``(2,n,n)`` global-action payoffs over previous states."""

    try:
        players = tuple(payoffs)
    except TypeError as exc:
        raise TypeError("payoffs must be an iterable with two player matrices") from exc
    if len(players) != 2:
        raise ValueError("payoffs must contain exactly two player matrices")
    try:
        matrices = tuple(tuple(tuple(row) for row in player) for player in players)
    except TypeError as exc:
        raise TypeError("each player matrix must be a two-dimensional iterable") from exc
    n = validate_n(len(matrices[0]))
    if any(len(matrix) != n or any(len(row) != n for row in matrix) for matrix in matrices):
        raise ValueError("player matrices must both have shape (n, n)")
    matrices = tuple(tuple(tuple(finite_reward(v) for v in row) for row in matrix) for matrix in matrices)
    values = (
        matrices[player][a0][a1]
        for player in (0, 1)
        for _p0 in range(n)
        for _p1 in range(n)
        for a0 in range(n)
        for a1 in range(n)
    )
    return RewardTensor(n, values)
