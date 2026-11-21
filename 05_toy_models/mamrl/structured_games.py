"""Procedural finite-state games with bounded action history.

Drop-in environment API for mamrl.rollout.run_episode. This module is opt-in:
existing RuleSpec, RewardTensor, MemoryOneEnv and D-031 studies are unchanged.

A game is a legacy Memory-One payoff tensor plus *logical*, player-symmetric
reward and state-transition clauses. Clauses are evaluated against the OLD
state and bounded action history. The joint actions are simultaneous.

No dense (n ** (2 * (L + 1))) tensor is allocated for history length L.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
from typing import Any

from .adaptive_agents import OnlineQ
from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import MemoryOneEnv, Observation, StepResult
from .rules import compile_rule, generate_rule
from .tensor import RewardTensor

PREDICATES = frozenset({
    'always', 'own_is', 'other_is', 'both_is', 'match', 'repeat',
    'copy', 'own_count', 'other_count', 'matching_count',
    'state_min', 'state_max',
})
COUNT_PREDICATES = frozenset({'own_count', 'other_count', 'matching_count'})
ACTION_PREDICATES = frozenset({'own_is', 'other_is', 'both_is', 'own_count', 'other_count'})
STATE_PREDICATES = frozenset({'state_min', 'state_max'})


@dataclass(frozen=True, slots=True)
class StateVariable:
    """One shared persistent integer state, range [0, levels-1]."""
    name: str
    levels: int
    initial: int = 0
    public: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError('state name must be a valid identifier')
        positive_integer(self.levels, 'levels')
        if self.levels < 2:
            raise ValueError('a state variable requires at least two levels')
        if type(self.initial) is not int or not 0 <= self.initial < self.levels:
            raise ValueError('initial state must lie within its declared levels')
        if type(self.public) is not bool:
            raise TypeError('public must be boolean')


@dataclass(frozen=True, slots=True)
class Predicate:
    """A local-player predicate, optionally negated.

    lag=1 refers to the immediately preceding joint action; current actions
    are available to own_is/other_is/both_is/match/repeat/copy predicates.
    """
    kind: str
    action: int | None = None
    lag: int = 1
    window: int = 1
    count: int = 1
    state: str | None = None
    threshold: int = 0
    negate: bool = False

    def __post_init__(self):
        if self.kind not in PREDICATES:
            raise ValueError(f'unknown predicate: {self.kind!r}')
        positive_integer(self.lag, 'lag')
        positive_integer(self.window, 'window')
        positive_integer(self.count, 'count')
        if type(self.threshold) is not int:
            raise TypeError('threshold must be an integer')
        if type(self.negate) is not bool:
            raise TypeError('negate must be boolean')


@dataclass(frozen=True, slots=True)
class RewardClause:
    """Add a value to each player's reward if all predicates match locally."""
    value: float
    when: tuple[Predicate, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'value', finite_reward(self.value))
        object.__setattr__(self, 'when', _predicates(self.when))


@dataclass(frozen=True, slots=True)
class TransitionClause:
    """Change one shared state if any/both local perspectives match.

    Multiple matching changes are added simultaneously then clamped. Each
    condition has an independent Bernoulli gate, with local environment RNG.
    """
    state: str
    delta: int
    when: tuple[Predicate, ...] = ()
    scope: str = 'any'
    probability: float = 1.0

    def __post_init__(self):
        if type(self.delta) is not int or self.delta == 0:
            raise ValueError('delta must be a nonzero integer')
        if self.scope not in ('any', 'both'):
            raise ValueError('scope must be any or both')
        probability = finite_reward(self.probability)
        if not 0 <= probability <= 1:
            raise ValueError('probability must lie within [0,1]')
        object.__setattr__(self, 'probability', probability)
        object.__setattr__(self, 'when', _predicates(self.when))


def _predicates(items):
    copied = tuple(items)
    if any(not isinstance(p, Predicate) for p in copied):
        raise TypeError('conditions must be Predicate instances')
    return copied


@dataclass(frozen=True, slots=True)
class StructuredGame:
    """Immutable compiled game specification; safe to share between trials.

    `base` uses the original global (2,n,n,n,n) payoff axes. Additional
    clauses use player-local predicates and a shared state, preserving
    player-exchange symmetry when the base tensor is also symmetric.
    """
    n: int
    memory_depth: int
    base: RewardTensor
    states: tuple[StateVariable, ...] = ()
    rewards: tuple[RewardClause, ...] = ()
    transitions: tuple[TransitionClause, ...] = ()
    family: str = 'custom'
    roles: tuple[int, ...] = ()
    reward_scale: float = 1.0

    def __post_init__(self):
        n = validate_n(self.n)
        positive_integer(self.memory_depth, 'memory_depth')
        if not isinstance(self.base, RewardTensor) or self.base.n != n:
            raise ValueError('base payoff must be a RewardTensor with matching n')
        for name, cls in [('states', StateVariable), ('rewards', RewardClause),
                          ('transitions', TransitionClause)]:
            copied = tuple(getattr(self, name))
            if any(not isinstance(item, cls) for item in copied):
                raise TypeError(f'{name} must contain {cls.__name__}')
            object.__setattr__(self, name, copied)
        names = [s.name for s in self.states]
        if len(set(names)) != len(names):
            raise ValueError('state names must be unique')
        roles = tuple(validate_action(a, n) for a in self.roles)
        if len(set(roles)) != len(roles):
            raise ValueError('action roles must be distinct')
        object.__setattr__(self, 'roles', roles)
        if not isinstance(self.family, str):
            raise TypeError('family must be a string')
        scale = finite_reward(self.reward_scale)
        if scale <= 0:
            raise ValueError('reward_scale must be positive')
        object.__setattr__(self, 'reward_scale', scale)
        by_name = {s.name: s for s in self.states}
        for clause in (*self.rewards, *self.transitions):
            if isinstance(clause, TransitionClause) and clause.state not in by_name:
                raise ValueError('transition refers to unknown state')
            for p in clause.when:
                _validate_predicate(p, n, self.memory_depth, by_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            'version': 1, 'n': self.n, 'memory_depth': self.memory_depth,
            'base_values': list(self.base.values),
            'states': [asdict(s) for s in self.states],
            'rewards': [asdict(r) for r in self.rewards],
            'transitions': [asdict(t) for t in self.transitions],
            'family': self.family, 'roles': list(self.roles),
            'reward_scale': self.reward_scale,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> StructuredGame:
        if record.get('version') != 1:
            raise ValueError('unsupported structured game version')
        def conditions(items):
            return tuple(Predicate(**p) for p in items)
        return cls(
            n=record['n'], memory_depth=record['memory_depth'],
            base=RewardTensor(record['n'], record['base_values']),
            states=tuple(StateVariable(**v) for v in record['states']),
            rewards=tuple(RewardClause(value=r['value'], when=conditions(r['when']))
                          for r in record['rewards']),
            transitions=tuple(TransitionClause(state=t['state'], delta=t['delta'],
                               when=conditions(t['when']), scope=t['scope'],
                               probability=t['probability']) for t in record['transitions']),
            family=record.get('family', 'custom'), roles=tuple(record.get('roles', ())),
            reward_scale=record.get('reward_scale', 1.0),
        )

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    def legacy_tensor(self) -> RewardTensor:
        """Exact round-trip, only for untouched Memory-One payoff games."""
        if (self.memory_depth != 1 or self.states or self.rewards or
                self.transitions or self.reward_scale != 1.0):
            raise ValueError('extended rules cannot be represented by a legacy RewardTensor')
        return self.base


def _validate_predicate(p: Predicate, n: int, depth: int, states: dict[str, StateVariable]):
    if p.kind in ACTION_PREDICATES:
        validate_action(p.action, n)
    elif p.action is not None:
        raise ValueError(f'{p.kind} does not accept an action')
    if p.kind in ('repeat', 'copy') and p.lag > depth:
        raise ValueError('lag exceeds game memory depth')
    if p.kind in COUNT_PREDICATES:
        if p.window > depth or p.count > p.window:
            raise ValueError('count/window exceed game memory depth')
    if p.kind in STATE_PREDICATES:
        if p.state not in states:
            raise ValueError('predicate refers to unknown state')
        if not 0 <= p.threshold < states[p.state].levels:
            raise ValueError('state threshold is out of range')
    elif p.state is not None:
        raise ValueError(f'{p.kind} does not accept a state')


def _check(p: Predicate, player: int, state: dict[str, int],
           history: tuple[tuple[int, int], ...], actions: tuple[int, int]) -> bool:
    own, other = actions[player], actions[1-player]
    if p.kind == 'always':
        yes = True
    elif p.kind == 'own_is':
        yes = own == p.action
    elif p.kind == 'other_is':
        yes = other == p.action
    elif p.kind == 'both_is':
        yes = own == other == p.action
    elif p.kind == 'match':
        yes = own == other
    elif p.kind == 'repeat':
        yes = own == history[p.lag - 1][player]
    elif p.kind == 'copy':
        yes = own == history[p.lag - 1][1-player]
    elif p.kind == 'own_count':
        yes = sum(h[player] == p.action for h in history[:p.window]) >= p.count
    elif p.kind == 'other_count':
        yes = sum(h[1-player] == p.action for h in history[:p.window]) >= p.count
    elif p.kind == 'matching_count':
        yes = sum(h[0] == h[1] for h in history[:p.window]) >= p.count
    elif p.kind == 'state_min':
        yes = state[p.state] >= p.threshold
    elif p.kind == 'state_max':
        yes = state[p.state] <= p.threshold
    else:
        raise RuntimeError('invalid compiled predicate')
    return not yes if p.negate else yes


def _matches(predicates: tuple[Predicate, ...], player: int, state,
             history, actions) -> bool:
    return all(_check(p, player, state, history, actions) for p in predicates)


@dataclass(frozen=True, slots=True)
class StructuredObservation(Observation):
    """Legacy four fields remain intact; new fields may be ignored by old agents.

    action_history: player-local (own, other) pairs, most recent first.
    public_state: integers for visible variables; None for hidden variables.
    """
    public_state: tuple[int | None, ...] = ()
    action_history: tuple[tuple[int, int], ...] = ()


class StructuredEnv:
    """Same reset/step interface as MemoryOneEnv, with persistent state."""
    def __init__(self, rule: StructuredGame, *, max_steps: int):
        if not isinstance(rule, StructuredGame):
            raise TypeError('rule must be StructuredGame')
        self._rule = rule
        self._max_steps = positive_integer(max_steps, 'max_steps')
        self._rng = random.Random()
        self._history: tuple[tuple[int, int], ...] = ()
        self._state: dict[str, int] = {}
        self._elapsed = 0
        self._active = False

    @property
    def n(self) -> int:
        return self._rule.n

    @property
    def public_snapshot(self) -> tuple[int | None, ...]:
        return tuple(self._state[v.name] if v.public else None for v in self._rule.states)

    def _observations(self, rewards: tuple[float, float], start: bool):
        prev = self._history[0]
        public = self.public_snapshot
        return tuple(StructuredObservation(
            previous_own_action=prev[player],
            previous_opponent_action=prev[1-player],
            previous_own_reward=rewards[player], episode_start_flag=start,
            public_state=public,
            action_history=tuple((x[player], x[1-player]) for x in self._history),
        ) for player in (0, 1))

    def reset(self, *, seed: int | None = None, rule: StructuredGame | None = None):
        validate_seed(seed)
        if rule is not None:
            if not isinstance(rule, StructuredGame):
                raise TypeError('replacement rule must be StructuredGame')
            if rule.n != self.n:
                raise ValueError('replacement rule must have the same action count')
        if seed is not None:
            self._rng.seed(seed)
        if rule is not None:
            self._rule = rule
        # The first pair matches MemoryOneEnv's seeded initial pair exactly.
        self._history = tuple((self._rng.randrange(self.n), self._rng.randrange(self.n))
                              for _ in range(self._rule.memory_depth))
        self._state = {v.name: v.initial for v in self._rule.states}
        self._elapsed = 0
        self._active = True
        return self._observations((0.0, 0.0), True)

    def step(self, action_0: int, action_1: int) -> StepResult:
        if not self._active:
            raise RuntimeError('reset is required before stepping')
        actions = (validate_action(action_0, self.n), validate_action(action_1, self.n))
        previous = self._history[0]
        old_state = dict(self._state)
        base = self._rule.base
        rewards = [base[player, *previous, *actions] for player in (0, 1)]
        for clause in self._rule.rewards:
            for player in (0, 1):
                if _matches(clause.when, player, old_state, self._history, actions):
                    rewards[player] += clause.value
        deltas = {v.name: 0 for v in self._rule.states}
        for clause in self._rule.transitions:
            matches = [_matches(clause.when, p, old_state, self._history, actions)
                       for p in (0, 1)]
            fired = any(matches) if clause.scope == 'any' else all(matches)
            if fired and (clause.probability == 1 or self._rng.random() < clause.probability):
                deltas[clause.state] += clause.delta
        self._state = {v.name: min(v.levels - 1, max(0, old_state[v.name] + deltas[v.name]))
                       for v in self._rule.states}
        self._history = (actions, *self._history[:self._rule.memory_depth-1])
        self._elapsed += 1
        truncated = self._elapsed == self._max_steps
        self._active = not truncated
        final_rewards = tuple(finite_reward(r * self._rule.reward_scale) for r in rewards)
        return StepResult(self._observations(final_rewards, False),
                          final_rewards, False, truncated)


class StructuredQ(OnlineQ):
    """Optional Q baseline that keys on observed state and bounded history.

    Original OnlineQ still runs, but deliberately ignores these extra fields.
    Hidden state variables are not leaked into this key.
    """
    def _key(self, observation):
        if not isinstance(observation, StructuredObservation):
            raise TypeError('StructuredQ expects StructuredObservation')
        return (*super()._key(observation), observation.public_state,
                observation.action_history)

    def snapshot(self):
        data = super().snapshot()
        for row, key in zip(data['entries'], sorted(self._q)):
            row['observation']['public_state'] = list(key[4])
            row['observation']['action_history'] = [list(x) for x in key[5]]
        return data


def from_legacy(rule, *, family='legacy') -> StructuredGame:
    """Wrap any RuleSpec/LogicalRule/RelationalRule/RewardTensor without changes."""
    tensor = rule if isinstance(rule, RewardTensor) else compile_rule(rule)
    return StructuredGame(tensor.n, 1, tensor, family=family)


def _small_base(n: int, rng: random.Random) -> RewardTensor:
    source = compile_rule(generate_rule(n=n, seed=rng.getrandbits(63), family='mixed'))
    maximum = max((abs(value) for value in source.values), default=0)
    factor = .25 / maximum if maximum else 0.0
    return RewardTensor(n, (round(v * factor, 8) for v in source.values))


def generate_structured_rule(*, n=3, memory_depth=3, seed=0, family='mixed',
                             state_levels=(3, 3, 3), stochastic=False) -> StructuredGame:
    """Generate a deterministic logical family; randomness is local to seed.

    Each family creates an interpretable reward/transition mechanism, including
    deliberately costly and immediately rewarding choices. Role names are
    generator metadata; agents only see action numbers and observations.
    """
    n = validate_n(n)
    positive_integer(memory_depth, 'memory_depth')
    validate_seed(seed)
    if n < 3:
        raise ValueError('structured generator requires at least three distinct actions')
    if family not in ('resource', 'sequence', 'reciprocity', 'mixed'):
        raise ValueError('unknown family')
    levels = tuple(state_levels)
    if not 1 <= len(levels) <= 3:
        raise ValueError('state_levels must contain between one and three sizes')
    for value in levels:
        positive_integer(value, 'state_levels entry')
        if value < 2:
            raise ValueError('state levels must be at least 2')
    if type(stochastic) is not bool:
        raise TypeError('stochastic must be boolean')
    rng = random.Random(seed)
    invest, harvest, signal = rng.sample(range(n), 3)
    base = _small_base(n, rng)
    r, t = [], []
    variables = [StateVariable(name, count, 0, public=(i != 2))
                 for i, (name, count) in enumerate(zip(('resource','progress','trust'), levels))]
    if family == 'resource':
        variables = variables[:1]
    elif family == 'sequence':
        variables = variables[:2] if len(variables) > 1 else variables[:1]
    elif family == 'reciprocity':
        variables = variables[:1]
    as_pred = lambda name, threshold: Predicate('state_min', state=name, threshold=threshold)
    want = lambda a: Predicate('own_is', action=a)
    if family in ('resource', 'mixed'):
        # Investment is costly; it creates a shared resource only if both invest.
        t.append(TransitionClause('resource', +1, (want(invest),), 'both',
                                  probability=.8 if stochastic else 1.0))
        t.append(TransitionClause('resource', -1, (want(harvest),), 'any'))
        r.append(RewardClause(-round(rng.uniform(.12,.35), 4), (want(invest),)))
        r.append(RewardClause(round(rng.uniform(1.0,1.7),4),
                              (want(harvest), as_pred('resource', 1))))
        r.append(RewardClause(-round(rng.uniform(.1,.3),4),
                              (want(harvest), Predicate('state_max', state='resource', threshold=0))))
    if family in ('sequence','mixed'):
        name = 'progress' if any(s.name == 'progress' for s in variables) else 'resource'
        lag = min(memory_depth, 2)
        t.append(TransitionClause(name, +1,
                                  (want(invest), Predicate('repeat', lag=lag)),
                                  'both'))
        t.append(TransitionClause(name, -1, (want(harvest),), 'any'))
        r.append(RewardClause(round(rng.uniform(.8,1.6),4),
                              (want(harvest), as_pred(name, variables[[s.name for s in variables].index(name)].levels-1))))
        r.append(RewardClause(-round(rng.uniform(.1,.25),4), (want(invest),)))
    if family in ('reciprocity','mixed'):
        name = 'trust' if any(s.name == 'trust' for s in variables) else 'resource'
        t.append(TransitionClause(name, +1, (want(invest),), 'both'))
        t.append(TransitionClause(name, -1, (want(harvest),), 'any'))
        r.append(RewardClause(round(rng.uniform(.5,1.2),4),
                              (want(invest), as_pred(name, 1))))
        if memory_depth > 1:
            r.append(RewardClause(round(rng.uniform(.3,.8),4),
                                  (want(signal), Predicate('own_count', action=invest,
                                    window=memory_depth, count=min(2,memory_depth)))))
    # A nonuniform, seeded probing cost; it does not guarantee information gain.
    r.append(RewardClause(-round(rng.uniform(.05,.2),4), (want(signal),)))
    # A conservative global reward bound, independent of sampled trajectories.
    # This preserves every within-game preference (positive uniform scaling).
    bound = max(abs(v) for v in base.values) + sum(abs(c.value) for c in r)
    scale = 1.0 / max(1.0, bound)
    return StructuredGame(n, memory_depth, base, tuple(variables), tuple(r), tuple(t),
                          family=family, roles=(invest,harvest,signal), reward_scale=scale)


def generate_catalog(*, count: int, seed: int, n=3, memory_depth=3,
                     families=('resource','sequence','reciprocity','mixed'),
                     state_levels=(3,3,3), stochastic=False):
    """Deterministic catalog without exact-serialization duplicates.

    This is NOT canonical under action relabeling; a separate semantic or
    best-response equivalence audit is needed for H1/H2 experiments.
    """
    positive_integer(count, 'count')
    validate_seed(seed)
    if not families:
        raise ValueError('families cannot be empty')
    rng = random.Random(seed)
    seen = set()
    games = []
    for i in range(count):
        for _ in range(10000):
            game = generate_structured_rule(n=n, memory_depth=memory_depth,
                seed=rng.getrandbits(63), family=families[i % len(families)],
                state_levels=state_levels, stochastic=stochastic)
            digest = game.fingerprint()
            if digest not in seen:
                seen.add(digest)
                games.append(game)
                break
        else:
            raise RuntimeError('not enough distinct games for requested catalog')
    return tuple(games)


def _main():
    parser = argparse.ArgumentParser(description='Structured procedural game demo')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--family', choices=('resource','sequence','reciprocity','mixed'), default='mixed')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--output', help='Optional reproducible game specification JSON')
    args = parser.parse_args()
    game = generate_structured_rule(n=args.n, seed=args.seed, memory_depth=args.depth,
                                    family=args.family, stochastic=args.stochastic)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(game.to_dict(), indent=2), encoding='utf-8')
    env = StructuredEnv(game, max_steps=args.steps)
    obs = env.reset(seed=args.seed)
    rng = random.Random(args.seed + 1)
    trace, totals = [], [0.0, 0.0]
    for tick in range(args.steps):
        actions = (rng.randrange(args.n), rng.randrange(args.n))
        result = env.step(*actions)
        totals = [totals[i]+result.rewards[i] for i in (0,1)]
        if tick < 12:
            trace.append({'turn': tick+1, 'actions': actions,
                          'rewards': result.rewards,
                          'public_state': result.observations[0].public_state,
                          'action_history': result.observations[0].action_history})
        obs = result.observations
    print(json.dumps({'fingerprint': game.fingerprint(), 'n': game.n,
                      'memory_depth': game.memory_depth, 'family': game.family,
                      'state_names': [v.name for v in game.states],
                      'action_roles': game.roles, 'total_rewards': totals,
                      'trace': trace}, indent=2))


if __name__ == '__main__':
    _main()
