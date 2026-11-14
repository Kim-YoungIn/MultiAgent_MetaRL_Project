"""Bounded compositional Memory-One rules and symmetric deterministic compiler.

Sampling is over logical predicates and expression constants, never tensor cells.
The operands of each expression are interpreted relative to the scoring player.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .common import finite_reward, validate_action, validate_n, validate_seed
from .tensor import RewardTensor

_VARIABLES = frozenset({'own', 'other', 'prev_own', 'prev_other'})
_MAX_DEPTH = 6
_MAX_NODES = 63


def _operand(value):
    if isinstance(value, str) and value in _VARIABLES:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError('operand must be an action variable or a nonnegative action constant')


@dataclass(frozen=True, slots=True)
class Predicate:
    op: str
    left: str | int | None = None
    right: str | int | None = None
    children: tuple['Predicate', ...] = ()

    def __post_init__(self):
        if not isinstance(self.op, str):
            raise TypeError('predicate op must be a string')
        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise TypeError('predicate children must be iterable') from exc
        if any(not isinstance(child, Predicate) for child in children):
            raise TypeError('predicate children must be Predicate objects')
        object.__setattr__(self, 'children', children)
        if self.op in ('eq', 'ne'):
            _operand(self.left)
            _operand(self.right)
            if children:
                raise ValueError('comparison does not take children')
        elif self.op in ('and', 'or', 'not'):
            if self.left is not None or self.right is not None:
                raise ValueError('logical operator does not take operands')
            if len(children) != (1 if self.op == 'not' else 2):
                raise ValueError('wrong number of predicate children')
        else:
            raise ValueError('unknown predicate operator')


@dataclass(frozen=True, slots=True)
class Payoff:
    op: str
    value: float = 0.0
    condition: Predicate | None = None
    children: tuple['Payoff', ...] = ()

    def __post_init__(self):
        if not isinstance(self.op, str):
            raise TypeError('payoff op must be a string')
        value = finite_reward(self.value)
        object.__setattr__(self, 'value', value)
        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise TypeError('payoff children must be iterable') from exc
        if any(not isinstance(child, Payoff) for child in children):
            raise TypeError('payoff children must be Payoff objects')
        object.__setattr__(self, 'children', children)
        if self.op == 'const':
            if self.condition is not None or children:
                raise ValueError('constant does not take condition or children')
        elif self.op == 'if':
            if not isinstance(self.condition, Predicate) or len(children) != 2 or value != 0:
                raise ValueError('if requires a predicate, two children and zero value')
        elif self.op == 'add':
            if self.condition is not None or len(children) != 2 or value != 0:
                raise ValueError('add requires two children and no value or condition')
        elif self.op == 'scale':
            if self.condition is not None or len(children) != 1:
                raise ValueError('scale requires one child and no condition')
        else:
            raise ValueError('unknown payoff operator')


def _validate_tree(root, n):
    # Iterative walk bounds trees constructed directly, too. Shared subtrees count per use.
    todo = [(root, 1)]
    count = 0
    while todo:
        node, depth = todo.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError('logical rule exceeds node or depth limit')
        if isinstance(node, Predicate):
            for operand in (node.left, node.right):
                if isinstance(operand, int):
                    validate_action(operand, n)
            todo.extend((child, depth + 1) for child in node.children)
        elif isinstance(node, Payoff):
            if node.condition is not None:
                todo.append((node.condition, depth + 1))
            todo.extend((child, depth + 1) for child in node.children)
        else:
            raise TypeError('invalid logical rule node')


@dataclass(frozen=True, slots=True)
class LogicalRule:
    n: int
    root: Payoff

    def __post_init__(self):
        validate_n(self.n)
        if not isinstance(self.root, Payoff):
            raise TypeError('root must be a Payoff')
        _validate_tree(self.root, self.n)


def _predicate(node, context):
    if node.op in ('eq', 'ne'):
        left = context[node.left] if isinstance(node.left, str) else node.left
        right = context[node.right] if isinstance(node.right, str) else node.right
        return left == right if node.op == 'eq' else left != right
    if node.op == 'and':
        return _predicate(node.children[0], context) and _predicate(node.children[1], context)
    if node.op == 'or':
        return _predicate(node.children[0], context) or _predicate(node.children[1], context)
    return not _predicate(node.children[0], context)


def _payoff(node, context):
    if node.op == 'const':
        return node.value
    if node.op == 'if':
        return _payoff(node.children[0 if _predicate(node.condition, context) else 1], context)
    if node.op == 'add':
        return finite_reward(_payoff(node.children[0], context) + _payoff(node.children[1], context))
    return finite_reward(node.value * _payoff(node.children[0], context))


def _pair(values, n, name):
    try:
        pair = tuple(values)
    except TypeError as exc:
        raise TypeError(f'{name} must be a pair') from exc
    if len(pair) != 2:
        raise ValueError(f'{name} must be a pair')
    return tuple(validate_action(value, n) for value in pair)


def evaluate_logical_rule(rule: LogicalRule, previous, actions):
    if not isinstance(rule, LogicalRule):
        raise TypeError('rule must be a LogicalRule')
    previous = _pair(previous, rule.n, 'previous')
    actions = _pair(actions, rule.n, 'actions')
    rewards = []
    for player in (0, 1):
        opponent = 1 - player
        context = {'own': actions[player], 'other': actions[opponent],
                   'prev_own': previous[player], 'prev_other': previous[opponent]}
        rewards.append(finite_reward(_payoff(rule.root, context)))
    return tuple(rewards)


def compile_logical_rule(rule: LogicalRule):
    if not isinstance(rule, LogicalRule):
        raise TypeError('rule must be a LogicalRule')
    rows = [[], []]
    for p0, p1, a0, a1 in product(range(rule.n), repeat=4):
        values = evaluate_logical_rule(rule, (p0, p1), (a0, a1))
        rows[0].append(values[0])
        rows[1].append(values[1])
    return RewardTensor(rule.n, (*rows[0], *rows[1]))


def _predicate_dict(node):
    return {'op': node.op, 'left': node.left, 'right': node.right,
            'children': [_predicate_dict(child) for child in node.children]}


def _payoff_dict(node):
    return {'op': node.op, 'value': node.value,
            'condition': None if node.condition is None else _predicate_dict(node.condition),
            'children': [_payoff_dict(child) for child in node.children]}


def rule_to_dict(rule):
    if not isinstance(rule, LogicalRule):
        raise TypeError('rule must be a LogicalRule')
    return {'version': 1, 'n': rule.n, 'root': _payoff_dict(rule.root)}


def rule_from_dict(data):
    if not isinstance(data, dict) or set(data) != {'version', 'n', 'root'}:
        raise ValueError('invalid logical rule structure')
    if type(data['version']) is not int or data['version'] != 1:
        raise ValueError('unsupported logical rule version')
    n = validate_n(data['n'])
    count = [0]
    def take(node, fields, depth):
        count[0] += 1
        if depth > _MAX_DEPTH or count[0] > _MAX_NODES:
            raise ValueError('logical rule exceeds node or depth limit')
        if not isinstance(node, dict) or set(node) != fields:
            raise ValueError('invalid logical node structure')
        if not isinstance(node['children'], list):
            raise TypeError('serialized children must be a list')
    def parse_predicate(node, depth):
        take(node, {'op', 'left', 'right', 'children'}, depth)
        return Predicate(node['op'], node['left'], node['right'],
                         tuple(parse_predicate(c, depth+1) for c in node['children']))
    def parse_payoff(node, depth):
        take(node, {'op', 'value', 'condition', 'children'}, depth)
        condition = None if node['condition'] is None else parse_predicate(node['condition'], depth+1)
        children = tuple(parse_payoff(c, depth+1) for c in node['children'])
        return Payoff(node['op'], node['value'], condition, children)
    return LogicalRule(n, parse_payoff(data['root'], 1))


def generate_logical_rule(n=3, seed=0, structure='atom'):
    """Sample a bounded expression from explicit relation/Boolean primitives.

    `atom`, `and`, `or` and `nested` select condition syntax. Each atomic
    relation uniformly samples two different variables, or a variable and an
    action constant, and eq/ne. Numeric branches and a coordination component
    come from fixed small sets; no tensor entries are independently sampled.
    Candidate selection may reject degenerate/equivalent outputs externally.
    Generated depth <=4, leaving two levels for a global affine transform.
    """
    n = validate_n(n)
    seed = validate_seed(seed)
    if seed is None:
        raise TypeError('explicit seed required')
    if structure not in ('atom', 'and', 'or', 'nested'):
        raise ValueError('unknown logical structure')
    rng = random.Random(seed)
    variables = ('own', 'other', 'prev_own', 'prev_other')
    def atom():
        left = rng.choice(variables)
        if rng.randrange(3) == 0:
            right = rng.randrange(n)
        else:
            right = rng.choice(tuple(v for v in variables if v != left))
        return Predicate(rng.choice(('eq', 'ne')), left, right)
    if structure == 'atom':
        condition = atom()
    elif structure in ('and', 'or'):
        condition = Predicate(structure, children=(atom(), atom()))
    else:
        condition = Predicate('and', children=(atom(), Predicate('or', children=(atom(), atom()))))
    coordination = Predicate('eq', 'own', 'other')
    def const(value):
        return Payoff('const', value=value)
    if structure == 'nested':
        branches = [const(rng.choice((-2, -1, 0, 1, 2, 3))) for _ in range(4)]
        root = Payoff('if', condition=condition, children=(
            Payoff('if', condition=coordination, children=tuple(branches[:2])),
            Payoff('if', condition=coordination, children=tuple(branches[2:])),
        ))
    else:
        base = Payoff('if', condition=coordination, children=(const(1), const(0)))
        component = Payoff('if', condition=condition,
                           children=(const(rng.choice((-2, -1, 0, 1, 2, 3))),
                                     const(rng.choice((-2, -1, 0, 1, 2, 3)))))
        root = Payoff('add', children=(
            Payoff('scale', value=rng.choice((-2, -1, .5, 1, 2)), children=(base,)), component,
        ))
    return LogicalRule(n, root)
