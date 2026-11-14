"""Compositional symmetric Memory-One games with explicit action relations.

Relations, payoff expressions, and the base outcome table are separate. Rule
generation samples logical expressions and parameters, never reward-tensor cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from .common import finite_reward, validate_action, validate_n, validate_seed
from .tensor import RewardTensor

_VARIABLES = frozenset({'own', 'other', 'prev_own', 'prev_other'})
_MAX_DEPTH = 10
_MAX_NODES = 255


def _operand(value):
    if isinstance(value, str) and value in _VARIABLES:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError('operand must be a local action variable or nonnegative action constant')


@dataclass(frozen=True, slots=True)
class Relation:
    op: str
    left: str | int | None = None
    right: str | int | None = None
    children: tuple['Relation', ...] = ()

    def __post_init__(self):
        if not isinstance(self.op, str):
            raise TypeError('relation op must be a string')
        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise TypeError('relation children must be iterable') from exc
        if any(not isinstance(child, Relation) for child in children):
            raise TypeError('relation children must be Relation nodes')
        object.__setattr__(self, 'children', children)
        if self.op in ('eq', 'ne', 'win', 'draw', 'lose'):
            _operand(self.left)
            _operand(self.right)
            if children:
                raise ValueError('binary relation must not contain children')
        elif self.op in ('and', 'or', 'not'):
            if self.left is not None or self.right is not None:
                raise ValueError('logical operator must not have operands')
            if len(children) != (1 if self.op == 'not' else 2):
                raise ValueError('wrong number of relation children')
        else:
            raise ValueError('unknown relation op')


@dataclass(frozen=True, slots=True)
class Reward:
    op: str
    value: float = 0.0
    left: str | int | None = None
    right: str | int | None = None
    condition: Relation | None = None
    children: tuple['Reward', ...] = ()

    def __post_init__(self):
        if not isinstance(self.op, str):
            raise TypeError('reward op must be a string')
        value = finite_reward(self.value)
        object.__setattr__(self, 'value', value)
        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise TypeError('reward children must be iterable') from exc
        if any(not isinstance(child, Reward) for child in children):
            raise TypeError('reward children must be Reward nodes')
        object.__setattr__(self, 'children', children)
        if self.op == 'const':
            if self.left is not None or self.right is not None or self.condition is not None or children:
                raise ValueError('const only accepts value')
        elif self.op == 'outcome':
            _operand(self.left)
            _operand(self.right)
            if value != 0 or self.condition is not None or children:
                raise ValueError('outcome only accepts two operands')
        elif self.op == 'if':
            if (value != 0 or self.left is not None or self.right is not None or
                    not isinstance(self.condition, Relation) or len(children) != 2):
                raise ValueError('if requires a relation and two reward children')
        elif self.op in ('add', 'scale'):
            expected = 2 if self.op == 'add' else 1
            if (self.left is not None or self.right is not None or self.condition is not None or
                    len(children) != expected or (self.op == 'add' and value != 0)):
                raise ValueError('invalid arithmetic reward')
        else:
            raise ValueError('unknown reward op')


def _outcomes(value, n):
    try:
        table = tuple(tuple(row) for row in value)
    except TypeError as exc:
        raise TypeError('outcomes must be a square iterable') from exc
    if len(table) != n or any(len(row) != n for row in table):
        raise ValueError('outcomes must have shape (n,n)')
    for i in range(n):
        for j in range(n):
            x = table[i][j]
            if isinstance(x, bool) or not isinstance(x, int) or x not in (-1, 0, 1):
                raise ValueError('outcomes must contain integer -1, 0 or 1')
            if (i == j and x != 0) or x != -table[j][i]:
                raise ValueError('outcomes must have zero diagonal and be antisymmetric')
    return table


def _validate_tree(root, n):
    pending = [(root, 1)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if depth > _MAX_DEPTH or count > _MAX_NODES:
            raise ValueError('relational rule exceeds depth or node bound')
        if not isinstance(node, (Reward, Relation)):
            raise TypeError('invalid rule node')
        for value in (node.left, node.right):
            if isinstance(value, int):
                validate_action(value, n)
        pending.extend((child, depth + 1) for child in node.children)
        if isinstance(node, Reward) and node.condition is not None:
            pending.append((node.condition, depth + 1))


@dataclass(frozen=True, slots=True)
class RelationalRule:
    n: int
    outcomes: tuple[tuple[int, ...], ...]
    root: Reward
    scale: float = 1.0
    shift: float = 0.0

    def __post_init__(self):
        n = validate_n(self.n)
        object.__setattr__(self, 'outcomes', _outcomes(self.outcomes, n))
        if not isinstance(self.root, Reward):
            raise TypeError('root must be a Reward')
        _validate_tree(self.root, n)
        scale, shift = finite_reward(self.scale), finite_reward(self.shift)
        if scale <= 0:
            raise ValueError('scale must be positive')
        object.__setattr__(self, 'scale', scale)
        object.__setattr__(self, 'shift', shift)


def cyclic_outcomes(n):
    n = validate_n(n)
    return tuple(tuple(0 if i == j or 2*((i-j) % n) == n else
                       (1 if 2*((i-j) % n) < n else -1)
                       for j in range(n)) for i in range(n))


def _action(value, context):
    return context[value] if isinstance(value, str) else value


def _relation(node, context, outcomes):
    if node.op in ('eq', 'ne', 'win', 'draw', 'lose'):
        a, b = _action(node.left, context), _action(node.right, context)
        if node.op == 'eq':
            return a == b
        if node.op == 'ne':
            return a != b
        return outcomes[a][b] == {'win': 1, 'draw': 0, 'lose': -1}[node.op]
    if node.op == 'not':
        return not _relation(node.children[0], context, outcomes)
    if node.op == 'and':
        return _relation(node.children[0], context, outcomes) and _relation(node.children[1], context, outcomes)
    return _relation(node.children[0], context, outcomes) or _relation(node.children[1], context, outcomes)


def _reward(node, context, outcomes):
    if node.op == 'const':
        return node.value
    if node.op == 'outcome':
        return float(outcomes[_action(node.left, context)][_action(node.right, context)])
    if node.op == 'if':
        branch = 0 if _relation(node.condition, context, outcomes) else 1
        return _reward(node.children[branch], context, outcomes)
    if node.op == 'add':
        return finite_reward(sum(_reward(c, context, outcomes) for c in node.children))
    return finite_reward(node.value * _reward(node.children[0], context, outcomes))


def _pair(values, n, name):
    try:
        pair = tuple(values)
    except TypeError as exc:
        raise TypeError(f'{name} must be a pair') from exc
    if len(pair) != 2:
        raise ValueError(f'{name} must be a pair')
    return tuple(validate_action(v, n) for v in pair)


def evaluate_relational_rule(rule, previous, actions):
    if not isinstance(rule, RelationalRule):
        raise TypeError('rule must be a RelationalRule')
    previous, actions = _pair(previous, rule.n, 'previous'), _pair(actions, rule.n, 'actions')
    values = []
    for player in (0, 1):
        other = 1-player
        context = {'own': actions[player], 'other': actions[other],
                   'prev_own': previous[player], 'prev_other': previous[other]}
        value = _reward(rule.root, context, rule.outcomes)
        values.append(finite_reward(rule.scale*value + rule.shift))
    return tuple(values)


def compile_relational_rule(rule):
    if not isinstance(rule, RelationalRule):
        raise TypeError('rule must be a RelationalRule')
    rows = [[], []]
    for p, q, a, b in product(range(rule.n), repeat=4):
        for player, value in enumerate(evaluate_relational_rule(rule, (p,q), (a,b))):
            rows[player].append(value)
    return RewardTensor(rule.n, (*rows[0], *rows[1]))


def _relation_dict(node):
    return {'op':node.op, 'left':node.left, 'right':node.right,
            'children':[_relation_dict(c) for c in node.children]}


def _reward_dict(node):
    return {'op':node.op, 'value':node.value, 'left':node.left, 'right':node.right,
            'condition':None if node.condition is None else _relation_dict(node.condition),
            'children':[_reward_dict(c) for c in node.children]}


def relational_to_dict(rule):
    if not isinstance(rule, RelationalRule):
        raise TypeError('rule must be a RelationalRule')
    return {'format':'relational_rule', 'version':1, 'n':rule.n,
            'outcomes':[list(row) for row in rule.outcomes], 'root':_reward_dict(rule.root),
            'scale':rule.scale, 'shift':rule.shift}


def relational_from_dict(data):
    required = {'format','version','n','outcomes','root','scale','shift'}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError('invalid relational rule structure')
    if data['format'] != 'relational_rule' or type(data['version']) is not int or data['version'] != 1:
        raise ValueError('unsupported relational rule format/version')
    n = validate_n(data['n'])
    if not isinstance(data['outcomes'], list) or any(not isinstance(row,list) for row in data['outcomes']):
        raise TypeError('serialized outcomes must be lists')
    count = [0]
    def take(node, fields, depth):
        count[0] += 1
        if depth > _MAX_DEPTH or count[0] > _MAX_NODES:
            raise ValueError('relational rule exceeds depth or node bound')
        if not isinstance(node, dict) or set(node) != fields:
            raise ValueError('invalid relational node structure')
        if not isinstance(node['children'], list):
            raise TypeError('serialized children must be lists')
    def parse_relation(node, depth):
        take(node, {'op','left','right','children'}, depth)
        return Relation(node['op'],node['left'],node['right'],tuple(parse_relation(c,depth+1) for c in node['children']))
    def parse_reward(node, depth):
        take(node, {'op','value','left','right','condition','children'}, depth)
        condition = None if node['condition'] is None else parse_relation(node['condition'],depth+1)
        children = tuple(parse_reward(c,depth+1) for c in node['children'])
        return Reward(node['op'],node['value'],node['left'],node['right'],condition,children)
    return RelationalRule(n,data['outcomes'],parse_reward(data['root'],1),data['scale'],data['shift'])


def _const(x): return Reward('const',value=x)
def _if(condition, yes, no): return Reward('if',condition=condition,children=(yes,no))
def _add(left,right): return Reward('add',children=(left,right))
def _scale(value,child): return Reward('scale',value=value,children=(child,))


def social_rule(n=3, T=5, R=3, P=1, S=0):
    """Repeat/change encode C/D; arbitrary supplied values need not be PD."""
    n = validate_n(n)
    T,R,P,S = tuple(finite_reward(v) for v in (T,R,P,S))
    mine = Relation('eq','own','prev_own')
    theirs = Relation('eq','other','prev_other')
    root = _if(mine, _if(theirs,_const(R),_const(S)),
               _if(theirs,_const(T),_const(P)))
    return RelationalRule(n,cyclic_outcomes(n),root)


def _permuted_outcomes(n, rng):
    base = cyclic_outcomes(n)
    perm = rng.sample(range(n),n)
    return tuple(tuple(base[perm[i]][perm[j]] for j in range(n)) for i in range(n))


def generate_relational_rule(n=3, seed=0, family='social'):
    """Deterministically sample a bounded semantic expression and its parameters.

    Social payoff orders are sampled, not restricted to PD. Other families
    select relational conditions, outcome or temporal bases, and conditional
    modifiers. Exception is an ordered if/else override. All random choices
    use a private generator, never independent reward tensor entries.
    """
    n = validate_n(n)
    if validate_seed(seed) is None:
        raise TypeError('explicit seed required')
    if family not in ('social','contest','reciprocity','hidden','exception'):
        raise ValueError('unknown relational family')
    rng = random.Random(seed)
    outcomes = _permuted_outcomes(n,rng)
    outcome = Reward('outcome',left='own',right='other')
    repeat = Relation('eq','own','prev_own')
    other_repeat = Relation('eq','other','prev_other')
    copy = Relation('eq','own','prev_other')
    other_copy = Relation('eq','other','prev_own')
    lose_previous = Relation('lose','own','prev_other')
    other_loses = Relation('lose','other','prev_own')
    conditions = (
        Relation('and',children=(lose_previous,other_loses)),
        Relation('and',children=(copy,other_copy)),
        Relation('and',children=(repeat,other_repeat)),
        Relation('and',children=(Relation('not',children=(repeat,)), Relation('not',children=(other_repeat,)))),
        Relation('eq','prev_own','prev_other'),
        Relation('ne','prev_own','prev_other'),
    )
    scalar = rng.choice((-3,-2,-1,1,2,3))
    if family == 'social':
        values = rng.sample((-3,-2,-1,0,1,2,3,4,5),4)
        root = social_rule(n,*values).root
    elif family == 'contest':
        temporal = rng.choice((repeat,copy,Relation('ne','own','prev_own'),
                               Relation('win','own','prev_other'),Relation('lose','own','prev_other')))
        root = _add(_scale(rng.choice((-2,-1,1,2)),outcome),
                    _if(temporal,_const(scalar),_const(0)))
    elif family == 'reciprocity':
        base = rng.choice((outcome,_if(repeat,_const(1),_const(-1)),
                           _if(copy,_const(1),_const(0)),_const(0)))
        root = _add(_scale(rng.choice((-2,-1,1,2)),base),
                    _if(rng.choice(conditions),_const(scalar),_const(0)))
    elif family == 'hidden':
        base = rng.choice((outcome,_if(repeat,_const(1),_const(-1)),
                           _if(copy,_const(1),_const(0)),_const(0)))
        action = Relation('eq','own',rng.randrange(n))
        root = _add(_scale(rng.choice((-2,-1,1,2)),base),_if(action,_const(scalar),_const(0)))
    else:
        # Explicit priority: when condition is true the selected override
        # replaces the base; otherwise the base applies unchanged.
        base = rng.choice((outcome,_if(repeat,_const(1),_const(-1)),
                           _if(copy,_const(1),_const(0))))
        override = rng.choice((_scale(-1,base),_const(scalar),
                               _add(_scale(-1,base),_const(scalar))))
        root = _if(rng.choice(conditions),override,base)
    return RelationalRule(n,outcomes,root)


def describe_relational_rule(rule):
    if not isinstance(rule, RelationalRule):
        raise TypeError('rule must be a RelationalRule')
    names = {'own':'현재 자기 행동','other':'현재 상대 행동',
             'prev_own':'직전 자기 행동','prev_other':'직전 상대 행동'}
    def operand(x): return names[x] if isinstance(x,str) else f'행동 {x}'
    def rel(node):
        if node.op in ('eq','ne','win','draw','lose'):
            op={'eq':'같음','ne':'다름','win':'승리','draw':'무승부','lose':'패배'}[node.op]
            return f'({operand(node.left)} 대 {operand(node.right)}: {op})'
        if node.op == 'not': return f'아님({rel(node.children[0])})'
        return (' 그리고 ' if node.op=='and' else ' 또는 ').join(rel(c) for c in node.children)
    def reward(node):
        if node.op=='const': return f'{node.value:g}'
        if node.op=='outcome': return f'기본 상성({operand(node.left)}, {operand(node.right)})'
        if node.op=='if': return f'{rel(node.condition)}이면 [{reward(node.children[0])}], 아니면 [{reward(node.children[1])}]'
        if node.op=='add': return ' + '.join(f'({reward(c)})' for c in node.children)
        return f'{node.value:g} × ({reward(node.children[0])})'
    return f'기본 상성표: {rule.outcomes}; 실제 보상 = {rule.scale:g} × ({reward(rule.root)}) + {rule.shift:g}'
