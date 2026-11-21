"""Batch helpers for structured-game rollouts with the original simulator semantics."""
import copy
import random

import numpy as np


class EpisodeBatch:
    """Vectorized execution of the generated three-action structured games."""
    def __init__(self, rules, reset_seeds, horizon):
        if not rules or len(rules) != len(reset_seeds) or horizon < 1:
            raise ValueError('invalid rules, reset seeds, or horizon')
        if any(rule.n != 3 or rule.memory_depth != 3 for rule in rules):
            raise ValueError('batch requires three-action depth-three games')
        allowed = {'own_is', 'state_min', 'state_max', 'repeat', 'own_count'}
        for rule in rules:
            if any(predicate.kind not in allowed for clause in (*rule.rewards, *rule.transitions)
                   for predicate in clause.when):
                raise ValueError('unsupported predicate')
            if any(clause.probability == 0 for clause in rule.transitions) or sum(
                0 < clause.probability < 1 for clause in rule.transitions
            ) > 1:
                raise ValueError('unsupported stochastic gates')
        self.rules = rules
        self.count = len(rules)
        self.initial_state = np.zeros((self.count, 3), dtype=int)
        self.levels = np.ones((self.count, 3), dtype=int)
        self.public = np.zeros((self.count, 3), dtype=bool)
        self.state_index = []
        self.base = np.stack([np.asarray(r.base.values).reshape(2,3,3,3,3) for r in rules])
        self.scale = np.asarray([r.reward_scale for r in rules])
        for i, rule in enumerate(rules):
            names = {state.name: j for j, state in enumerate(rule.states)}
            self.state_index.append(names)
            for j, state in enumerate(rule.states):
                self.initial_state[i,j] = state.initial
                self.levels[i,j] = state.levels
                self.public[i,j] = state.public
        self.reward_meta, rewards = self._compile('rewards')
        self.transition_meta, transitions = self._compile('transitions')
        self.reward_amounts = np.zeros(rewards.shape)
        self.transition_prob = np.zeros(transitions.shape)
        self.transition_delta = np.zeros(transitions.shape,dtype=int)
        self.transition_both = np.zeros_like(self.transition_delta,dtype=bool)
        self.transition_slot = np.zeros_like(self.transition_delta)
        for row, rule in enumerate(rules):
            for column, clause in enumerate(rule.rewards):
                self.reward_amounts[row,column] = clause.value
            for column, clause in enumerate(rule.transitions):
                self.transition_prob[row,column] = clause.probability
                self.transition_delta[row,column] = clause.delta
                self.transition_both[row,column] = clause.scope == 'both'
                self.transition_slot[row,column] = self.state_index[row][clause.state]
        self.index = np.arange(self.count)
        self.reset(reset_seeds, horizon)

    def _compile(self, attr):
        kinds = {'own_is':1,'repeat':2,'own_count':3,'state_min':4,'state_max':5}
        groups = [getattr(r,attr) for r in self.rules]
        clauses = max(map(len,groups))
        terms = max((len(clause.when) for row in groups for clause in row),default=0)
        shape = (self.count,clauses)
        # Metadata is grouped by conjunction position, with operation 0 for padding.
        meta = np.zeros((terms,*shape,6),dtype=np.int16)
        for i, group in enumerate(groups):
            for j, clause in enumerate(group):
                for k, p in enumerate(clause.when):
                    meta[k,i,j] = [kinds[p.kind], 0 if p.action is None else p.action,
                                   p.lag-1,p.window,p.count,p.negate]
                    if p.state is not None:
                        meta[k,i,j,1] = self.state_index[i][p.state]
                        meta[k,i,j,4] = p.threshold
        return [tuple(entry.transpose(2,0,1)) for entry in meta], np.empty(shape)

    def reset(self, reset_seeds, horizon):
        if len(reset_seeds) != self.count or horizon < 1:
            raise ValueError('invalid reset batch size or horizon')
        self.horizon = horizon
        self.elapsed = 0
        histories = []
        for seed in reset_seeds:
            # StructuredEnv.reset draws two uniform actions per history slot.
            rng = random.Random(int(seed))
            histories.append([(rng.randrange(3), rng.randrange(3)) for _ in range(3)])
        self.history = np.array(histories, dtype=int)
        self.previous_reward = np.zeros(self.count)
        self.state = self.initial_state.copy()

    def fork_reset(self, reset_seeds, horizon):
        result = copy.copy(self)
        result.reset(reset_seeds, horizon)
        return result

    @property
    def last(self):
        return self.history[:, 0]

    def features(self):
        x = np.zeros((self.count, 3, 20))
        x[:,:,0] = 1
        x[:,:,1:3] = (self.state[:,:2] * self.public[:,:2])[:,None,:] / 2
        x[:,:,3] = self.previous_reward[:,None]
        x[:,:,4] = float(self.elapsed == 0)
        actions = np.arange(3)[None,:]
        for lag in range(3):
            x[:,:,5+3*lag] = actions == self.history[:,lag,0,None]
            x[:,:,6+3*lag] = actions == self.history[:,lag,1,None]
            x[:,:,7+3*lag] = 1
        return x

    def _matches(self, meta, actions, shape):
        yes = np.ones((*shape,2),dtype=bool)
        rows = self.index[:,None]
        for op, ref, lag, window, limit, negate in meta:
            action_hit = actions[:,None,:] == ref[...,None]
            repeats = actions[:,None,:] == self.history[rows,lag]
            counts = ((self.history[:,None,:,:] == ref[...,None,None]) &
                      (np.arange(3)<window[...,None])[...,None]).sum(axis=2)
            state = self.state[rows,ref]
            match = ((op == 0)[...,None] |
                     ((op == 1)[...,None] & action_hit) |
                     ((op == 2)[...,None] & repeats) |
                     ((op == 3)[...,None] & (counts >= limit[...,None])) |
                     ((op == 4) & (state >= limit))[...,None] |
                     ((op == 5) & (state <= limit))[...,None])
            yes &= match ^ negate[...,None].astype(bool)
        return yes

    def step(self, actions, opponent_actions, gate_uniforms):
        if self.elapsed >= self.horizon:
            raise RuntimeError('reset is required before stepping')
        actions = np.asarray(actions)
        opponent_actions = np.asarray(opponent_actions)
        gate_uniforms = np.asarray(gate_uniforms)
        if any(x.shape != (self.count,) for x in (actions, opponent_actions, gate_uniforms)):
            raise ValueError('each input must contain one entry per environment')
        if (not np.issubdtype(actions.dtype, np.integer) or
            not np.issubdtype(opponent_actions.dtype, np.integer) or
            np.any((actions < 0) | (actions > 2)) or np.any((opponent_actions < 0) | (opponent_actions > 2))):
            raise ValueError('actions must be integer indices between zero and two')
        joint = np.stack([actions, opponent_actions], axis=1)
        index = np.arange(self.count)
        previous = self.last.copy()
        rewards = self.base[index,:,previous[:,0],previous[:,1],actions,opponent_actions].copy()
        earned = self._matches(self.reward_meta,joint,self.reward_amounts.shape)
        amounts = self.reward_amounts[...,None]*earned
        # Add in each rule's original order, including duplicate clauses.
        for by_clause in amounts.transpose(1,0,2):
            rewards += by_clause
        conditions = self._matches(self.transition_meta,joint,self.transition_prob.shape)
        fired = np.where(self.transition_both,conditions.all(axis=-1),conditions.any(axis=-1))
        fired &= (self.transition_prob == 1.) | (gate_uniforms[:,None] < self.transition_prob)
        delta = np.zeros_like(self.state)
        np.add.at(delta,(index[:,None],self.transition_slot),self.transition_delta*fired)
        self.state = np.minimum(self.levels-1, np.maximum(0, self.state+delta))
        self.history = np.concatenate([joint[:,None,:],self.history[:,:-1,:]],axis=1)
        self.elapsed += 1
        rewards *= self.scale[:,None]
        self.previous_reward = rewards[:,0].copy()
        return rewards


def opponent_distribution(q, last):
    """Return epsilon-greedy distributions for opponent-local previous actions."""
    last = np.asarray(last)
    q = np.asarray(q)
    if q.shape != (len(last), 3, 3, 3) or last.shape != (len(q), 2):
        raise ValueError('invalid opponent table or previous joint action shape')
    values = q[np.arange(len(q)), last[:, 1], last[:, 0]]
    best = np.isclose(values, values.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    return .25 / 3 + .75 * best / best.sum(axis=1, keepdims=True)


def opponent_update(q, previous, current, opponent_actions, rewards, terminal=False):
    """In-place Q-learning update matching structured_scale.Opponent."""
    count = len(q)
    index = np.arange(count)
    previous, current = np.asarray(previous), np.asarray(current)
    actions, rewards = np.asarray(opponent_actions), np.asarray(rewards)
    if previous.shape != (count, 2) or current.shape != (count, 2):
        raise ValueError('invalid joint histories')
    old = q[index, previous[:, 1], previous[:, 0], actions]
    future = q[index, current[:, 1], current[:, 0]].max(axis=1)
    delta = .3 * (rewards + .8 * future * (1 - np.asarray(terminal, dtype=int)) - old)
    q[index, previous[:, 1], previous[:, 0], actions] += delta
    return delta


class StudentBatch:
    """Independent action-wise Gaussian posterior means and inverse precisions."""
    def __init__(self, count, prior):
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (20,) or count < 1:
            raise ValueError('expected a 20-dimensional prior and positive batch size')
        self.mean = np.broadcast_to(prior, (count, 3, 20)).copy()
        self.inverse = np.broadcast_to(np.eye(20), (count, 3, 20, 20)).copy()

    def predict(self, x):
        return np.einsum('bad,bad->ba', self.mean, x)

    def uncertainty(self, x):
        return np.sqrt(np.maximum(0, np.einsum('bad,bade,bae->ba', x, self.inverse, x)))

    def observe(self, x, actions, rewards):
        index = np.arange(len(self.mean))
        action = np.asarray(actions)
        selected = x[index, action]
        inverse = self.inverse[index, action]
        product = np.einsum('bij,bj->bi', inverse, selected)
        denominator = 1 + np.einsum('bi,bi->b', selected, product)
        residual = np.asarray(rewards) - np.einsum('bi,bi->b', selected, self.mean[index, action])
        self.mean[index, action] += product * (residual / denominator)[:, None]
        self.inverse[index, action] -= np.einsum('bi,bj->bij', product, product) / denominator[:, None, None]
