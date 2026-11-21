"""Paired probes and live deployments with separate state for every policy."""
import numpy as np

from .game import FAST, RATES, SIGNS, catalog, mixture, pooled_features, posterior, sigmoid

BUDGETS = (0, 1, 2, 4, 8, 16)
HORIZON = 32


class Controller:
    def __init__(self, name, prior, pooled, gain):
        self.name, self.prior, self.coef, self.gain = name, prior, pooled, gain
        n = len(gain)
        self.i = np.arange(n)
        self.traces = np.full((n, 2), .5)
        self.loglik = np.zeros((n, 8))
        self.ones = np.ones((n, 2))
        self.total = np.full((n, 2), 2.)

    def logits(self, c):
        return SIGNS.T[c] * self.gain[self.i, c, None] * self.traces[:, FAST.astype(int)]

    def predict(self, c):
        if self.name == 'frequency':
            return self.ones[self.i, c] / self.total[self.i, c]
        if self.name == 'pooled':
            x = pooled_features(c, self.gain[self.i, c], self.traces)
            return sigmoid(x @ self.coef)
        if self.name == 'frozen':
            return np.full(len(c), .5)  # each sign marginal stays exactly symmetric
        return (posterior(self.loglik, self.prior) * sigmoid(self.logits(c))).sum(axis=1)

    def observe(self, c, a, b):
        self.loglik -= np.logaddexp(0, (1-2*b[:, None]) * self.logits(c))
        self.ones[self.i, c] += b
        self.total[self.i, c] += 1
        self.traces += RATES * (2*a[:, None]-1-self.traces)


class Opponent:
    def __init__(self, tasks, local):
        self.sign, self.rate, self.gain = tasks['signs'], tasks['rate'], tasks['gain']
        self.i = np.arange(len(self.rate))
        self.m = np.full((len(self.i), 2), .5)
        self.local = local

    def field(self, c):
        return self.m[self.i, c if self.local else 0]

    def predict(self, c):
        return sigmoid(self.field(c) * self.gain[self.i, c] * self.sign[self.i, c])

    def update(self, c, a):
        self.m[self.i, c if self.local else 0] += self.rate * (2*a-1-self.field(c))


def choose(pred, bonus):
    return (2*pred - 1 + bonus > 0).astype(np.int8)


def expected_value(action, truth, bonus):
    return np.where(action == 1, truth + bonus, 1-truth)


def panel(rng, reps):
    n = 8 * reps
    return dict(**catalog(rng, n, .5, .5), source=rng.integers(0, 2, n),
                probe_a=(rng.random((n, 16)) < .8).astype(np.int8),
                probe_u=rng.random((n, 16)), online_arena=rng.integers(0, 2, (n, HORIZON)),
                online_u=rng.random((n, HORIZON)))


def evaluate(tasks, fits, coef, local):
    priors = {name: mixture(*qr) for name, qr in fits.items()}
    priors.update({name: mixture(q, .5) for name, q in [('independent', .5), ('frozen', .75),
                  ('frequency', .5), ('pooled', .5), ('oracle75', .75), ('oracle50', .5),
                  ('oracle25', .25), ('type_oracle', .5)]})
    n = len(tasks['rate'])
    i, source = np.arange(n), tasks['source']
    m = len(priors)
    curves = {key: np.empty((m, n, len(BUDGETS), 2)) for key in ('pred', 'value', 'brier')}
    live = {key: np.empty((m, n, HORIZON), dtype=np.float64 if key not in ('focal_move','opponent_move') else np.int8)
            for key in ('online_reward', 'online_expected', 'opponent_reward', 'opponent_move',
                        'opponent_prob', 'field_abs', 'focal_move')}
    probe_b = np.empty_like(tasks['probe_a'])
    for j, (name, prior) in enumerate(priors.items()):
        def reset():
            return Controller(name, prior, coef, tasks['gain']), Opponent(tasks, local)

        policy, opponent = reset()
        for t in range(max(BUDGETS)+1):
            if t in BUDGETS:
                k = BUDGETS.index(t)
                for arena_index, c in enumerate((source, 1-source)):
                    truth = opponent.predict(c)
                    pred = truth if name == 'type_oracle' else policy.predict(c)
                    bonus = tasks['bonus'][i, c]
                    ac = choose(pred, bonus)
                    curves['pred'][j, :, k, arena_index] = pred
                    curves['value'][j, :, k, arena_index] = expected_value(ac, truth, bonus)
                    curves['brier'][j, :, k, arena_index] = (truth-pred)**2 + truth*(1-truth)
            if t < max(BUDGETS):
                a = tasks['probe_a'][:, t]
                b = (tasks['probe_u'][:, t] < opponent.predict(source)).astype(np.int8)
                probe_b[:, t] = b
                policy.observe(source, a, b)
                opponent.update(source, a)
        policy, opponent = reset()
        for t, c in enumerate(tasks['online_arena'].T):
            truth = opponent.predict(c)
            pred = truth if name == 'type_oracle' else policy.predict(c)
            bonus = tasks['bonus'][i, c]
            a = choose(pred, bonus)
            b = (tasks['online_u'][:, t] < truth).astype(np.int8)
            live['online_reward'][j, :, t] = (a == b) + bonus*a
            live['online_expected'][j, :, t] = expected_value(a, truth, bonus)
            live['opponent_reward'][j, :, t] = (a == b) * tasks['gain'][i, c] * tasks['signs'][i, c]
            live['field_abs'][j, :, t] = abs(opponent.field(c))
            live['opponent_prob'][j, :, t] = truth
            live['focal_move'][j, :, t], live['opponent_move'][j, :, t] = a, b
            policy.observe(c, a, b)
            opponent.update(c, a)
    return dict(**tasks, **curves, **live, methods=np.array(list(priors)), budgets=np.array(BUDGETS), probe_b=probe_b)
