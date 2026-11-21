"""Two-arena simultaneous games and finite latent-type likelihoods."""
from itertools import product

import numpy as np

RATES = np.array([0.12, 0.65])
TYPES = np.array(list(product((-1, 1), (-1, 1), RATES)))
SIGNS = TYPES[:, :2].astype(int)
FAST = TYPES[:, 2] == RATES[1]
SAME = SIGNS[:, 0] == SIGNS[:, 1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mixture(q, r):
    return 0.5 * np.where(SAME, q, 1-q) * np.where(FAST, r, 1-r)


def catalog(rng, count, concordance, fast_fraction):
    # Exact crossed strata. The rank of a game determines sign and speed;
    # across composition arms retain its public parameters and random draws.
    i = rng.permutation(count)
    sign = np.where(i % 2, 1, -1)
    same = i // 4 < round(concordance * count / 4)
    rate = RATES[((i // 2 % 2 == 1) & (fast_fraction > 0)).astype(int)]
    return dict(signs=np.column_stack([sign, np.where(same, sign, -sign)]),
                rate=rate, gain=rng.uniform(1.5, 3.5, (count, 2)),
                bonus=rng.choice([-1, 1], (count, 2)) * rng.uniform(.05, .20, (count, 2)))


def posterior(loglik, weights):
    scaled = np.exp(loglik - loglik.max(axis=-1, keepdims=True)) * weights
    return scaled / scaled.sum(axis=-1, keepdims=True)


def fit_prior(loglik):
    q, r = .5, .5
    for _ in range(80):
        resp = posterior(loglik, mixture(q, r))
        q = (1 + resp[:, SAME].sum()) / (2 + len(resp))
        r = (1 + resp[:, FAST].sum()) / (2 + len(resp))
    return np.array([q, r])


def pooled_features(arena, gain, traces):
    # Separate intercept and two action-history covariates per arena.
    columns = np.concatenate([np.ones(arena.shape + (1,)), gain[..., None] * traces], axis=-1)
    return np.concatenate([columns * (arena == c)[..., None] for c in range(2)], axis=-1)


def fit_pooled(x, y):
    x = x.reshape(-1, 6)
    y = y.ravel()
    coef = np.zeros(6)
    for _ in range(12):
        p = sigmoid(x @ coef)
        hessian = x.T @ ((p * (1-p))[:, None] * x) + np.eye(6)
        coef += np.linalg.solve(hessian, x.T @ (y-p) - coef)
    return coef


def training(rng, count, q, r):
    games = catalog(rng, count, q, r)
    resets, horizon = 128 // count, 16
    shape = (count, resets, horizon)
    arenas = rng.integers(0, 2, shape, dtype=np.int8)
    preferred = rng.integers(0, 2, (*shape[:2], horizon // 4), dtype=np.int8).repeat(4, axis=2)
    actions = np.where(rng.random(shape) < .85, preferred, 1-preferred)
    uniforms = rng.random(shape)
    moves = np.empty_like(actions)
    likelihood = np.zeros((count, 8))
    traces = np.full((*shape[:2], 2), .5)
    x = np.empty((*shape, 6))
    index = np.arange(count)[:, None]
    fast = (games['rate'] == RATES[1])[:, None, None]
    for t in range(horizon):
        c = arenas[..., t]
        strength = games['gain'][index, c]
        evidence = strength[..., None] * traces
        true_field = np.take_along_axis(evidence, fast.astype(int), axis=2)[..., 0]
        p = sigmoid(games['signs'][index, c] * true_field)
        b = (uniforms[..., t] < p).astype(np.int8)
        moves[..., t] = b
        logits = SIGNS.T[c] * evidence[..., FAST.astype(int)]
        likelihood -= np.logaddexp(0, (1 - 2*b[..., None]) * logits).sum(axis=1)
        x[..., t, :] = pooled_features(c, strength, traces)
        traces += RATES * (2*actions[..., t, None] - 1 - traces)
    return dict(**games, arena=arenas, a=actions, b=moves, loglik=likelihood,
                fit=fit_prior(likelihood), pooled=fit_pooled(x, moves))
