"""Generate matched trajectories against frozen and learning opponents."""
from __future__ import annotations

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.structured_oracle.run import Gate, RewardKernel
from research.structured_scale.run import FAMILIES, Opponent, features, game, step

N = 3
HORIZON = 32
CONDITIONS = ('frozen', 'learning')


def _allocate(games):
    shapes = {
        'seed': (games,), 'reset_seed': (games,), 'warm_seed': (games,),
        'fingerprint': (games,), 'family': (games,),
        'warm_a': (games, HORIZON), 'warm_b': (games, HORIZON),
        'warm_r': (games, HORIZON), 'warm_u': (games, HORIZON),
        'warm_q': (games, N, N, N),
        'a': (games, HORIZON), 'u': (games, HORIZON),
        'gate_u': (games, HORIZON),
        'x': (games, 2, HORIZON + 1, N, 20),
        'y': (games, 2, HORIZON), 'opp_y': (games, 2, HORIZON),
        'b': (games, 2, HORIZON), 'truth': (games, 2, HORIZON + 1, N),
        'opp_p': (games, 2, HORIZON + 1, N),
        'q_delta': (games, 2, HORIZON),
    }
    integers = {'seed', 'reset_seed', 'warm_seed', 'family', 'warm_a', 'warm_b', 'a', 'b'}
    return {
        name: np.empty(shape, dtype='U64' if name == 'fingerprint' else
                       np.int64 if name in integers else np.float64)
        for name, shape in shapes.items()
    }


def sample(seed: int, games: int) -> dict[str, np.ndarray]:
    """Sample games with cloned warmed Q tables and paired random streams.

    q_delta is the absolute single-entry change in opponent Q. truth is the
    exact expected focal payoff against the current opponent distribution.
    The final t=32 query occurs before any subsequent action.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(games) is not int or games <= 0 or games % len(FAMILIES):
        raise ValueError('games must be a positive multiple of four')

    rng = np.random.default_rng(seed)
    result = _allocate(games)
    seen_seeds = set()
    seen_fingerprints = set()

    for row in range(games):
        while True:
            game_seed = int(rng.integers(0, 2**62))
            if game_seed not in seen_seeds:
                seen_seeds.add(game_seed)
                break
        family = row % len(FAMILIES)
        rule = game(game_seed, FAMILIES[family], N, 3)
        fingerprint = rule.fingerprint()
        if fingerprint in seen_fingerprints:
            raise ValueError('duplicate serialized game')
        seen_fingerprints.add(fingerprint)
        gates = [clause.probability for clause in rule.transitions if clause.probability < 1]
        if len(gates) > 1 or any(p <= 0 for p in gates):
            raise ValueError('paired gate stream requires at most one nonzero stochastic gate')

        warm_seed = int(rng.integers(0, 2**62))
        reset_seed = int(rng.integers(0, 2**62))
        if warm_seed == reset_seed:
            raise ValueError('warmup and evaluation resets must differ')
        result['seed'][row] = game_seed
        result['warm_seed'][row] = warm_seed
        result['reset_seed'][row] = reset_seed
        result['fingerprint'][row] = fingerprint
        result['family'][row] = family

        warm_env = StructuredEnv(rule, max_steps=HORIZON)
        warm_obs = warm_env.reset(seed=warm_seed)
        warmed = Opponent(N)
        warm_actions = rng.integers(N, size=HORIZON)
        warm_uniforms = rng.random(HORIZON)
        for t, (action, uniform) in enumerate(zip(warm_actions, warm_uniforms)):
            outcome, other = step(warm_env, warmed, warm_obs, int(action), float(uniform))
            result['warm_a'][row, t] = action
            result['warm_b'][row, t] = other
            result['warm_r'][row, t] = outcome.rewards[1]
            result['warm_u'][row, t] = uniform
            warm_obs = outcome.observations
        result['warm_q'][row] = warmed.q

        actions = rng.integers(N, size=HORIZON)
        uniforms = rng.random(HORIZON)
        gates = rng.random(HORIZON)
        result['a'][row], result['u'][row], result['gate_u'][row] = actions, uniforms, gates
        kernel = RewardKernel(rule)
        for condition, learn in enumerate((False, True)):
            env = StructuredEnv(rule, max_steps=HORIZON + 1)
            obs = env.reset(seed=reset_seed)
            opponent = Opponent(N)
            opponent.q = warmed.q.copy()
            for t in range(HORIZON + 1):
                probs = opponent.distribution(obs[1])
                result['x'][row, condition, t] = features(obs[0], N)
                result['opp_p'][row, condition, t] = probs
                result['truth'][row, condition, t] = kernel(env) @ probs
                if t == HORIZON:
                    break
                action = int(actions[t])
                other = min(int(np.searchsorted(np.cumsum(probs), uniforms[t], side='right')), N-1)
                env._rng = Gate(float(gates[t]))
                outcome = env.step(action, other)
                before = opponent.change
                if learn:
                    opponent.update(obs[1], other, outcome.rewards[1],
                                    outcome.observations[1], outcome.truncated)
                result['q_delta'][row, condition, t] = opponent.change - before
                result['y'][row, condition, t] = outcome.rewards[0]
                result['opp_y'][row, condition, t] = outcome.rewards[1]
                result['b'][row, condition, t] = other
                obs = outcome.observations
    return result
