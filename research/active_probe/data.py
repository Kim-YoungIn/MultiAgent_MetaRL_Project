"""Produce paired two-turn branches from structured games with learning opponents.

The sample is a random game and a reset. Every opening action receives the same
pre-opening trajectory, opponent draw, and gate uniform. Live terminal values
condition on that branch. Rewind values restore the shared pre-opening world
and opponent while retaining the recorded opening observation for a controller.
"""
from __future__ import annotations

import copy

import numpy as np

from mamrl.structured_games import StructuredEnv
from research.structured_oracle.run import Gate, RewardKernel
from research.structured_scale.run import FAMILIES, Opponent, game, step


N = 3
PREFIX = 4


def _public(obs):
    result = np.zeros(2, dtype=np.int8)
    for index, value in enumerate(obs.public_state[:2]):
        if value is not None:
            result[index] = value
    return result


def _empty(samples, prefix):
    arrays = {
        'game_seed': (samples,), 'game_fingerprint': (samples,), 'reset_seed': (samples,), 'motif': (samples,),
        'prefix_a': (samples, prefix), 'prefix_b': (samples, prefix),
        'prefix_reward': (samples, prefix), 'prefix_opp_reward': (samples, prefix),
        'prefix_u': (samples, prefix), 'prefix_public': (samples, prefix+1, 2),
        'init_hist': (samples, 3, 2), 'first_u': (samples,), 'gate_u': (samples,),
        'first_b': (samples,), 'first_reward': (samples, N),
        'first_opp_reward': (samples, N), 'post_public': (samples, N, 2),
        'next_value': (samples, N, 2, N), 'first_value': (samples, N),
        'opp_p': (samples, N), 'post_opp_p': (samples, N, N),
        'next_value_no_update': (samples, N, N), 'opponent_q_update': (samples, N),
    }
    integers = {'game_seed', 'reset_seed', 'motif', 'prefix_a', 'prefix_b',
                'prefix_public', 'init_hist', 'first_b', 'post_public'}
    return {name: np.empty(shape, dtype='U64' if name == 'game_fingerprint' else
                           np.int64 if name in integers else np.float64)
            for name, shape in arrays.items()}


def generate(seed: int, games: int, repeats: int, prefix: int = PREFIX) -> dict[str, np.ndarray]:
    """Draw balanced game motifs and generate every paired opening branch.

    ``games`` must be divisible by four so each motif contributes equally. Game
    seeds are unique within this call. Samples from one game are contiguous.
    The first action is intervened on; all earlier choices remain common. The
    reward and transition rules are those in ``StructuredEnv``. For the live
    terminal payoff the opponent distribution is computed after its Q update.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(games) is not int or games <= 0 or games % len(FAMILIES):
        raise ValueError('games must be a positive multiple of four')
    if type(repeats) is not int or repeats <= 0:
        raise ValueError('repeats must be a positive integer')
    if type(prefix) is not int or prefix < 0:
        raise ValueError('prefix must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    result = _empty(games * repeats, prefix)
    game_seeds = set()
    for game_index in range(games):
        while True:
            game_seed = int(rng.integers(0, 2**62))
            if game_seed not in game_seeds:
                game_seeds.add(game_seed)
                break
        motif = game_index % len(FAMILIES)
        rule = game(game_seed, FAMILIES[motif], N, 3)
        gates = [clause for clause in rule.transitions if clause.probability < 1]
        if len(gates) > 1 or any(clause.probability == 0 for clause in gates):
            raise ValueError('paired gates require at most one stochastic transition')
        fingerprint = rule.fingerprint()
        kernel = RewardKernel(rule)
        for repeat in range(repeats):
            row = game_index * repeats + repeat
            reset_seed = int(rng.integers(0, 2**62))
            env = StructuredEnv(rule, max_steps=prefix+2)
            obs = env.reset(seed=reset_seed)
            opponent = Opponent(N)
            result['game_seed'][row] = game_seed
            result['game_fingerprint'][row] = fingerprint
            result['reset_seed'][row] = reset_seed
            result['motif'][row] = motif
            result['init_hist'][row] = obs[0].action_history
            result['prefix_public'][row, 0] = _public(obs[0])
            for t in range(prefix):
                action = int(rng.integers(N))
                uniform = float(rng.random())
                transition, other = step(env, opponent, obs, action, uniform)
                result['prefix_a'][row, t] = action
                result['prefix_b'][row, t] = other
                result['prefix_u'][row, t] = uniform
                result['prefix_reward'][row, t] = transition.rewards[0]
                result['prefix_opp_reward'][row, t] = transition.rewards[1]
                obs = transition.observations
                result['prefix_public'][row, t+1] = _public(obs[0])

            uniform = float(rng.random())
            gate_uniform = float(rng.random())
            probs = opponent.distribution(obs[1])
            result['first_u'][row] = uniform
            result['gate_u'][row] = gate_uniform
            result['opp_p'][row] = probs
            result['first_b'][row] = min(int(np.searchsorted(np.cumsum(probs), uniform, side='right')), N-1)
            pre_values = kernel(env) @ probs
            result['first_value'][row] = pre_values
            for action in range(N):
                branch = copy.copy(env)
                branch._rng = Gate(gate_uniform)
                branch_opponent = copy.deepcopy(opponent)
                transition, other = step(branch, branch_opponent, obs, action, uniform)
                if other != result['first_b'][row]:
                    raise AssertionError('opponent action changed across simultaneous openings')
                post_probs = branch_opponent.distribution(transition.observations[1])
                result['first_reward'][row, action] = transition.rewards[0]
                result['first_opp_reward'][row, action] = transition.rewards[1]
                result['post_public'][row, action] = _public(transition.observations[0])
                result['post_opp_p'][row, action] = post_probs
                result['opponent_q_update'][row, action] = branch_opponent.change - opponent.change
                frozen_probs = opponent.distribution(transition.observations[1])
                result['next_value_no_update'][row, action] = kernel(branch) @ frozen_probs
                result['next_value'][row, action, 0] = kernel(branch) @ post_probs
                result['next_value'][row, action, 1] = pre_values
    return result
