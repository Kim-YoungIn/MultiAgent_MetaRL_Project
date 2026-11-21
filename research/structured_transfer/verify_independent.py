"""Independent audit of structured-transfer saved data and selected trajectories."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_toy_models'))
from mamrl.structured_games import StructuredEnv, StructuredGame


def close(name, actual, expected, atol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=atol, atol=atol):
        error = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
        raise AssertionError(f'{name}: shapes {actual.shape} {expected.shape}, max error {error}')


def feature(obs):
    f = np.zeros(20)
    f[0] = 1
    visible = [x for x in obs.public_state[:2]]
    if len(visible) == 1:
        visible *= 2
    f[1:3] = np.asarray(visible) / 2
    f[3] = obs.previous_own_reward
    f[4] = obs.episode_start_flag
    a, b = obs.action_history[0]
    f[5+3*a+b] = 1
    f[14+obs.action_history[1][0]] = 1
    f[17+obs.action_history[2][0]] = 1
    return f


def fit_from_raw(games):
    meta = np.zeros((3,20))
    pooled = np.zeros((3,20))
    for action in range(3):
        a_meta = .01*np.eye(20)
        b_meta = np.zeros(20)
        a_pool = .01*np.eye(20)
        b_pool = np.zeros(20)
        for game in games:
            rows = [row for row in game['rows'] if row[1] == action]
            if not rows:
                continue
            x = np.array([r[0] for r in rows])
            y = np.array([r[3] for r in rows])
            # Marginalize game coefficients using the observation-space covariance.
            v = np.linalg.solve(np.eye(len(rows)) + x @ x.T, x)
            a_meta += x.T @ v
            b_meta += v.T @ y
            a_pool += x.T @ x
            b_pool += x.T @ y
        meta[action] = np.linalg.solve(a_meta, b_meta)
        pooled[action] = np.linalg.solve(a_pool, b_pool)
    return meta, pooled


def distribution(q, obs):
    values = q[obs.previous_own_action, obs.previous_opponent_action]
    best = np.abs(values-values.max()) <= 1e-12
    return np.ones(3)/12 + .75*best / best.sum()


def observe_q(q, obs, b, reward, next_obs, terminal):
    key = (obs.previous_own_action, obs.previous_opponent_action)
    key_next = (next_obs.previous_own_action, next_obs.previous_opponent_action)
    change = .3*(reward + (0 if terminal else .8*q[key_next].max()) - q[key][b])
    q[key][b] += change
    return abs(change)


def action(prior, frozen, rows, obs):
    x = feature(obs)
    predicted = np.empty(3)
    for a in range(3):
        sample = [(row[0], row[2]) for row in rows if row[1] == a]
        if frozen or not sample:
            w = prior[a]
        else:
            xx = np.stack([item[0] for item in sample])
            yy = np.array([item[1] for item in sample])
            w = np.linalg.solve(np.eye(20)+xx.T @ xx, prior[a]+xx.T @ yy)
        predicted[a] = x @ w
    return int(np.argmax(predicted))


def manual_reward(rule, env, a, b):
    history, state, actions = env._history, env._state, (a,b)
    base = np.asarray(rule['base_values']).reshape(2,3,3,3,3)
    reward = base[:,history[0][0],history[0][1],a,b].copy()
    for clause in rule['rewards']:
        for player in range(2):
            matches = []
            for predicate in clause['when']:
                kind = predicate['kind']
                if kind == 'own_is':
                    match = actions[player] == predicate['action']
                elif kind == 'state_min':
                    match = state[predicate['state']] >= predicate['threshold']
                elif kind == 'state_max':
                    match = state[predicate['state']] <= predicate['threshold']
                elif kind == 'own_count':
                    match = sum(pair[player] == predicate['action'] for pair in history[:predicate['window']]) >= predicate['count']
                elif kind == 'repeat':
                    match = actions[player] == history[predicate['lag']-1][player]
                else:
                    raise AssertionError(f'unrecognized generated reward predicate {kind}')
                matches.append(not match if predicate['negate'] else match)
            if all(matches):
                reward[player] += clause['value']
    return reward * rule['reward_scale']

def replay_test(test, saved, index):
    game = StructuredGame.from_dict(test['game'])
    close('saved fingerprint', game.fingerprint() == test['fingerprint'], True, 0)
    env = StructuredEnv(game, max_steps=17)
    obs = env.reset(seed=test['seed'] + 1)
    q = np.zeros((3,3,3))
    priors = [saved['mixed'], saved['resource'], saved['mixed'], saved['resource'],
              saved['pooled'], saved['pooled'], np.zeros((3,20))]
    frozen = [False,False,True,True,True,False,False]
    rows = []
    changes = 0.
    budgets = [0,4,8,16]
    for t in range(17):
        x = feature(obs[0])
        if t in budgets:
            k = budgets.index(t)
            probabilities = distribution(q, obs[1])
            expected = []
            for a in range(3):
                rewards = []
                for b in range(3):
                    manual = manual_reward(test['game'], env, a, b)
                    close('manually indexed hypothetical reward', manual, copy.deepcopy(env).step(a,b).rewards)
                    rewards.append(manual[0])
                expected.append(np.dot(probabilities,rewards))
            close('query oracle', expected, saved['oracle'][index,k])
            close('opponent distribution', probabilities, test['probe_states'][k]['opponent_probs'])
            for j in range(7):
                a = action(priors[j], frozen[j], rows, obs[0])
                close('query action', a, saved['actions'][index,j,k], 0)
                close('query value', expected[a], saved['values'][index,j,k])
        if t == 16:
            break
        xr,a,b,r0,r1,u = test['probe'][t]
        close('probe observed features', x, xr)
        probs = distribution(q, obs[1])
        close('probe opponent draw', b, np.searchsorted(np.cumsum(probs), u, side='right'), 0)
        outcome = env.step(a,b)
        close('probe reward', outcome.rewards, (r0,r1))
        changes += observe_q(q,obs[1],b,r1,outcome.observations[1],outcome.truncated)
        rows.append((x,a,r0))
        obs = outcome.observations
    close('probe Q changes', changes, saved['probe_changes'][index])
    for j in range(7):
        env = StructuredEnv(game, max_steps=32)
        obs = env.reset(seed=test['seed']+2)
        q = np.zeros((3,3,3))
        rows, changes = [], 0.
        for t in range(32):
            x = feature(obs[0])
            a = action(priors[j], frozen[j], rows, obs[0])
            close('rollout action', a, saved['rollout_a'][index,j,t], 0)
            probs = distribution(q,obs[1])
            b = int(np.searchsorted(np.cumsum(probs),test['rollout_uniforms'][t],side='right'))
            close('rollout opponent action', b, saved['rollout_b'][index,j,t], 0)
            outcome=env.step(a,b)
            close('rollout rewards', outcome.rewards, saved['rewards'][index,j,t])
            public = list(outcome.observations[0].public_state[:2])
            if len(public) == 1:
                public *= 2
            close('rollout state', public, saved['rollout_states'][index,j,t])
            changes += observe_q(q,obs[1],b,outcome.rewards[1],outcome.observations[1],outcome.truncated)
            rows.append((x,a,outcome.rewards[0]))
            obs=outcome.observations
        close('rollout Q changes', changes, saved['changes'][index,j])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    out=args.output
    manifest=json.loads((out/'manifest.json').read_text())
    source_mismatches = [filename for filename,digest in manifest['source_sha256'].items()
                         if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest]
    differences=[]
    for index in range(manifest['blocks']):
        folder=out/f'block_{index:03d}'
        train=json.loads((folder/'train.json').read_text())
        tests=json.loads((folder/'test.json').read_text())
        with np.load(folder/'results.npz',allow_pickle=False) as data:
            saved=dict(data)
        identities=[item['fingerprint'] for arm in train.values() for item in arm]+[item['fingerprint'] for item in tests]
        if len(identities)!=96 or len(set(identities))!=96:
            raise AssertionError(f'{folder}: exact fingerprint duplicate')
        for arm in train:
            if len(train[arm])!=32 or any(len(item['rows'])!=32 for item in train[arm]):
                raise AssertionError(f'{folder}: wrong training budget')
        for i in range(32):
            for j in range(7):
                for k in range(4):
                    close('values indexed by actions',saved['values'][i,j,k],saved['oracle'][i,k,saved['actions'][i,j,k]])
        gain=saved['values'][:,:,2]-saved['values'][:,:,0]
        differences.append((gain[:,0]-gain[:,6]).mean())
        if index == 0:
            mixed, pooled = fit_from_raw(train['mixed'])
            resource, _ = fit_from_raw(train['resource'])
            close('mixed empirical Bayes fit', mixed, saved['mixed'])
            close('resource empirical Bayes fit', resource, saved['resource'])
            close('pooled fit', pooled, saved['pooled'])
            for i in range(4):
                replay_test(tests[i],saved,i)
    differences=np.asarray(differences)
    rng=np.random.default_rng(470919)
    samples=rng.integers(0,len(differences),(10000,len(differences)))
    bounds=np.quantile(differences[samples].mean(axis=1),[.025,.975])
    result={'reward_tables_manually_reconstructed':True,'source_hash_mismatches':source_mismatches,'blocks':len(differences),'replayed_motifs':4,'primary_estimate':float(differences.mean()),
            'bootstrap_95_interval':bounds.tolist(),'training_fits_reconstructed_block':0,
            'scope':'all block values indexed against saved oracles; block 0 one game per motif replayed from saved game and actions, including all seven deployments'}
    analysis=json.loads((out/'analysis.json').read_text())
    close('reported primary',analysis['primary_meta_vs_scratch_gain'],[result['primary_estimate'],*bounds])
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
