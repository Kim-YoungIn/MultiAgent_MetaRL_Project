"""Independently replay saved games and recompute posterior and payoff queries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from mamrl.structured_games import StructuredEnv, StructuredGame

D = 20
METHODS = ('meta_mixed', 'meta_resource', 'frozen_mixed', 'frozen_resource',
           'pooled', 'pooled_adapt', 'scratch')


def vector(o):
    history = o.action_history
    return np.array([1, o.public_state[0] / 2, o.public_state[min(1, len(o.public_state)-1)] / 2,
                     o.previous_own_reward, int(o.episode_start_flag),
                     *[int((history[0][0],history[0][1]) == (a,b)) for a in range(3) for b in range(3)],
                     *[int(history[1][0] == a) for a in range(3)],
                     *[int(history[2][0] == a) for a in range(3)]], float)


def distribution(table, obs):
    q = table[obs.previous_own_action, obs.previous_opponent_action]
    best = np.isclose(q, q.max(), rtol=0, atol=1e-12)
    return 1 / 12 + .75 * best / best.sum()


def update(table, obs, action, reward, newobs, terminal):
    index = (obs.previous_own_action, obs.previous_opponent_action, action)
    before = table[index]
    future = 0 if terminal else table[newobs.previous_own_action,newobs.previous_opponent_action].max()
    table[index] = before + .3*(reward + .8*future-before)
    return abs(table[index]-before)


def check_close(a, b, label):
    if not np.allclose(a, b, rtol=1e-9, atol=1e-10):
        raise AssertionError(label + f' mismatch: {a} != {b}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    out=args.output
    manifest=json.loads((out/'manifest.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        source=Path(path).read_bytes()
        if path.endswith('PROTOCOL.md') and hashlib.sha256(source).hexdigest() != digest:
            source=source.split(b'\n## Recorded implementation deviations, discovered after final data were opened\n')[0]
        assert hashlib.sha256(source).hexdigest() == digest
    all_fingerprints=set()
    block_diffs=[]
    counts={'training_turns':0, 'probe_turns':0, 'queries':0, 'rollout_turns':0}
    for block in range(manifest['blocks']):
        folder=out/f'block_{block:03d}'
        train=json.loads((folder/'train.json').read_text())
        tests=json.loads((folder/'test.json').read_text())
        d=np.load(folder/'results.npz')
        for arm in ('mixed','resource'):
            sum_meta=np.tile(np.eye(D)*.01,(3,1,1))
            sum_meta_y=np.zeros((3,D))
            sum_pool=sum_meta.copy()
            sum_pool_y=sum_meta_y.copy()
            for i,entry in enumerate(train[arm]):
                game=StructuredGame.from_dict(entry['game'])
                digest=game.fingerprint()
                assert digest == entry['fingerprint'] and digest not in all_fingerprints
                all_fingerprints.add(digest)
                env=StructuredEnv(game,max_steps=32)
                obs=env.reset(seed=entry['seed']+1)
                table=np.zeros((3,3,3))
                for row in entry['rows']:
                    x,a,b,y,z=row
                    check_close(x,vector(obs[0]),'train features')
                    result=env.step(a,b)
                    check_close((y,z),result.rewards,'train rewards')
                    update(table,obs[1],b,z,result.observations[1],result.truncated)
                    obs=result.observations
                    counts['training_turns']+=1
                for a in range(3):
                    rows=[row for row in entry['rows'] if row[1]==a]
                    if not rows: continue
                    x=np.array([r[0] for r in rows]); y=np.array([r[3] for r in rows])
                    gram=x.T@x; cross=x.T@y
                    # Marginal normal equations, equivalent to task-covariance inverse.
                    cov=np.eye(len(rows))+x@x.T
                    sum_meta[a] += x.T@np.linalg.solve(cov,x)
                    sum_meta_y[a] += x.T@np.linalg.solve(cov,y)
                    sum_pool[a] += gram
                    sum_pool_y[a] += cross
            estimate=np.stack([np.linalg.solve(sum_meta[a],sum_meta_y[a]) for a in range(3)])
            check_close(estimate,d[arm],'empirical Bayes')
            if arm=='mixed':
                estimate=np.stack([np.linalg.solve(sum_pool[a],sum_pool_y[a]) for a in range(3)])
                check_close(estimate,d['pooled'],'pooled')
        means=[d['mixed'],d['resource'],d['mixed'],d['resource'],d['pooled'],d['pooled'],np.zeros((3,D))]
        for i,entry in enumerate(tests):
            game=StructuredGame.from_dict(entry['game'])
            digest=game.fingerprint()
            assert digest==entry['fingerprint'] and digest not in all_fingerprints
            all_fingerprints.add(digest)
            env=StructuredEnv(game,max_steps=17)
            obs=env.reset(seed=entry['seed']+1)
            table=np.zeros((3,3,3))
            xs=[[] for _ in range(3)]; ys=[[] for _ in range(3)]
            total_change=0
            for t in range(17):
                x=vector(obs[0])
                if t in (0,4,8,16):
                    k=(0,4,8,16).index(t)
                    probs=distribution(table,obs[1])
                    check_close(probs,entry['probe_states'][k]['opponent_probs'],'opponent policy')
                    assert list(obs[0].public_state)==entry['probe_states'][k]['public']
                    oracle=np.array([sum(probs[b]*copy.deepcopy(env).step(a,b).rewards[0] for b in range(3)) for a in range(3)])
                    check_close(oracle,d['oracle'][i,k],'oracle payoff')
                    for m,prior in enumerate(means):
                        pred=[]
                        for a in range(3):
                            if m in (2,3,4) or not xs[a]:
                                weights=prior[a]
                            else:
                                xx=np.asarray(xs[a]); yy=np.asarray(ys[a])
                                weights=np.linalg.solve(np.eye(D)+xx.T@xx,prior[a]+xx.T@yy)
                            pred.append(weights@x)
                        a=int(d['actions'][i,m,k])
                        assert max(pred)-pred[a] < 1e-12, (block,i,t,m,pred,a)
                        check_close(oracle[a],d['values'][i,m,k],'query value')
                        counts['queries']+=1
                if t==16: break
                row=entry['probe'][t]
                check_close(row[0],x,'probe features')
                a,b,y,z,u=row[1:]
                drawn=min(int(np.searchsorted(np.cumsum(distribution(table,obs[1])),u,side='right')),2)
                assert b==drawn
                outcome=env.step(a,b)
                check_close((y,z),outcome.rewards,'probe rewards')
                total_change+=update(table,obs[1],b,z,outcome.observations[1],outcome.truncated)
                xs[a].append(x);ys[a].append(y)
                obs=outcome.observations
                counts['probe_turns']+=1
            check_close(total_change,d['probe_changes'][i],'probe opponent change')
            for m,prior in enumerate(means):
                env=StructuredEnv(game,max_steps=32)
                obs=env.reset(seed=entry['seed']+2)
                table=np.zeros((3,3,3)); xs=[[] for _ in range(3)]; ys=[[] for _ in range(3)]
                total_change=0
                for t in range(32):
                    x=vector(obs[0]); pred=[]
                    for a in range(3):
                        if m in (2,3,4) or not xs[a]: weights=prior[a]
                        else:
                            xx=np.asarray(xs[a]); yy=np.asarray(ys[a])
                            weights=np.linalg.solve(np.eye(D)+xx.T@xx,prior[a]+xx.T@yy)
                        pred.append(weights@x)
                    a=int(d['rollout_a'][i,m,t])
                    assert max(pred)-pred[a] < 1e-12, (block,i,t,m,pred,a)
                    u=entry['rollout_uniforms'][t]
                    b=min(int(np.searchsorted(np.cumsum(distribution(table,obs[1])),u,side='right')),2)
                    assert b==d['rollout_b'][i,m,t]
                    result=env.step(a,b)
                    check_close(result.rewards,d['rewards'][i,m,t],'deployment rewards')
                    state=result.observations[0].public_state[:2]
                    if len(state)==1: state=(state[0], state[0])
                    check_close(state,d['rollout_states'][i,m,t],'state')
                    total_change+=update(table,obs[1],b,result.rewards[1],result.observations[1],result.truncated)
                    xs[a].append(x);ys[a].append(result.rewards[0]);obs=result.observations
                    counts['rollout_turns']+=1
                check_close(total_change,d['changes'][i,m],'deployment opponent change')
        gains=d['values'][:,:,2]-d['values'][:,:,0]
        block_diffs.append((gains[:,0]-gains[:,6]).mean())
    rng=np.random.default_rng(470919)
    indices=rng.integers(0,len(block_diffs),(10000,len(block_diffs)))
    block_diffs=np.array(block_diffs)
    independently=[float(block_diffs.mean()),*np.quantile(block_diffs[indices].mean(axis=1),[.025,.975]).tolist()]
    report=json.loads((out/'analysis.json').read_text())
    check_close(independently,report['primary_meta_vs_scratch_gain'],'primary bootstrap')
    result={'status':'passed','counts':counts,'unique_fingerprints':len(all_fingerprints),
            'primary_recomputed':independently,'sources_verified':len(manifest['source_sha256']),
            'limitations':'Exact fingerprints only. Training opponent action draws are not reconstructed, but rewards and states are replayed.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
