"""Independent reconstruction of shared-history multi-agent adaptation results."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

BUDGETS = (0, 2, 4, 8, 16, 32)
METHODS = ('meta_adapt', 'pool_adapt', 'meta_frozen', 'pool_frozen', 'scratch')
D = 20


def check(label, actual, expected, atol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=atol):
        maximum = (np.max(np.abs(actual-expected)) if actual.shape == expected.shape else 'shape mismatch')
        raise AssertionError(f'{label}: {actual.shape}, {expected.shape}, max error {maximum}')


def independent_fit(data):
    """Construct preconditioned features from earlier same-action observations."""
    meta_gram = np.eye(D)*.01
    meta_rhs = np.zeros(D)
    pooled_gram = np.eye(D)*.01
    pooled_rhs = np.zeros(D)
    for game in range(data['x'].shape[0]):
        for condition in range(2):
            rows = [[], [], []]
            for t in range(32):
                action = int(data['a'][game,t])
                x = data['x'][game,condition,t,action]
                y = data['y'][game,condition,t]
                history = rows[action]
                if history:
                    X = np.stack([item[0] for item in history])
                    Y = np.array([item[1] for item in history])
                    gram = np.eye(D)+X.T@X
                    z = np.linalg.solve(gram, x)
                    fitted = np.linalg.solve(gram, X.T@Y)
                    offset = x@fitted
                else:
                    z, offset = x, 0.
                meta_gram += np.outer(z,z)
                meta_rhs += z*(y-offset)
                pooled_gram += np.outer(x,x)
                pooled_rhs += x*y
                rows[action].append((x,y))
    return np.stack((np.linalg.solve(meta_gram,meta_rhs),np.linalg.solve(pooled_gram,pooled_rhs)))


def posterior(prior, observations):
    if not observations:
        return prior
    x = np.stack([item[0] for item in observations])
    y = np.array([item[1] for item in observations])
    return np.linalg.solve(np.eye(D)+x.T@x, prior+x.T@y)


def verify_decisions(data, result):
    weights = result['w']
    for g in range(data['x'].shape[0]):
        for condition in range(2):
            records = [[], [], []]
            for t in range(33):
                if t in BUDGETS:
                    k = BUDGETS.index(t)
                    x = data['x'][g, condition, t]
                    order = result['order'][g,t]
                    if sorted(order.tolist()) != [0, 1, 2]:
                        raise AssertionError('action order is not a permutation')
                    for m in range(5):
                        prior = weights[m%2] if m<4 else np.zeros(D)
                        predicted = np.array([
                            x[a] @ (posterior(prior,records[a]) if m in (0,1,4) else prior)
                            for a in range(3)
                        ])
                        # Sequential and batch least-squares may differ at machine precision.
                        saved = int(result['decision'][g,condition,m,k])
                        if predicted[saved] < predicted.max()-1e-9:
                            raise AssertionError(f'policy action g={g} cond={condition} m={m} t={t}')
                        check('selected value',result['value'][g,condition,m,k],data['truth'][g,condition,t,saved])
                if t<32:
                    action=int(data['a'][g,t])
                    records[action].append((data['x'][g,condition,t,action], data['y'][g,condition,t]))



def public_features(obs):
    result = np.zeros((3,D))
    result[:,0] = 1
    for j, state in enumerate(obs.public_state[:2]):
        if state is not None:
            result[:,j+1] = state/2
    result[:,3] = obs.previous_own_reward
    result[:,4] = obs.episode_start_flag
    for lag, (own,other) in enumerate(obs.action_history[:5]):
        for action in range(3):
            result[action,5+lag*3] = (own == action)
            result[action,6+lag*3] = (other == action)
            result[action,7+lag*3] = 1
    return result


def distribution(q, obs):
    q_row = q[obs.previous_own_action,obs.previous_opponent_action]
    best = np.abs(q_row-q_row.max()) <= 1e-12
    return np.full(3,.25/3) + .75*best/best.sum()


def update(q, obs, action, reward, nxt, terminal):
    loc = obs.previous_own_action, obs.previous_opponent_action, action
    future = 0 if terminal else q[nxt.previous_own_action,nxt.previous_opponent_action].max()
    delta = .3*(reward+.8*future-q[loc])
    q[loc] += delta
    return abs(delta)


class Gate:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def replay(data, g):
    # Independently evaluate real simulator actions and an independently updated Q table.
    from mamrl.structured_games import StructuredEnv, generate_structured_rule
    families = ('resource','sequence','reciprocity','mixed')
    rule = generate_structured_rule(n=3, memory_depth=3, seed=int(data['seed'][g]),
        family=families[int(data['family'][g])], state_levels=(3,3,3), stochastic=True)
    if rule.fingerprint() != data['fingerprint'][g]:
        raise AssertionError('game fingerprint differs')
    env = StructuredEnv(rule,max_steps=32)
    obs = env.reset(seed=int(data['warm_seed'][g]))
    q = np.zeros((3,3,3))
    for t in range(32):
        prob = distribution(q,obs[1])
        action = int(data['warm_a'][g,t])
        other = min(int(np.searchsorted(np.cumsum(prob),data['warm_u'][g,t],side='right')),2)
        check('warm opponent action',other,data['warm_b'][g,t],0)
        next_step = env.step(action,other)
        check('warm opponent reward',next_step.rewards[1],data['warm_r'][g,t])
        update(q,obs[1],other,next_step.rewards[1],next_step.observations[1],next_step.truncated)
        obs = next_step.observations
    check('warmed Q',q,data['warm_q'][g])
    for condition in range(2):
        env = StructuredEnv(rule,max_steps=33)
        obs = env.reset(seed=int(data['reset_seed'][g]))
        opponent_q = q.copy()
        for t in range(33):
            prob = distribution(opponent_q,obs[1])
            check('public feature',public_features(obs[0]),data['x'][g,condition,t])
            check('opponent distribution',prob,data['opp_p'][g,condition,t])
            # Repeated real simulator step calls independently validate reward indexing.
            truth = np.array([sum(prob[b]*copy.deepcopy(env).step(a,b).rewards[0]
                                  for b in range(3)) for a in range(3)])
            check('counterfactual reward',truth,data['truth'][g,condition,t])
            if t==32:
                break
            a=int(data['a'][g,t])
            b=min(int(np.searchsorted(np.cumsum(prob),data['u'][g,t],side='right')),2)
            check('opponent sampled action',b,data['b'][g,condition,t],0)
            env._rng = Gate(float(data['gate_u'][g,t]))
            result=env.step(a,b)
            check('joint rewards',result.rewards,[data['y'][g,condition,t],data['opp_y'][g,condition,t]])
            delta=update(opponent_q,obs[1],b,result.rewards[1],result.observations[1],result.truncated) if condition else 0.
            check('opponent update',delta,data['q_delta'][g,condition,t])
            obs=result.observations


def replay_deployment(data, result, g):
    from mamrl.structured_games import StructuredEnv, generate_structured_rule
    families = ('resource','sequence','reciprocity','mixed')
    rule = generate_structured_rule(n=3,memory_depth=3,seed=int(data['seed'][g]),
        family=families[int(data['family'][g])],state_levels=(3,3,3),stochastic=True)
    for condition in range(2):
        for method,prior in enumerate((result['w'][0],result['w'][1],np.zeros(D))):
            env=StructuredEnv(rule,max_steps=24)
            obs=env.reset(seed=int(data['reset_seed'][g]))
            q=data['warm_q'][g].copy()
            records=[[],[],[]]
            for t in range(24):
                x=public_features(obs[0])
                scores=np.array([x[a]@posterior(prior,records[a]) for a in range(3)])
                a=int(result['deploy_actions'][g,condition,method,t,0])
                if scores[a] < scores.max()-1e-9:
                    raise AssertionError('deployment focal decision differs')
                prob=distribution(q,obs[1])
                b=min(int(np.searchsorted(np.cumsum(prob),result['deploy_u'][g,t],side='right')),2)
                check('deployment opponent action',b,result['deploy_actions'][g,condition,method,t,1],0)
                env._rng=Gate(float(result['deploy_gate'][g,t]))
                transition=env.step(a,b)
                check('deployment joint rewards',transition.rewards,result['deploy_rewards'][g,condition,method,t])
                delta=update(q,obs[1],b,transition.rewards[1],transition.observations[1],transition.truncated) if condition else 0.
                check('deployment q update',delta,result['deploy_q_delta'][g,condition,method,t])
                records[a].append((x[a],transition.rewards[0]))
                obs=transition.observations

def effects(data):
    value = data['value']
    first = np.mean(value[:,1,0,3]-value[:,1,1,3])
    gains = (value[:,:,0,3]-value[:,:,2,3])-(value[:,:,1,3]-value[:,:,3,3])
    second = np.mean(gains[:,1]-gains[:,0])
    return np.array([first,second])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    root=args.output
    manifest=json.loads((root/'manifest.json').read_text())
    blocks=sorted(root.glob('block_[0-9][0-9][0-9]'))
    if len(blocks) != manifest['config']['blocks']:
        raise AssertionError('block count differs from manifest')
    for name, expected in manifest['source'].items():
        file=root/'source'/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise AssertionError(f'archived source hash differs: {name}')
    print('blocks',len(blocks),flush=True)
    contrasts=[]
    seen=set()
    replays=0
    for folder in blocks:
        rec=json.loads((folder/'record.json').read_text())
        for name, expected in rec['sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
                raise AssertionError(f'raw data hash differs: {folder/name}')
        with np.load(folder/'train.npz',allow_pickle=False) as raw:
            train=dict(raw)
        with np.load(folder/'test.npz',allow_pickle=False) as raw:
            test=dict(raw)
        with np.load(folder/'results.npz',allow_pickle=False) as raw:
            result=dict(raw)
        for label, data in [('train',train),('test',test)]:
            fingerprints=set(data['fingerprint'].tolist())
            if len(fingerprints) != len(data['fingerprint']) or fingerprints.intersection(seen):
                raise AssertionError(f'{folder}/{label}: duplicated serialized game')
            seen.update(fingerprints)
            check('initial public observations',data['x'][:,0,0],data['x'][:,1,0],0)
            check('initial counterpart action probabilities',data['opp_p'][:,0,0],data['opp_p'][:,1,0],0)
            check('initial focal expected rewards',data['truth'][:,0,0],data['truth'][:,1,0],0)
            check('fixed counterpart updates',data['q_delta'][:,0],0*data['q_delta'][:,0],0)
            for family in range(4):
                indices=np.flatnonzero(data['family']==family)
                if len(indices) == 0:
                    raise AssertionError('family missing')
                replay(data,int(indices[0]))
                replays+=1
        for family in range(4):
            replay_deployment(test,result,int(np.flatnonzero(test['family']==family)[0]))
        check('fitted coefficients', independent_fit(train),result['w'])
        verify_decisions(test,result)
        contrasts.append(effects(result))
    contrasts=np.array(contrasts)
    rng=np.random.default_rng(384916)
    samples=rng.integers(len(blocks), size=(20000,len(blocks)))
    bounds=np.quantile(contrasts[samples].mean(1),[.0125,.9875],axis=0)
    summary=json.loads((root/'summary.json').read_text())
    for j,row in enumerate(summary['primary']):
        check('analysis mean',row['estimate'],contrasts[:,j].mean())
        check('analysis interval',(row['lower'],row['upper']),bounds[:,j])
    report=dict(source_and_block_hashes_match=True, analysis_estimates_and_bounds_match=True, blocks=len(blocks), independent_estimates=contrasts.mean(0).tolist(),
                bootstrap_intervals=bounds.T.tolist(), exact_fingerprints=len(seen), independently_replayed=replays)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
