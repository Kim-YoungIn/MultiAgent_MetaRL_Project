"""Cross-size structured-game empirical-Bayes experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from mamrl.structured_games import StructuredEnv, generate_structured_rule
from research.structured_oracle.run import RewardKernel

FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
SETTINGS = ((3, 1), (3, 3), (6, 5))
METHODS = ('meta_adapt', 'pooled_adapt', 'meta_frozen', 'scratch', 'no_state', 'no_long_history')
BUDGETS = (0, 4, 8, 16)
D = 20


def features(obs, n):
    f = np.zeros((n, D))
    f[:, 0] = 1
    for j, state in enumerate(obs.public_state[:2]):
        if state is not None:
            f[:, 1+j] = state / 2
    f[:, 3] = obs.previous_own_reward
    f[:, 4] = obs.episode_start_flag
    actions = np.arange(n)
    for i, (own, other) in enumerate(obs.action_history[:5]):
        f[:, 5+3*i] = actions == own
        f[:, 6+3*i] = actions == other
        f[:, 7+3*i] = 1
    return f


class Opponent:
    def __init__(self, n):
        self.n = n
        self.q = np.zeros((n, n, n))
        self.change = 0.

    def distribution(self, obs):
        row = self.q[obs.previous_own_action, obs.previous_opponent_action]
        best = np.isclose(row, row.max(), atol=1e-12, rtol=0)
        return .25/self.n + .75*best/best.sum()

    def update(self, obs, action, reward, nextobs, terminal):
        key = obs.previous_own_action, obs.previous_opponent_action, action
        future = 0 if terminal else self.q[nextobs.previous_own_action, nextobs.previous_opponent_action].max()
        delta = .3*(reward + .8*future - self.q[key])
        self.q[key] += delta
        self.change += abs(delta)


def step(env, opponent, obs, action, uniform):
    probs = opponent.distribution(obs[1])
    b = min(int(np.searchsorted(np.cumsum(probs), uniform, side='right')), env.n-1)
    result = env.step(int(action), b)
    opponent.update(obs[1], b, result.rewards[1], result.observations[1], result.truncated)
    return result, b


def game(seed, family, n, depth):
    return generate_structured_rule(n=n, memory_depth=depth, seed=int(seed),
                                    family=family, state_levels=(3, 3, 3), stochastic=True)


def collect(g, seed, rng):
    env = StructuredEnv(g, max_steps=32)
    obs = env.reset(seed=int(seed))
    opponent = Opponent(g.n)
    rows = []
    for _ in range(32):
        x = features(obs[0], g.n)
        a = int(rng.integers(g.n))
        u = float(rng.random())
        result, b = step(env, opponent, obs, a, u)
        rows.append([x[a].tolist(), a, b, *result.rewards, u])
        obs = result.observations
    return rows


def fit(tasks):
    gram = np.eye(D)*.01
    rhs = np.zeros(D)
    pooled_gram = gram.copy()
    pooled_rhs = rhs.copy()
    for rows in tasks:
        for a in range(3):
            selected = [row for row in rows if row[1] == a]
            if not selected:
                continue
            x = np.array([row[0] for row in selected])
            y = np.array([row[3] for row in selected])
            s, v = x.T@x, x.T@y
            pooled_gram += s
            pooled_rhs += v
            gram += np.linalg.solve(np.eye(D)+s, s)
            rhs += np.linalg.solve(np.eye(D)+s, v)
    return np.linalg.solve(gram, rhs), np.linalg.solve(pooled_gram, pooled_rhs)


class Model:
    def __init__(self, n, mean, frozen=False, mask=None):
        self.mean = np.repeat(np.asarray(mean)[None, :], n, axis=0)
        self.inverse = np.repeat(np.eye(D)[None, :, :], n, axis=0)
        self.frozen = frozen
        self.mask = np.ones(D) if mask is None else mask

    def act(self, x, order):
        prediction = np.sum(self.mean * (x*self.mask), axis=1)
        return int(order[np.argmax(prediction[order])])

    def observe(self, x, a, reward):
        if self.frozen:
            return
        x = x[a]*self.mask
        u = self.inverse[a]@x
        den = 1 + x@u
        self.mean[a] += u*((reward-x@self.mean[a])/den)
        self.inverse[a] -= np.outer(u,u)/den


def models(n, meta, pooled):
    no_state = np.ones(D); no_state[1:3] = 0
    no_history = np.ones(D); no_history[8:] = 0
    return [Model(n,meta), Model(n,pooled), Model(n,meta,frozen=True),
            Model(n,np.zeros(D)), Model(n,meta,mask=no_state), Model(n,meta,mask=no_history)]


def run_test(g, seed, rng, meta, pooled):
    n = g.n
    kernel = RewardKernel(g)
    env = StructuredEnv(g, max_steps=17)
    obs = env.reset(seed=seed)
    opp = Opponent(n)
    model = models(n,meta,pooled)
    values = np.zeros((len(METHODS),len(BUDGETS)))
    chosen = np.zeros_like(values,dtype=int)
    best = np.zeros((len(BUDGETS),n))
    records = []
    checkpoints = []
    random_actions = rng.integers(n,size=16)
    uniforms = rng.random(16)
    orders = np.array([rng.permutation(n) for _ in BUDGETS])
    for t in range(17):
        x = features(obs[0],n)
        if t in BUDGETS:
            k = BUDGETS.index(t)
            probs = opp.distribution(obs[1])
            truth = kernel(env)@probs
            best[k] = truth
            checkpoints.append({'public_state':list(obs[0].public_state),
                                'history':list(obs[0].action_history),
                                'opponent_probs':probs.tolist(),
                                'order':orders[k].tolist()})
            for m, learner in enumerate(model):
                a = learner.act(x,orders[k])
                chosen[m,k] = a
                values[m,k] = truth[a]
        if t==16: break
        a = int(random_actions[t])
        result,b = step(env,opp,obs,a,float(uniforms[t]))
        for learner in model: learner.observe(x,a,result.rewards[0])
        records.append([x[a].tolist(),a,b,*result.rewards,float(uniforms[t])])
        obs = result.observations
    probe_change=opp.change
    uniforms = rng.random(24)
    orders = np.array([rng.permutation(n) for _ in range(24)])
    acts = np.zeros((len(METHODS),24,2),dtype=int)
    rewards = np.zeros((len(METHODS),24,2))
    states = np.zeros((len(METHODS),24,2),dtype=int)
    changes = np.zeros(len(METHODS))
    for m, learner in enumerate(models(n,meta,pooled)):
        env = StructuredEnv(g,max_steps=24)
        obs = env.reset(seed=seed+1)
        opp = Opponent(n)
        for t in range(24):
            x = features(obs[0],n)
            a = learner.act(x,orders[t])
            result,b = step(env,opp,obs,a,float(uniforms[t]))
            learner.observe(x,a,result.rewards[0])
            acts[m,t] = a,b
            rewards[m,t] = result.rewards
            state = result.observations[0].public_state[:2]
            states[m,t,:len(state)] = state
            obs = result.observations
        changes[m] = opp.change
    return dict(values=values,chosen=chosen,truth=best,probe=records,checkpoints=checkpoints,
                probe_change=probe_change,acts=acts,rewards=rewards,states=states,changes=changes,
                uniforms=uniforms,orders=orders)


def write_json(path,data):
    path.write_text(json.dumps(data,sort_keys=True,separators=(',', ':'),allow_nan=False))


def run_block(index, master, folder):
    started = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([master,index,1]))
    seeds = rng.integers(0,2**53,size=56,dtype=np.int64)
    seen = set()
    tasks=[]; train=[]
    for i in range(32):
        seed=int(seeds[i]); g=game(seed,FAMILIES[i%4],3,3)
        assert g.fingerprint() not in seen
        seen.add(g.fingerprint())
        random=np.random.default_rng(np.random.SeedSequence([master,index,2,i]))
        rows=collect(g,seed+1,random)
        tasks.append(rows)
        train.append(dict(seed=seed,game=g.to_dict(),fingerprint=g.fingerprint(),rows=rows))
    meta,pooled=fit(tasks)
    val=[]; chosen=[]; truth=[]; acts=[]; rewards=[]; states=[]; changes=[]
    probe_changes=[]; tests=[]
    for setting,(n,depth) in enumerate(SETTINGS):
        sv=[];sc=[];st=[];sa=[];sr=[];ss=[];sq=[];sp=[]; games=[]
        for i in range(24):
            seed=int(seeds[32+i]); g=game(seed,FAMILIES[i%4],n,depth)
            assert g.fingerprint() not in seen
            seen.add(g.fingerprint())
            random=np.random.default_rng(np.random.SeedSequence([master,index,3,setting,i]))
            result=run_test(g,seed+1,random,meta,pooled)
            games.append(dict(seed=seed,game=g.to_dict(),fingerprint=g.fingerprint(),
                              probe=result['probe'],checkpoints=result['checkpoints'],
                              rollout_uniforms=result['uniforms'].tolist(),
                              rollout_orders=result['orders'].tolist()))
            sv.append(result['values']);sc.append(result['chosen']);st.append(result['truth'])
            sa.append(result['acts']);sr.append(result['rewards']);ss.append(result['states'])
            sq.append(result['changes']);sp.append(result['probe_change'])
        val.append(sv);chosen.append(sc);truth.append(st);acts.append(sa)
        rewards.append(sr);states.append(ss);changes.append(sq);probe_changes.append(sp);tests.append(games)
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'train.json',train)
    write_json(folder/'test.json',tests)
    # Different action counts require separate arrays for query action values.
    np.savez_compressed(folder/'results.npz', values=val, chosen=chosen, acts=acts,
                        rewards=rewards,states=states,changes=changes,probe_changes=probe_changes,
                        truth_0=truth[0],truth_1=truth[1],truth_2=truth[2],meta=meta,pooled=pooled)
    write_json(folder/'complete.json',dict(index=index,seconds=time.monotonic()-started,
                                           fingerprints=len(seen)))


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--blocks',type=int,default=16)
    parser.add_argument('--seed',type=int,default=91827361)
    args=parser.parse_args()
    if args.blocks<1: parser.error('blocks must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),
           Path('research/structured_oracle/run.py'),Path('05_toy_models/mamrl/structured_games.py'),
           Path('05_toy_models/mamrl/rules.py'),Path('05_toy_models/mamrl/tensor.py')]
    manifest=dict(blocks=args.blocks,seed=args.seed,methods=METHODS,settings=SETTINGS,budgets=BUDGETS,
                  python=sys.version,numpy=np.__version__,platform=platform.platform(),
                  sources={str(p):sha(p) for p in files})
    file=args.output/'manifest.json'
    if file.exists():
        if json.loads(file.read_text())!=json.loads(json.dumps(manifest)):
            raise ValueError('resume configuration or source differs')
    else: write_json(file,manifest)
    all_seen=set()
    for i in range(args.blocks):
        folder=args.output/f'block_{i:03d}'
        if (folder/'complete.json').exists(): print('reusing',i,flush=True)
        elif folder.exists(): raise RuntimeError(f'incomplete block retained at {folder}')
        else:
            print('starting',i,flush=True)
            run_block(i,args.seed,folder)
        train=json.loads((folder/'train.json').read_text())
        test=json.loads((folder/'test.json').read_text())
        for record in train+[r for setting in test for r in setting]:
            if record['fingerprint'] in all_seen: raise ValueError('duplicate game across blocks')
            all_seen.add(record['fingerprint'])
    print('complete',flush=True)


if __name__=='__main__': main()
