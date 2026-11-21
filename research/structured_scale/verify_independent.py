"""Reconstruct the cross-size experiment from saved trajectories and game rules."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'05_toy_models'))
from mamrl.structured_games import StructuredEnv, StructuredGame


def check(name,actual,expected,tol=1e-8):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=tol,atol=tol):
        e=float(np.max(np.abs(a-b))) if a.shape==b.shape else None
        raise AssertionError(f'{name}: shapes {a.shape} and {b.shape}, max difference {e}')


def features(obs,n):
    x=np.zeros((n,20))
    x[:,0]=1
    for i,v in enumerate(obs.public_state[:2]):
        if v is not None: x[:,i+1]=v/2
    x[:,3]=obs.previous_own_reward
    x[:,4]=obs.episode_start_flag
    for i,(a,b) in enumerate(obs.action_history[:5]):
        for candidate in range(n):
            x[candidate,5+3*i]=candidate==a
            x[candidate,6+3*i]=candidate==b
            x[candidate,7+3*i]=1
    return x


def fit(train):
    A=np.eye(20)*.01
    B=np.zeros(20)
    C=A.copy()
    D=B.copy()
    for task in train:
        for a in range(3):
            rows=[r for r in task['rows'] if r[1]==a]
            if not rows: continue
            x=np.array([r[0] for r in rows]); y=np.array([r[3] for r in rows])
            z=np.linalg.solve(np.eye(len(rows))+x@x.T,x)
            A += x.T@z
            B += z.T@y
            C += x.T@x
            D += x.T@y
    return np.linalg.solve(A,B),np.linalg.solve(C,D)


def policy(obs,n,order,prior,rows,method):
    x=features(obs,n)
    mask=np.ones(20)
    if method==4: mask[1:3]=0
    if method==5: mask[8:]=0
    x*=mask
    values=[]
    for a in range(n):
        sample=[(r[0]*mask,r[2]) for r in rows if r[1]==a]
        if method==2 or not sample:
            w=prior
        else:
            xa=np.stack([r[0] for r in sample]); y=np.array([r[1] for r in sample])
            w=np.linalg.solve(np.eye(20)+xa.T@xa,prior+xa.T@y)
        values.append(x[a]@w)
    return int(order[np.argmax(np.array(values)[order])])


def probabilities(q,obs,n):
    row=q[obs.previous_own_action,obs.previous_opponent_action]
    best=np.abs(row-row.max())<=1e-12
    return np.ones(n)*(.25/n)+.75*best/best.sum()


def advance(q,obs,b,reward,next_obs,terminal):
    key=(obs.previous_own_action,obs.previous_opponent_action,b)
    nxt=(next_obs.previous_own_action,next_obs.previous_opponent_action)
    delta=.3*(reward+(0 if terminal else .8*q[nxt].max())-q[key])
    q[key]+=delta
    return abs(delta)


def replay(test,saved,setting,index):
    game=StructuredGame.from_dict(test['game'])
    if game.fingerprint()!=test['fingerprint']: raise AssertionError('game fingerprint differs')
    n=game.n
    meta,pooled=saved['meta'],saved['pooled']
    priors=[meta,pooled,meta,np.zeros(20),meta,meta]
    env=StructuredEnv(game,max_steps=17)
    obs=env.reset(seed=test['seed']+1)
    q=np.zeros((n,n,n)); rows=[]; change=0.
    for t in range(17):
        x=features(obs[0],n)
        if t in (0,4,8,16):
            k=(0,4,8,16).index(t)
            checkpoint=test['checkpoints'][k]
            probs=probabilities(q,obs[1],n)
            check('probe opponent probabilities',probs,checkpoint['opponent_probs'])
            check('probe public state',obs[0].public_state[:2],checkpoint['public_state'][:2],0)
            check('probe action history',obs[0].action_history,checkpoint['history'],0)
            # A fresh real-environment clone evaluates each simultaneous action pair.
            payoff=np.empty(n)
            for a in range(n):
                payoff[a]=sum(probs[b]*copy.deepcopy(env).step(a,b).rewards[0] for b in range(n))
            check('query reward vector',payoff,saved[f'truth_{setting}'][index,k])
            for method in range(6):
                a=policy(obs[0],n,checkpoint['order'],priors[method],rows,method)
                check('query focal action',a,saved['chosen'][setting,index,method,k],0)
                check('query selected reward',payoff[a],saved['values'][setting,index,method,k])
        if t==16: break
        recorded,a,b,r0,r1,u=test['probe'][t]
        check('probe observed features',x[a],recorded)
        probs=probabilities(q,obs[1],n)
        check('probe opponent action',b,min(np.searchsorted(np.cumsum(probs),u,side='right'),n-1),0)
        result=env.step(a,b)
        check('probe rewards',result.rewards,(r0,r1))
        change+=advance(q,obs[1],b,r1,result.observations[1],result.truncated)
        rows.append((x[a].copy(),a,r0)); obs=result.observations
    check('probe Q changes',change,saved['probe_changes'][setting,index])
    for method in range(6):
        env=StructuredEnv(game,max_steps=24)
        obs=env.reset(seed=test['seed']+2)
        q=np.zeros((n,n,n)); rows=[]; change=0.
        for t in range(24):
            x=features(obs[0],n)
            a=policy(obs[0],n,test['rollout_orders'][t],priors[method],rows,method)
            probs=probabilities(q,obs[1],n)
            b=int(min(np.searchsorted(np.cumsum(probs),test['rollout_uniforms'][t],side='right'),n-1))
            check('online actions',(a,b),saved['acts'][setting,index,method,t],0)
            result=env.step(a,b)
            check('online rewards',result.rewards,saved['rewards'][setting,index,method,t])
            state=np.zeros(2,dtype=int); public=result.observations[0].public_state[:2]
            state[:len(public)]=public
            check('online visible state',state,saved['states'][setting,index,method,t],0)
            change+=advance(q,obs[1],b,result.rewards[1],result.observations[1],result.truncated)
            rows.append((x[a].copy(),a,result.rewards[0])); obs=result.observations
        check('online opponent Q change',change,saved['changes'][setting,index,method])


def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); args=p.parse_args()
    root=args.output
    manifest=json.loads((root/'manifest.json').read_text())
    mismatches=[name for name,h in manifest['sources'].items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h]
    effects=[]; seen=set(); replayed=0
    for block in range(manifest['blocks']):
        folder=root/f'block_{block:03d}'
        train=json.loads((folder/'train.json').read_text())
        tests=json.loads((folder/'test.json').read_text())
        with np.load(folder/'results.npz',allow_pickle=False) as f: data=dict(f)
        for record in train+[r for setting in tests for r in setting]:
            encoded=json.dumps(record['game'],sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest()!=record['fingerprint']:
                raise AssertionError('serialized game fingerprint mismatch')
            if record['fingerprint'] in seen: raise AssertionError('duplicate across panels')
            seen.add(record['fingerprint'])
        if len(train)!=32 or any(len(item['rows'])!=32 for item in train): raise AssertionError('training budget')
        for s in range(3):
            for i in range(24):
                for m in range(6):
                    for k in range(4):
                        action=data['chosen'][s,i,m,k]
                        check('selected value',data['values'][s,i,m,k],data[f'truth_{s}'][i,k,action])
        effects.append(np.mean(data['values'][2,:,0,2]-data['values'][2,:,1,2]))
        if block==0:
            m,po=fit(train)
            check('meta fit',m,data['meta']); check('pooled fit',po,data['pooled'])
            for s in range(3):
                for i in range(4):
                    replay(tests[s][i],data,s,i); replayed+=1
    effects=np.array(effects)
    rng=np.random.default_rng(248291)
    indices=rng.integers(0,len(effects),(10000,len(effects)))
    interval=np.quantile(effects[indices].mean(axis=1),[.025,.975])
    result={'blocks':len(effects),'primary_estimate':float(effects.mean()),'bootstrap_95_interval':interval.tolist(),
            'replayed_games':replayed,'source_hash_mismatches':mismatches,
            'scope':'All selected-value indexing and fingerprints checked. One game per setting and motif replayed against real environment for all six query and deployment policies. Block 0 fitted priors reconstructed.'}
    analysis=root/'analysis.json'
    if analysis.exists():
        primary=json.loads(analysis.read_text())['primary']
        check('reported primary estimate',primary['estimate'],result['primary_estimate'])
        check('reported primary interval',primary['interval_95'],result['bootstrap_95_interval'])
        result['analysis_primary_matches']=True
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
