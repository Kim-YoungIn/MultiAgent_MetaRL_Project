"""Independent raw-data replay using closed-form updates and simulator payoff queries."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from mamrl.structured_games import StructuredEnv,StructuredGame

D=20


def phi(obs,n):
    rows=[]
    for a in range(n):
        state=list(obs.public_state[:2])
        state += [0]*(2-len(state))
        row=[1,state[0]/2,state[1]/2,obs.previous_own_reward,int(obs.episode_start_flag)]
        for lag in range(5):
            if lag<len(obs.action_history):
                x,y=obs.action_history[lag]
                row += [int(a==x),int(a==y),1]
            else: row += [0,0,0]
        rows.append(row)
    return np.array(rows,dtype=float)


def policy(q,obs,n):
    v=q[obs.previous_own_action,obs.previous_opponent_action]
    best=np.isclose(v,v.max(),rtol=0,atol=1e-12)
    return .25/n+.75*best/best.sum()


def update(q,obs,b,reward,next_obs,terminal):
    key=obs.previous_own_action,obs.previous_opponent_action,b
    future=0 if terminal else q[next_obs.previous_own_action,next_obs.previous_opponent_action].max()
    delta=.3*(reward+.8*future-q[key]);q[key]+=delta
    return abs(delta)


def close(a,b,where):
    if not np.allclose(a,b,atol=1e-10,rtol=1e-9):
        raise AssertionError((where,a,b))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    out=parser.parse_args().output
    manifest=json.loads((out/'manifest.json').read_text())
    for path,digest in manifest['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    counts=dict(training=0,probe=0,queries=0,deployment=0)
    fingerprints=set(); diffs=[]
    for block in range(manifest['blocks']):
        directory=out/f'block_{block:03d}'
        train=json.loads((directory/'train.json').read_text())
        tests=json.loads((directory/'test.json').read_text())
        data=np.load(directory/'results.npz')
        m=np.eye(D)*.01;v=np.zeros(D);pm=m.copy();pv=v.copy()
        for record in train:
            g=StructuredGame.from_dict(record['game'])
            assert g.fingerprint()==record['fingerprint'] and record['fingerprint'] not in fingerprints
            fingerprints.add(record['fingerprint'])
            env=StructuredEnv(g,max_steps=32);obs=env.reset(seed=record['seed']+1)
            q=np.zeros((3,3,3))
            for row in record['rows']:
                x,a,b,y,z,u=row
                close(phi(obs[0],3)[a],x,'training features')
                check=min(int(np.searchsorted(np.cumsum(policy(q,obs[1],3)),u,side='right')),2)
                assert b==check
                result=env.step(a,b);close(result.rewards,(y,z),'training rewards')
                update(q,obs[1],b,z,result.observations[1],result.truncated)
                obs=result.observations;counts['training']+=1
            for a in range(3):
                selected=[r for r in record['rows'] if r[1]==a]
                if not selected: continue
                x=np.array([r[0] for r in selected]);y=np.array([r[3] for r in selected])
                cov=np.eye(len(y))+x@x.T
                m+=x.T@np.linalg.solve(cov,x);v+=x.T@np.linalg.solve(cov,y)
                pm+=x.T@x;pv+=x.T@y
        close(np.linalg.solve(m,v),data['meta'],'meta fitted prior')
        close(np.linalg.solve(pm,pv),data['pooled'],'pooled prior')
        for setting,records in enumerate(tests):
            for i,record in enumerate(records):
                g=StructuredGame.from_dict(record['game'])
                assert g.fingerprint()==record['fingerprint'] and record['fingerprint'] not in fingerprints
                fingerprints.add(record['fingerprint'])
                if i>=4: continue
                n=g.n
                masks=np.ones((6,D));masks[4,1:3]=0;masks[5,8:]=0
                prior=[data['meta'],data['pooled'],data['meta'],np.zeros(D),data['meta'],data['meta']]
                env=StructuredEnv(g,max_steps=17);obs=env.reset(seed=record['seed']+1)
                q=np.zeros((n,n,n));xs=[[] for _ in range(n)];ys=[[] for _ in range(n)]
                total=0
                for t in range(17):
                    x=phi(obs[0],n)
                    if t in (0,4,8,16):
                        k=(0,4,8,16).index(t);prob=policy(q,obs[1],n)
                        cp=record['checkpoints'][k]
                        close(prob,cp['opponent_probs'],'probe opponent probability')
                        assert cp['public_state']==list(obs[0].public_state)
                        true=[sum(prob[b]*copy.deepcopy(env).step(a,b).rewards[0] for b in range(n)) for a in range(n)]
                        close(true,data[f'truth_{setting}'][i,k],'probe values')
                        for method in range(6):
                            predictions=[]
                            for a in range(n):
                                if method==2 or not xs[a]: w=prior[method]
                                else:
                                    z=np.asarray(xs[a])*masks[method];y=np.asarray(ys[a])
                                    w=np.linalg.solve(np.eye(D)+z.T@z,prior[method]+z.T@y)
                                predictions.append(w@(x[a]*masks[method]))
                            a=int(data['chosen'][setting,i,method,k])
                            assert max(predictions)-predictions[a]<1e-12
                            close(true[a],data['values'][setting,i,method,k],'selected value')
                            counts['queries']+=1
                    if t==16: break
                    row=record['probe'][t];_,a,b,y,z,u=row
                    close(x[a],row[0],'probe features')
                    assert min(int(np.searchsorted(np.cumsum(policy(q,obs[1],n)),u,side='right')),n-1)==b
                    result=env.step(a,b);close(result.rewards,(y,z),'probe rewards')
                    total+=update(q,obs[1],b,z,result.observations[1],result.truncated)
                    xs[a].append(x[a]);ys[a].append(y);obs=result.observations;counts['probe']+=1
                close(total,data['probe_changes'][setting,i],'opponent probe changes')
                for method in range(6):
                    env=StructuredEnv(g,max_steps=24);obs=env.reset(seed=record['seed']+2)
                    q=np.zeros((n,n,n));xs=[[] for _ in range(n)];ys=[[] for _ in range(n)];total=0
                    for t in range(24):
                        x=phi(obs[0],n);predictions=[]
                        for a in range(n):
                            if method==2 or not xs[a]:w=prior[method]
                            else:
                                z=np.asarray(xs[a])*masks[method];y=np.asarray(ys[a])
                                w=np.linalg.solve(np.eye(D)+z.T@z,prior[method]+z.T@y)
                            predictions.append(w@(x[a]*masks[method]))
                        a,b=data['acts'][setting,i,method,t]
                        assert max(predictions)-predictions[a]<1e-12
                        u=record['rollout_uniforms'][t]
                        assert min(int(np.searchsorted(np.cumsum(policy(q,obs[1],n)),u,side='right')),n-1)==b
                        result=env.step(int(a),int(b))
                        close(result.rewards,data['rewards'][setting,i,method,t],'deployment rewards')
                        public=list(result.observations[0].public_state[:2]);public += [0]*(2-len(public))
                        close(public,data['states'][setting,i,method,t],'deployment state')
                        total+=update(q,obs[1],b,result.rewards[1],result.observations[1],result.truncated)
                        xs[a].append(x[a]);ys[a].append(result.rewards[0]);obs=result.observations;counts['deployment']+=1
                    close(total,data['changes'][setting,i,method],'opponent deployment change')
        values=data['values']
        diffs.append((values[2,:,0,2]-values[2,:,1,2]).mean())
    rng=np.random.default_rng(248291);indices=rng.integers(0,len(diffs),(10000,len(diffs)))
    diffs=np.array(diffs);interval=np.quantile(diffs[indices].mean(axis=1),[.025,.975])
    report=json.loads((out/'analysis.json').read_text())
    close(diffs.mean(),report['primary']['estimate'],'primary mean')
    close(interval,report['primary']['interval_95'],'primary interval')
    result=dict(status='passed',counts=counts,exact_unique_fingerprints=len(fingerprints),
                primary_mean=float(diffs.mean()),primary_interval=interval.tolist(),
                scope='All training and fingerprints replayed. Four test games per block per setting replayed, one per family. Action ties use 1e-12 tolerance.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
