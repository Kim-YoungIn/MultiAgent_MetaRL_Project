"""Fit and assess initializations under identical online adaptation rules."""
import numpy as np
from mamrl.structured_games import StructuredEnv
from research.structured_oracle.run import Gate
from research.structured_scale.run import D, FAMILIES, Model, Opponent, features, game

BUDGETS = (0, 2, 4, 8, 16, 32)
METHODS = ('meta_adapt', 'pool_adapt', 'meta_frozen', 'pool_frozen', 'scratch')


def histories(data):
    """Yield pre-observation intercepts and slopes of predictions as functions of w."""
    games = len(data['a'])
    trajectories = 2*games
    features_by_time = data['x'].reshape(trajectories, 33, 3, D)
    rewards = data['y'].reshape(trajectories, 32)
    actions = np.repeat(data['a'], 2, axis=0)
    cov = np.tile(np.eye(D), (trajectories, 3, 1, 1))
    # Posterior mean for a zero initialization; posterior for w is offset + cov @ w.
    mean = np.zeros((trajectories, 3, D))
    i = np.arange(trajectories)
    for t in range(33):
        x = features_by_time[:, t]
        z = np.einsum('nadf,naf->nad', cov, x)
        offset = np.sum(x*mean, axis=-1)
        yield x, z, offset
        if t < 32:
            action = actions[:, t]
            gain = z[i, action]
            denom = 1+np.sum(x[i, action]*gain, axis=1)
            mean[i, action] += gain*((rewards[:, t]-offset[i, action])/denom)[:, None]
            cov[i, action] -= np.einsum('nd,nf->ndf', gain/denom[:, None], gain)


def fit(data):
    n = len(data['a'])*2
    i = np.arange(n)
    features_used = np.repeat(data['a'], 2, axis=0)
    observed = data['y'].reshape(n, 32)
    gram = np.tile(.01*np.eye(D), (2,1,1))
    rhs = np.zeros((2,D))
    for t,(x,z,offset) in enumerate(histories(data)):
        if t == 32:
            break
        a = features_used[:, t]
        for j,(design,target) in enumerate([(z[i,a],observed[:,t]-offset[i,a]),(x[i,a],observed[:,t])]):
            gram[j] += design.T @ design
            rhs[j] += design.T @ target
    return np.linalg.solve(gram, rhs[...,None])[...,0]


def evaluate(data, w, rng):
    games = len(data['a'])
    order = rng.random((games,33,3)).argsort(axis=-1)
    value = np.empty((games,2,5,len(BUDGETS)))
    decision = np.empty_like(value, dtype=np.int8)
    for t,(x,z,offset) in enumerate(histories(data)):
        if t not in BUDGETS:
            continue
        k = BUDGETS.index(t)
        for m,p in enumerate((z@w[0]+offset, z@w[1]+offset, x@w[0], x@w[1], offset)):
            scores = p.reshape(games,2,3)
            ranks = np.broadcast_to(order[:,None,t,:], scores.shape)
            selected = np.take_along_axis(scores,ranks,axis=-1).argmax(axis=-1)
            a = np.take_along_axis(ranks,selected[...,None],axis=-1)
            decision[:,:,m,k] = a[...,0]
            value[:,:,m,k] = np.take_along_axis(data['truth'][:,:,t],a,axis=-1)[...,0]
    return dict(order=order, decision=decision, value=value)


def deploy(data, w, rng):
    games, h = len(data['seed']), 24
    uniforms = rng.random((games,h))
    gates = rng.random((games,h))
    order = rng.random((games,h,3)).argsort(axis=-1)
    actions = np.empty((games,2,3,h,2), dtype=np.int8)
    rewards = np.empty_like(actions,dtype=float)
    updates = np.zeros(actions.shape[:-1])
    for i, seed in enumerate(data['seed']):
        rule = game(seed,FAMILIES[data['family'][i]],3,3)
        for e in (0,1):
            for m,mean in enumerate((w[0],w[1],np.zeros(D))):
                model, opponent = Model(3,mean), Opponent(3)
                opponent.q = data['warm_q'][i].copy()
                env = StructuredEnv(rule,max_steps=h)
                obs = env.reset(seed=int(data['reset_seed'][i]))
                for t in range(h):
                    x = features(obs[0],3)
                    a = model.act(x,order[i,t])
                    probs = opponent.distribution(obs[1])
                    b = min(2,int(np.searchsorted(probs.cumsum(),uniforms[i,t],side='right')))
                    env._rng = Gate(gates[i,t])
                    result = env.step(a,b)
                    model.observe(x,a,result.rewards[0])
                    before = opponent.change
                    if e:
                        opponent.update(obs[1],b,result.rewards[1],result.observations[1],result.truncated)
                    actions[i,e,m,t] = (a,b)
                    rewards[i,e,m,t] = result.rewards
                    updates[i,e,m,t] = opponent.change-before
                    obs = result.observations
    return dict(deploy_actions=actions,deploy_rewards=rewards,deploy_q_delta=updates,
                deploy_u=uniforms,deploy_gate=gates,deploy_order=order)
