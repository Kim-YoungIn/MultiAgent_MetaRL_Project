"""Shared batched collection and evaluation for the Fast Exploration study."""
import copy
from dataclasses import dataclass

import numpy as np
import torch

from research.structured_scale.run import FAMILIES, game
from .prior import absolute_auc, auc_weights
from .simulator import EpisodeBatch, StudentBatch, opponent_distribution, opponent_update

METHODS = ('learned', 'blind', 'random', 'greedy')


@dataclass
class Episodes:
    rules: list
    warm_seed: list
    warm_action: np.ndarray
    warm_uniform: np.ndarray
    explore_seed: list
    explore_uniform: np.ndarray
    random_actions: np.ndarray
    teacher_seed: list
    assess_seed: list
    assess_uniform: np.ndarray
    template: EpisodeBatch | None = None

    def episode(self, reset_seeds, horizon):
        if self.template is None:
            self.template = EpisodeBatch(self.rules, reset_seeds, horizon)
        return self.template.fork_reset(reset_seeds, horizon)


def prepare(seeds, families, k_max, h, *, rule_seeds=None):
    """Use the scalar collector's independent streams, with optional rule/reset separation."""
    rules, warm_seeds, warm_actions, warm_draws = [], [], [], []
    explore_seeds, explore_draws, random_actions, teacher_seeds = [], [], [], []
    assess_seeds, assess_draws = [], []
    if rule_seeds is None:
        rule_seeds = [None]*len(seeds)
    for seed, family, explicit in zip(seeds, families, rule_seeds, strict=True):
        children = np.random.SeedSequence(int(seed)).spawn(5)
        warm, explore, baseline, assess = (np.random.default_rng(c) for c in children[:4])
        implicit = int(warm.integers(2**61))
        rules.append(game(implicit if explicit is None else int(explicit), family, 3, 3))
        warm_seeds.append(int(warm.integers(2**61)))
        actions, uniforms = [], []
        for _ in range(32):
            actions.append(int(warm.integers(3)))
            uniforms.append(warm.random(2))
        warm_actions.append(actions)
        warm_draws.append(uniforms)
        explore_seeds.append(int(explore.integers(2**61)))
        explore_draws.append(explore.random((k_max,2)))
        random_actions.append(baseline.integers(3,size=k_max))
        teacher_seeds.append(int(children[4].generate_state(1)[0]))
        assess_seeds.append(int(assess.integers(2**61)))
        assess_draws.append(assess.random((h,2)))
    return Episodes(rules,warm_seeds,np.array(warm_actions).T,np.array(warm_draws).transpose(1,0,2),
                    explore_seeds,np.array(explore_draws).transpose(1,0,2),np.array(random_actions).T,
                    teacher_seeds,assess_seeds,np.array(assess_draws).transpose(1,0,2))


def opponent_actions(q, last, draws):
    probs = opponent_distribution(q,last)
    return np.minimum(2,(probs.cumsum(axis=1) <= draws[:,None]).sum(1))


def warm_up(episodes):
    q = np.zeros((len(episodes.rules),3,3,3))
    env = episodes.episode(episodes.warm_seed,32)
    for t,(action,draws) in enumerate(zip(episodes.warm_action,episodes.warm_uniform,strict=True)):
        old = env.last.copy()
        other = opponent_actions(q,old,draws[:,0])
        r = env.step(action,other,draws[:,1])
        opponent_update(q,old,env.last,other,r[:,1],terminal=t==31)
    return q


def actor_inputs(env, student, x, belief):
    last = env.last
    a = np.arange(3)[None,:]
    reward = env.previous_reward
    out = np.empty((len(last),3,7),dtype=np.float32)
    out[...,0] = last[:,0,None]==a
    out[...,1] = last[:,1,None]==a
    out[...,2] = np.tanh(np.asarray(reward)[:,None]/4)
    out[...,3] = env.elapsed == 0
    out[...,4] = 1/3
    out[...,5] = np.tanh(student.predict(x)) if belief else 0.
    out[...,6] = np.tanh(student.uncertainty(x)) if belief else 0.
    return out


@torch.no_grad()
def collect(episodes, initial_q, prior, k, h, *, method, teacher=None,
            live=True, controlled=False, record=False):
    """Collect one probe batch. Fork both evaluation regimes from its handoff."""
    if method not in METHODS or (method in METHODS[:2] and teacher is None):
        raise ValueError('invalid exploration policy')
    size = len(episodes.rules)
    student = StudentBatch(size,prior)
    q = initial_q.copy()
    assess = episodes.episode(episodes.assess_seed,h)
    setup = assess.last.copy()
    exploration = np.zeros((size,k))
    history_x = np.empty((size,k,3,7),dtype=np.float32) if record else None
    history_a = np.empty((size,k),dtype=np.int8)
    if k:
        env = episodes.episode(episodes.explore_seed,k)
        if method in METHODS[:2]:
            hidden = teacher.initial_hidden(n=3,batch=size)
            generators = [torch.Generator().manual_seed(seed) for seed in episodes.teacher_seed]
            device = next(teacher.parameters()).device
        for t in range(k):
            x = env.features()
            if method in METHODS[:2]:
                z = actor_inputs(env,student,x,method=='learned')
                logits,_,hidden = teacher.forward_features(torch.as_tensor(z,device=device),hidden)
                probs = torch.softmax(logits-logits.logsumexp(dim=-1,keepdim=True),dim=-1).cpu()
                action = np.array([int(torch.multinomial(row,1,generator=rng))
                                   for row,rng in zip(probs,generators,strict=True)])
                if record:
                    history_x[:,t] = z
            elif method == 'greedy':
                action = student.predict(x).argmax(axis=1)
            else:
                action = episodes.random_actions[t]
            history_a[:,t] = action
            old = env.last.copy()
            draw = episodes.explore_uniform[t]
            other = opponent_actions(q,old,draw[:,0])
            rewards = env.step(action,other,draw[:,1])
            opponent_update(q,old,setup if t==k-1 else env.last,other,rewards[:,1])
            student.observe(x,action,rewards[:,0])
            exploration[:,t] = rewards[:,0]
    rewards = {}
    if controlled:
        frozen_env = episodes.episode(episodes.assess_seed,h) if live else assess
        rewards['controlled'] = assess_batch(frozen_env,episodes.assess_uniform,student,initial_q,learn=False)
    if live:
        rewards['live'] = assess_batch(assess,episodes.assess_uniform,student,q,learn=True)
    return rewards, exploration, history_x, history_a


def assess_batch(env, draws, student, q, *, learn):
    h, size = draws.shape[:2]
    results = np.empty((size,h))
    for t,u in enumerate(draws):
        x = env.features()
        action = student.predict(x).argmax(axis=1)
        last = env.last.copy()
        other = opponent_actions(q,last,u[:,0])
        r = env.step(action,other,u[:,1])
        results[:,t] = r[:,0]
        if learn:
            opponent_update(q,last,env.last,other,r[:,1],terminal=t==h-1)
            student.observe(x,action,r[:,0])
    return results


def train_group(prior, seed, cfg, *, device, sampling, methods, history=None, resume=None):
    """Train paired policies from one shared warmed batch per update."""
    from .study import BeliefActor, replay_loss
    rng = np.random.default_rng(seed)
    actors = {m: BeliefActor(seed=seed).to(device) for m in methods}
    optimizers = {m: torch.optim.Adam(model.parameters(), lr=.001) for m, model in actors.items()}
    first = 0
    if resume is not None:
        for m in methods:
            actors[m].load_state_dict(resume['models'][m])
            optimizers[m].load_state_dict(resume['optimizers'][m])
        rng.bit_generator.state = resume['rng']
        first = resume['update']
    same_device = torch.device(sampling) == torch.device(device)
    sampling_actors = actors if same_device else {m: copy.deepcopy(v).to(sampling) for m, v in actors.items()}
    ks = tuple(cfg['ks'])
    weights = auc_weights(ks)[1:]
    weights = weights/weights.sum()
    samples = cfg['batch']
    for update in range(first,cfg['teacher_updates']):
        k = int(rng.choice(ks[1:],p=weights))
        seeds = rng.integers(2**61,size=samples)
        families = [FAMILIES[(update*samples+i)%4] for i in range(samples)]
        episodes = prepare(seeds,families,k,cfg['h'])
        initial_q = warm_up(episodes)
        metrics = {}
        for m, actor in actors.items():
            if not same_device:
                sampling_actors[m].load_state_dict(actor.state_dict())
            results, _, x, actions = collect(episodes,initial_q,prior,k,cfg['h'],
                method=m,teacher=sampling_actors[m],record=True)
            returns = results['live'].mean(axis=1)
            loss = replay_loss(actor,x,actions,returns)
            optimizer = optimizers[m]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            metrics[m] = float(returns.mean())
        if history is not None:
            history(update+1,actors,optimizers,rng,metrics)
    return {m: actor.eval() for m, actor in actors.items()}


def train(prior, seed, cfg, *, device, sampling, belief):
    method = 'learned' if belief else 'blind'
    return train_group(prior,seed,cfg,device=device,sampling=sampling,methods=(method,))[method]


def evaluate(priors, actors, episodes, cfg):
    h, ks = cfg['h'],cfg['ks']
    initial_q = warm_up(episodes)
    size = len(episodes.rules)
    output, raw = {}, {}
    for label, prior in priors.items():
        evaluate_controlled = label=='reward_auc'
        regimes = ('live','controlled') if evaluate_controlled else ('live',)
        shape = (len(METHODS),len(ks),size)
        scores = {r: np.empty(shape+(h,)) for r in regimes}
        exploration = np.empty(shape)
        for i,k in enumerate(ks):
            for j,method in enumerate(METHODS if k else METHODS[:1]):
                rewards, probe, _, _ = collect(episodes,initial_q,prior,k,h,method=method,
                    teacher=actors.get(method), controlled=evaluate_controlled)
                for r in regimes:
                    scores[r][j,i] = rewards[r]
                exploration[j,i] = probe.sum(1)
            if not k:
                for r in regimes:
                    scores[r][:,i] = scores[r][0,i]
                exploration[:,i] = exploration[0,i]
        for regime, values in scores.items():
            name = label if regime=='live' else 'controlled_auc'
            curves = values.mean(axis=(2,3))
            totals = (values.sum(3)+exploration).mean(axis=2)
            output[name] = dict(ks=list(ks), curves=dict(zip(METHODS,curves.tolist(),strict=True)),
                auc={m:absolute_auc(ks,row) for m,row in zip(METHODS,curves,strict=True)},
                assessment_steps={m:{str(k):v.tolist() for k,v in zip(ks,values[j].mean(1),strict=True)}
                                  for j,m in enumerate(METHODS)},
                total_return=dict(zip(METHODS,totals.tolist(),strict=True)),r0_equal=True)
            raw[name+'_assessment'] = values
            raw[name+'_exploration'] = exploration.copy()
    return output, raw
