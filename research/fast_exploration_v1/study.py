"""New exploratory Teacher/Student study; historical D-031 remains immutable.

Teacher: a belief-conditioned recurrent actor trained by episodic actor-critic.
Student: one fixed Bayesian linear reward learner (Gaussian prior + online ridge).
Opponent: a genuinely learning epsilon-greedy tabular Q agent.
"""
import argparse
import hashlib
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mamrl.structured_games import StructuredEnv
from research.structured_scale.run import FAMILIES, Model, Opponent, features, game
from research.opponent_adaptation.data import sample as prior_samples
from research.structured_oracle.run import Gate
from research.opponent_adaptation.study import fit as pooled_fit
from . import prior as prior_module
from .prior import absolute_auc, auc_weights, fit_return_prior

KS = (0, 2, 4, 8, 16)
H = 24

# A sealed experiment depends on more than the two entry-point scripts.
# Changing the environment, student, opponent, oracle gate, or evaluation
# code must not silently reuse checkpoints from an earlier scientific run.
SOURCE_PATHS = (
    'research/fast_exploration_v1/study.py',
    'research/fast_exploration_v1/prior.py',
    'research/fast_exploration_v1/figure.py',
    'research/fast_exploration_v1/PROTOCOL.md',
    'research/opponent_adaptation/data.py',
    'research/opponent_adaptation/study.py',
    'research/structured_scale/run.py',
    'research/structured_oracle/run.py',
    '05_toy_models/mamrl/structured_games.py',
    '05_toy_models/mamrl/rules.py',
    '05_toy_models/mamrl/tensor.py',
    '05_toy_models/mamrl/common.py',
    '05_toy_models/mamrl/environment.py',
)


def source_fingerprints():
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def resolve_device(requested):
    """An explicit CUDA request must never silently fall back to CPU."""
    if isinstance(requested, torch.device):
        requested=requested.type
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False; install a CUDA PyTorch wheel')
    return torch.device(requested)


class BeliefActor(nn.Module):
    """Recurrent exploration actor. CPU simulator and CUDA training share features.

    Per-action features comprise the five historical observations and the
    Student's posterior expected reward / predictive uncertainty. Train with
    batched replay on the chosen torch device, never per-step GPU backward.
    """
    def __init__(self, seed=0, hidden_size=32):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gru = nn.GRUCell(7, hidden_size)
            self.actor = nn.Sequential(nn.Linear(2*hidden_size,hidden_size),nn.Tanh(),nn.Linear(hidden_size,1))
            self.critic = nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.Tanh(),nn.Linear(hidden_size,1))
        self.hidden_size=hidden_size

    def initial_hidden(self, *, n, batch=1):
        return torch.zeros(batch,n,self.hidden_size,device=next(self.parameters()).device)

    @staticmethod
    def observation_features(observation, student, x, *, belief=True):
        n=len(x)
        if belief:
            prediction=np.einsum('ad,ad->a',student.mean,x)
            variance=np.einsum('ad,ade,ae->a',x,student.inverse,x)
            uncertainty=np.sqrt(np.maximum(variance,0.))
        else:
            prediction=np.zeros(n)
            uncertainty=np.zeros(n)
        return np.asarray([
            [float(a==observation.previous_own_action),
             float(a==observation.previous_opponent_action),
             math.tanh(observation.previous_own_reward/4),
             float(observation.episode_start_flag),1/n,
             float(np.tanh(prediction[a])),float(np.tanh(uncertainty[a]))]
            for a in range(n)],dtype=np.float32)

    def forward_features(self, x, hidden):
        """Vectorized GRU forward: x [batch, action, 7]."""
        if x.ndim != 3 or hidden.ndim != 3 or x.shape[:2] != hidden.shape[:2]:
            raise ValueError('expected [batch, actions, features] and matching hidden state')
        batch,n,_=x.shape
        updated=self.gru(x.reshape(batch*n,7),hidden.reshape(batch*n,self.hidden_size))
        updated=updated.reshape(batch,n,self.hidden_size)
        pooled=updated.mean(dim=1)
        logits=self.actor(torch.cat((updated,pooled[:,None,:].expand_as(updated)),dim=-1)).squeeze(-1)
        values=self.critic(pooled).squeeze(-1)
        return logits,values,updated

    def forward_observation(self, observation, hidden, *, student, x, belief=True):
        z=torch.as_tensor(self.observation_features(observation,student,x,belief=belief),
                          device=next(self.parameters()).device)
        if hidden.ndim == 2:
            logits,value,updated=self.forward_features(z[None],hidden[None])
            return logits[0],value[0],updated[0]
        logits,value,updated=self.forward_features(z[None],hidden)
        return logits[0],value[0],updated

CONFIGS = {
    'dev': dict(blocks=2, test_games=16, prior_games=32,
                teacher_updates=8, batch=6, prior_steps=25, test_resets=2),
    'final': dict(blocks=8, test_games=128, prior_games=128,
                  teacher_updates=400, batch=32, prior_steps=160, test_resets=2),
}


def _agent_step(env, opponent, observations, own_action, uniform, gate, *,
                learn=True, learning_next=None, terminal_override=None):
    probs = opponent.distribution(observations[1])
    other = min(env.n - 1, int(np.searchsorted(probs.cumsum(), uniform, side='right')))
    env._rng = Gate(float(gate))
    result = env.step(int(own_action), other)
    if learn:
        opponent.update(observations[1], other, result.rewards[1],
                        result.observations[1] if learning_next is None else learning_next[1],
                        result.truncated if terminal_override is None else terminal_override)
    return result


def rollout(seed, family, prior, k, *, teacher=None, method='learned', h=H,
            freeze_assessment=False, opponent_learning=True, neutralize_opponent=False):
    """One paired initial condition, fresh student, warmed adaptive opponent.

    Teacher controls exactly k exploration steps. Student passively observes
    these steps, then plays greedily for h assessment steps. The student's
    posterior is reset for each distinct rollout; only the fitted prior carries.
    """
    if method not in ('learned', 'blind', 'random', 'greedy'):
        raise ValueError('unknown teacher')
    if k < 0 or h < 1 or (method in ('learned','blind') and teacher is None):
        raise ValueError('invalid rollout')
    streams = np.random.SeedSequence(seed).spawn(5)
    rng = np.random.default_rng(streams[0])
    exploration_rng = np.random.default_rng(streams[1])
    baseline_rng = np.random.default_rng(streams[2])
    assess_rng = np.random.default_rng(streams[3])
    sample_seed = int(streams[4].generate_state(1)[0])
    sample_generator = torch.Generator().manual_seed(sample_seed)
    rule = game(int(rng.integers(2**61)), family, 3, 3)
    student = Model(3, prior)
    opponent = Opponent(3)
    warm = StructuredEnv(rule, max_steps=32)
    observations = warm.reset(seed=int(rng.integers(2**61)))
    for _ in range(32):
        action = int(rng.integers(3))
        result = _agent_step(warm, opponent, observations, action, float(rng.random()),float(rng.random()))
        observations = result.observations
    warm_q = opponent.q.copy()
    # Compute an assessment reset independently of all exploration actions.
    assess = StructuredEnv(rule, max_steps=h)
    assess_start = assess.reset(seed=int(assess_rng.integers(2**61)))
    explore = StructuredEnv(rule, max_steps=max(1,k))
    observations = explore.reset(seed=int(exploration_rng.integers(2**61)))
    logps, values, entropies = [], [], []
    policy_features, policy_actions = [], []
    hidden = teacher.initial_hidden(n=3)[0] if method in ('learned','blind') else None
    exploration_reward = 0.
    for t in range(k):
        x = features(observations[0], 3)
        if method in ('learned','blind'):
            policy_features.append(BeliefActor.observation_features(observations[0],student,x,belief=method=='learned'))
            logits, value, hidden = teacher.forward_observation(
                observations[0],hidden,student=student,x=x,belief=method=='learned')
            dist = Categorical(logits=logits)
            # The per-rollout generator is CPU-backed.  Direct CUDA actor
            # inference (e.g. in evaluation/tests) must not pass CUDA
            # probabilities to a CPU generator.  Sampling stays on CPU;
            # log-probability is evaluated on the actor's own device.
            probs_cpu = dist.probs.detach().cpu()
            a_tensor = torch.multinomial(probs_cpu, 1, generator=sample_generator).squeeze(0)
            action = int(a_tensor.item())
            policy_actions.append(action)
            logps.append(dist.log_prob(a_tensor.to(logits.device)))
            values.append(value)
            entropies.append(dist.entropy())
        elif method == 'greedy':
            action = student.act(x, np.arange(3))
        else:
            action = int(baseline_rng.integers(3))
        result = _agent_step(explore, opponent, observations, action,
                             float(exploration_rng.random()),float(exploration_rng.random()),
                             learning_next=assess_start if t==k-1 else None,
                             terminal_override=False)
        student.observe(x, action, result.rewards[0])
        exploration_reward += result.rewards[0]
        observations = result.observations
    # Assessment reset does not erase learned student coefficients or opponent Q.
    if neutralize_opponent:
        opponent.q = warm_q.copy()
    observations = assess_start
    student.frozen = freeze_assessment
    rewards = []
    for _ in range(h):
        x = features(observations[0],3)
        action = student.act(x, np.arange(3))
        probs = opponent.distribution(observations[1])
        other = min(2, int(np.searchsorted(probs.cumsum(), assess_rng.random(), side='right')))
        assess._rng = Gate(float(assess_rng.random()))
        result = assess.step(action, other)
        if opponent_learning:
            opponent.update(observations[1], other, result.rewards[1],
                            result.observations[1], result.truncated)
        student.observe(x, action, result.rewards[0])
        rewards.append(result.rewards[0])
        observations = result.observations
    return dict(assessment=float(np.mean(rewards)), assessment_rewards=[float(r) for r in rewards],
                explore_sum=float(exploration_reward),
                total=float(exploration_reward + sum(rewards)),
                logs=logps, values=values, entropies=entropies,
                features=policy_features, actions=policy_actions)


def _train_loss(actor, batch):
    """Replay the sampled CPU trajectories in one batched CUDA autograd graph.

    Every example in one batch has the same sampled exploration budget, so
    actions and feature tensors can be stacked without padding/masking.
    """
    if not batch or not batch[0]['actions']:
        raise ValueError('training batch needs positive exploration horizon')
    k=len(batch[0]['actions'])
    if any(len(item['actions'])!=k for item in batch):
        raise ValueError('mixed exploration horizons in a batched update')
    return replay_loss(actor, np.stack([item['features'] for item in batch]),
                       np.stack([item['actions'] for item in batch]),
                       [item['assessment'] for item in batch])


def replay_loss(actor, features, actions, rewards):
    """Replay stacked histories on the actor device and retain gradients."""
    device = next(actor.parameters()).device
    x = torch.as_tensor(features, device=device)
    actions = torch.as_tensor(actions, device=device, dtype=torch.long)
    reward = torch.as_tensor(rewards, device=device, dtype=torch.float32)
    k = x.shape[1]
    hidden = actor.initial_hidden(n=x.shape[2], batch=x.shape[0])
    policy=reward.new_zeros(())
    critic=reward.new_zeros(())
    entropy=reward.new_zeros(())
    for t in range(k):
        logits, values, hidden=actor.forward_features(x[:,t],hidden)
        dist=Categorical(logits=logits)
        advantage=(reward-values.detach())
        policy=policy-(dist.log_prob(actions[:,t])*advantage).mean()
        critic=critic+(values-reward).square().mean()
        entropy=entropy+dist.entropy().mean()
    return policy + .25*critic/k - .01*entropy/k


def train_teacher(prior, seed, cfg, *, device='auto', belief=True):
    """GPU-batched actor-critic with CPU environment sampling.

    One K is sampled per update with normalized trapezoid weights. This gives
    the same expected absolute-AUC objective as independent K per rollout,
    while enabling one batched GRU replay / backward pass on CUDA. The CPU
    inference copy avoids hundreds of tiny GPU synchronizations during the
    inherently Python/NumPy environment interactions.
    """
    device=resolve_device(device)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    actor=BeliefActor(seed=seed).to(device)
    optimizer=torch.optim.Adam(actor.parameters(),lr=.001)
    weights=auc_weights(KS)[1:].copy()
    weights /= weights.sum()
    for update in range(cfg['teacher_updates']):
        k=int(rng.choice(KS[1:],p=weights))
        # Cheap individual inferences are faster on CPU; batched gradients on GPU.
        sampling_actor=copy.deepcopy(actor).to('cpu').eval()
        batch=[]
        with torch.no_grad():
            for i in range(cfg['batch']):
                fam=FAMILIES[(update*cfg['batch']+i) % len(FAMILIES)]
                batch.append(rollout(int(rng.integers(2**61)),fam,prior,k,
                                     teacher=sampling_actor,method='learned' if belief else 'blind', h=cfg.get('h',H)))
        loss=_train_loss(actor,batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step()
    return actor.eval()

def evaluate(prior, teacher, seeds, *, h=H, controlled=False, blind_teacher=None):
    """Identical game/initial RNG and Student prior across teacher policies."""
    methods=('learned','blind','random','greedy') if blind_teacher is not None else ('learned','random','greedy')
    results = {m: {k: [] for k in KS} for m in methods}
    totals = {m: {k: [] for k in KS} for m in results}
    step_rewards = {m: {k: [] for k in KS} for m in results}
    with torch.no_grad():
        for i, seed in enumerate(seeds):
            fam = FAMILIES[i % len(FAMILIES)]
            for k in KS:
                for m in results:
                    # For k=0, no teacher acts, and outcomes must coincide.
                    controller=teacher if m=='learned' else blind_teacher if m=='blind' else None
                    record=rollout(int(seed), fam, prior, k, teacher=controller,
                                   method=m,h=h,freeze_assessment=controlled,neutralize_opponent=controlled,
                                   opponent_learning=not controlled)
                    results[m][k].append(record['assessment'])
                    totals[m][k].append(record['total'])
                    step_rewards[m][k].append(record['assessment_rewards'])
    curves={m:[float(np.mean(results[m][k])) for k in KS] for m in results}
    # Mean across held-out games/resets at every Student assessment step.
    # Distinct from the adaptation curve whose x axis is the probe budget K.
    assessment_steps={m:{str(k):np.mean(step_rewards[m][k],axis=0).tolist()
                         for k in KS} for m in results}
    return dict(ks=list(KS),curves=curves,assessment_steps=assessment_steps,
                auc={m:absolute_auc(KS,v) for m,v in curves.items()},
                total_return={m:[float(np.mean(totals[m][k])) for k in KS] for m in totals},
                r0_equal=all(np.array_equal(results['learned'][0],results[m][0]) for m in results))


def execute(split, output, seed=7311903, *, device='cpu'):
    cfg=CONFIGS[split]
    device=resolve_device(device)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(split=split,seed=seed,config=cfg,device=str(device),
                  torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  numpy=np.__version__,source=source_fingerprints())
    manifest=output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('output manifest mismatch; use a new --output directory')
    else:
        if list(output.glob('block_*')):
            raise ValueError('existing blocks without an experiment manifest')
        manifest.write_text(json.dumps(identity,indent=2)+'\n')
    records=[]
    for b in range(cfg['blocks']):
        dest=output/f'block_{b:03d}.json'
        weights=output/f'block_{b:03d}.pt'
        if dest.exists() != weights.exists():
            raise ValueError(f'incomplete saved block {b}: both JSON and .pt are required')
        if dest.exists() and weights.exists():
            record=json.loads(dest.read_text())
            if record.get('block') != b:
                raise ValueError(f'incorrect block index in {dest}')
            digest=hashlib.sha256(weights.read_bytes()).hexdigest()
            if record.get('weights_sha256') != digest:
                raise ValueError(f'checkpoint checksum mismatch: {weights}')
            records.append(record)
            continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,b]))
        train_seed, prior_seed, test_seed = [int(s) for s in rng.integers(2**31,size=3)]
        prior_data=prior_samples(prior_seed,cfg['prior_games'])
        meta=fit_return_prior(prior_data,steps=cfg['prior_steps'],seed=train_seed,device=device)
        # Explicit same-data baseline from historical v0.7 prediction objective.
        _,pooled=pooled_fit(prior_data)
        actor=train_teacher(meta,train_seed,cfg,device=device,belief=True)
        blind_actor=train_teacher(meta,train_seed,cfg,device=device,belief=False)
        # Per-step simulator evaluation is CPU-bound: use identical CPU copies.
        actor_cpu=copy.deepcopy(actor).to('cpu').eval()
        blind_cpu=copy.deepcopy(blind_actor).to('cpu').eval()
        evaluation_seeds=np.random.default_rng(test_seed).integers(2**61,size=cfg['test_games']*cfg['test_resets'])
        scores=evaluate(meta,actor_cpu,evaluation_seeds,blind_teacher=blind_cpu)
        controlled_scores=evaluate(meta,actor_cpu,evaluation_seeds,controlled=True,blind_teacher=blind_cpu)
        # Quantify initial-prior sensitivity with an unchanged, learned teacher.
        pooled_scores=evaluate(pooled,actor_cpu,evaluation_seeds,blind_teacher=blind_cpu)
        zero_scores=evaluate(np.zeros_like(meta),actor_cpu,evaluation_seeds,blind_teacher=blind_cpu)
        record=dict(block=b,seeds=[train_seed,prior_seed,test_seed],
                    trained_prior=meta.tolist(),pooled_prior=pooled.tolist(),
                    reward_auc=scores,controlled_auc=controlled_scores,
                    pooled_auc=pooled_scores,zero_auc=zero_scores)
        torch.save({'bayesian_teacher':actor_cpu.state_dict(),
                    'blind_teacher':blind_cpu.state_dict()}, weights)
        record['weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
        dest.write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
    # Independent training blocks are the resampling units, not games/turns.
    differences=np.array([[p['reward_auc']['auc']['learned']-p['reward_auc']['auc'][m]
                           for m in ('blind','random','greedy')] for p in records])
    rng=np.random.default_rng(873321)
    idx=rng.integers(len(records),size=(10000,len(records)))
    means=differences[idx].mean(axis=1)
    interval=np.quantile(means,[.025,.975],axis=0)
    controlled_differences=np.array([[p['controlled_auc']['auc']['learned']-p['controlled_auc']['auc'][m]
                           for m in ('blind','random','greedy')] for p in records])
    report=dict(split=split,device=str(device),blocks=len(records),sampled_test_games_per_block=cfg['test_games'],
                teacher_minus_blind=float(differences[:,0].mean()),
                teacher_minus_random=float(differences[:,1].mean()),
                teacher_minus_greedy=float(differences[:,2].mean()),
                descriptive_ci95=interval.T.tolist(),
                controlled_teacher_minus_blind=float(controlled_differences[:,0].mean()),
                controlled_teacher_minus_random=float(controlled_differences[:,1].mean()),
                controlled_teacher_minus_greedy=float(controlled_differences[:,2].mean()),
                all_zero_shot_equal=all(p['reward_auc']['r0_equal'] for p in records),
                limitations='Exploration may shape game/opponent state; prior uses IPS softmax surrogate; not pure information gain.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Plotting is derived: checkpoints and metrics are unchanged.
    try:
        from .figure import render
        render(output)
    except ModuleNotFoundError as exc:
        if exc.name != 'matplotlib':
            raise
        print('Plots skipped: install matplotlib and run '
              'python -m research.fast_exploration_v1.figure --output ' + str(output))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=CONFIGS,default='dev')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=7311903)
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto',
                    help='CUDA for neural training and Bayesian prior optimization; CPU simulation')
    args=ap.parse_args()
    print(json.dumps(execute(args.split,args.output,args.seed,device=args.device),indent=2))

if __name__=='__main__':
    main()
