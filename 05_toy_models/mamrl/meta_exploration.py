"""Learn exploration from downstream adaptation against ONLINE-learning opponents."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from .common import positive_integer, validate_seed
from .environment import Observation
from .generated_tasks import build_manifest
from .logical_rules import rule_from_dict
from .meta_exploration_report import render_meta_report
from .experiment_artifacts import save_meta_experiment, load_meta_experiment


SPLITS=('train_rule_reference','fresh_rules','structural','new_learning_profile')


def decode_rule(data):
    if isinstance(data,dict) and data.get('format')=='relational_rule':
        from .relational_rules import relational_from_dict
        return relational_from_dict(data)
    return rule_from_dict(data)


@dataclass(frozen=True)
class MetaConfig:
    seed: int = 7
    updates: int = 40
    trials_per_update: int = 8
    steps: int = 12
    hidden_size: int = 32
    epochs: int = 3
    learning_rate: float = .003

    def __post_init__(self):
        if validate_seed(self.seed) is None:
            raise TypeError('seed must be explicit')
        for name in ('updates','trials_per_update','steps','hidden_size','epochs'):
            positive_integer(getattr(self,name),name)
        if isinstance(self.learning_rate,bool) or not math.isfinite(self.learning_rate) or self.learning_rate<=0:
            raise ValueError('learning_rate must be finite and positive')


class NeuralExplorer:
    """Observation-only teacher. No student table or task metadata in forward."""
    def __init__(self, *, model, n, seed):
        import torch
        self.model,self.n=model,n
        self.generator=torch.Generator(device='cpu').manual_seed(seed % 2**63)
        self.hidden=model.initial_hidden(n=n)
        self.events,self.actions,self.old_log_probs,self.old_values=[],[],[],[]

    def act(self, observation):
        import torch
        with torch.no_grad():
            logits,value,self.hidden=self.model.forward_observation(observation,self.hidden,n=self.n)
            distribution=torch.log_softmax(logits,dim=-1)
            action=int(torch.multinomial(distribution.exp(),1,generator=self.generator).item())
        self.events.append(observation)
        self.actions.append(action)
        self.old_log_probs.append(float(distribution[action]))
        self.old_values.append(float(value))
        return action


class RandomExplorer:
    def __init__(self, *, n, seed, student):
        self.n,self.rng=n,random.Random(seed)
    def act(self,obs): return self.rng.randrange(self.n)


class EpsilonExplorer:
    def __init__(self, *, n, seed, student):
        self.student=student
        # Baseline action RNG belongs to the teacher, not the student's future
        # assessment RNG. Selection still uses only experience-derived student Q.
        self.rng=random.Random(seed)
    def act(self,obs):
        probs=self.student.distribution(obs,epsilon=.2)
        value=self.rng.random()
        cumulative=0.0
        for action,prob in enumerate(probs):
            cumulative+=prob
            if value<cumulative:
                return action
        return len(probs)-1


class RepeatExplorer:
    def __init__(self, *, n, seed, student): pass
    def act(self,obs): return obs.previous_own_action


class BalancedExplorer:
    """Balance repeat/change without knowing whether the true rule calls them C/D."""
    def __init__(self, *, n, seed, student):
        self.n,self.rng=n,random.Random(seed)
    def act(self,obs):
        if self.rng.random()<.5:
            return obs.previous_own_action
        chosen=self.rng.randrange(self.n-1)
        return chosen+(chosen>=obs.previous_own_action)


class ChangeExplorer:
    """Always choose a different action: a rule-blind stage-defection control."""
    def __init__(self, *, n, seed, student):
        self.n,self.rng=n,random.Random(seed)
    def act(self,obs):
        chosen=self.rng.randrange(self.n-1)
        return chosen+(chosen>=obs.previous_own_action)


@dataclass(frozen=True)
class ExplorationRollout:
    events: tuple[Observation,...]
    actions: tuple[int,...]
    old_log_probs: tuple[float,...]
    old_values: tuple[float,...]
    meta_reward: float
    trial: dict


def collect_exploration(model, rule, *, seed, steps=12, opponent_profile='steady'):
    from .meta_exploration_rollout import run_meta_trial
    teachers=[]
    def factory(*,n,seed,student):
        teacher=NeuralExplorer(model=model,n=n,seed=seed)
        teachers.append(teacher)
        return teacher
    trial=run_meta_trial(rule,factory,seed=seed,steps=steps,opponent_profile=opponent_profile)
    teacher=teachers[0]
    return ExplorationRollout(tuple(teacher.events),tuple(teacher.actions),tuple(teacher.old_log_probs),
                              tuple(teacher.old_values),trial['meta_reward'],trial)


def replay_exploration(model, rollout):
    import torch
    if not rollout.events or len(rollout.events)!=len(rollout.actions):
        raise ValueError('exploration events/actions must align and be nonempty')
    n=rollout.trial.get('n',3)
    hidden=model.initial_hidden(n=n)
    logs,values,entropies=[],[],[]
    for obs,action in zip(rollout.events,rollout.actions):
        logits,value,hidden=model.forward_observation(obs,hidden,n=n)
        distribution=torch.log_softmax(logits,dim=-1)
        logs.append(distribution[action]);values.append(value)
        entropies.append(-(distribution.exp()*distribution).sum())
    return torch.stack(logs),torch.stack(values),torch.stack(entropies)


def meta_targets(rollouts):
    """No exploration rewards here: a terminal downstream target for each action."""
    import torch
    if not rollouts:
        raise ValueError('meta batch must be nonempty')
    values=[r.meta_reward for r in rollouts for _ in r.actions]
    result=torch.tensor(values,dtype=torch.float32)
    if not result.numel() or not torch.isfinite(result).all():
        raise ValueError('meta targets must be nonempty and finite')
    return result


def train_meta(config, manifest=None, *, opponent_suite='q_learning'):
    import torch
    from copy import deepcopy
    from .action_set import ActionSetActorCritic
    from .ppo import ppo_loss
    from .meta_exploration_rollout import OPPONENT_SUITES
    if opponent_suite not in OPPONENT_SUITES:
        raise ValueError('unknown opponent suite')
    profiles=OPPONENT_SUITES[opponent_suite]['train']
    manifest=build_manifest() if manifest is None else manifest
    rules=[r for r in manifest['rules'] if r['split']=='train']
    model=ActionSetActorCritic(hidden_size=config.hidden_size,seed=config.seed)
    initial=deepcopy(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    rng=random.Random(config.seed)
    history=[]
    for update in range(1,config.updates+1):
        batch,provenance=[],[]
        for _ in range(config.trials_per_update):
            entry=rules[rng.randrange(len(rules))]
            profile=profiles[rng.randrange(len(profiles))]
            trial_seed=rng.getrandbits(63)
            rollout=collect_exploration(model,decode_rule(entry['rule']),seed=trial_seed,
                                         steps=config.steps,opponent_profile=profile)
            batch.append(rollout)
            provenance.append({'rule_id':entry['id'],'opponent_profile':profile,'seed':trial_seed,
                               'meta_reward':rollout.meta_reward,
                               'assessment_mean':rollout.trial['assessment_mean'],
                               'erased_mean':rollout.trial['erased_mean']})
        target=meta_targets(batch)
        old_values=torch.tensor([v for r in batch for v in r.old_values],dtype=torch.float32)
        advantage=target-old_values
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        old_logs=torch.tensor([v for r in batch for v in r.old_log_probs],dtype=torch.float32)
        metrics=[]
        for _ in range(config.epochs):
            replay=[replay_exploration(model,r) for r in batch]
            logs=torch.cat([r[0] for r in replay])
            values=torch.cat([r[1] for r in replay])
            entropy=torch.cat([r[2] for r in replay])
            loss,stats=ppo_loss(logs,old_logs,values,target,advantage,entropy,
                                clip_ratio=.2,value_coefficient=.5,entropy_coefficient=.01)
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('nonfinite teacher parameters')
            metrics.append({**stats,'gradient_norm':float(norm)})
        history.append({'update':update,**{k:statistics.mean(x[k] for x in metrics) for k in metrics[0]},
                        'mean_meta_reward':statistics.mean(r.meta_reward for r in batch),
                        'mean_assessment':statistics.mean(r.trial['assessment_mean'] for r in batch),
                        'mean_exploration':statistics.mean(r.trial['exploration_mean'] for r in batch),
                        'opponent_policy_l1_change':statistics.mean(r.trial['opponent_final']['policy_l1_change'] for r in batch),
                        'trials':provenance})
        if update==1 or update%10==0 or update==config.updates:
            print(f'[seed={config.seed} update={update}] meta={history[-1]["mean_meta_reward"]:.4f} assessment={history[-1]["mean_assessment"]:.4f}',file=sys.stderr,flush=True)
    return {'initial_model':initial,'model':model,'history':history,'config':asdict(config)}


def meta_tasks(manifest, split, *, opponent_suite='q_learning'):
    if split not in SPLITS:
        raise ValueError('unknown meta-exploration split')
    source='train' if split in ('train_rule_reference','new_learning_profile') else split
    from .meta_exploration_rollout import OPPONENT_SUITES
    if opponent_suite not in OPPONENT_SUITES:
        raise ValueError('unknown opponent suite')
    suite=OPPONENT_SUITES[opponent_suite]
    profiles=(suite['heldout'],) if split=='new_learning_profile' else suite['train']
    return tuple({'id':r['id']+'/'+profile,'rule_id':r['id'],'rule':r['rule'],
                  'opponent_profile':profile,'split':split,**({'family':r['family']} if 'family' in r else {})}
                 for r in manifest['rules'] if r['split']==source for profile in profiles)


def summarize_meta(rows):
    if not rows:
        raise ValueError('cannot summarize empty trials')
    from collections import defaultdict
    task_values=defaultdict(list)
    for row in rows:
        task_values[row['task_id']].append(row['meta_reward'])
    reps=next(iter(task_values.values()))
    mc_variance=sum(statistics.variance(v)/len(v) for v in task_values.values())/(len(task_values)**2) if len(reps)>1 else None
    result={'trials':len(rows),'tasks':len(task_values),
            **{k:statistics.mean(r[k] for r in rows) for k in ('meta_reward','assessment_mean','erased_mean','exploration_mean')},
            'meta_monte_carlo_se':math.sqrt(mc_variance) if mc_variance is not None else None,
            'assessment_curve':[statistics.mean(r['assessment'][i]['rewards'][0] for r in rows) for i in range(len(rows[0]['assessment']))],
            'exploration_curve':[statistics.mean(r['exploration'][i]['rewards'][0] for r in rows) for i in range(len(rows[0]['exploration']))],
            'mean_opponent_parameter_changes':statistics.mean(r['opponent_final']['parameter_changes'] for r in rows),
            'mean_opponent_policy_l1_change':statistics.mean(r['opponent_final']['policy_l1_change'] for r in rows),
            'opponent_strategy_changed_trials':sum(r['opponent_final']['policy_l1_change']>1e-12 for r in rows),
            'mean_opponent_nonuniform_action_events':statistics.mean(r['opponent_nonuniform_action_events'] for r in rows),
            'opponent_nonuniform_action_fraction':sum(r['opponent_nonuniform_action_events'] for r in rows)/(len(rows)*2*len(rows[0]['assessment']))}
    result['assessment_early']=statistics.mean(result['assessment_curve'][:min(3,len(result['assessment_curve']))])
    result['assessment_late']=statistics.mean(result['assessment_curve'][-min(3,len(result['assessment_curve'])):])
    result['total_gameplay_mean']=statistics.mean((r['exploration_mean']+r['assessment_mean'])/2 for r in rows)
    curve=result['assessment_curve']
    budgets=sorted({min(k,len(curve)) for k in (1,3,6,len(curve))})
    result['assessment_prefix_auc']={str(k):statistics.mean(curve[:k]) for k in budgets}
    center=(len(curve)-1)/2
    denominator=sum((i-center)**2 for i in range(len(curve)))
    result['assessment_slope']=sum((i-center)*v for i,v in enumerate(curve))/denominator if denominator else None
    return result


def evaluate_meta(teacher_factory, tasks, *, steps=12,reps=2,seed=918273):
    from .meta_exploration_rollout import run_meta_trial
    positive_integer(reps,'reps')
    positive_integer(steps,'steps')
    if validate_seed(seed) is None:
        raise TypeError('evaluation seed must be explicit')
    tasks=tuple(tasks)
    if not tasks:
        raise ValueError('evaluation tasks must be nonempty')
    rows=[]
    for task in tasks:
        for rep in range(reps):
            trial_seed=int.from_bytes(hashlib.sha256(f'{seed}/{task["id"]}/{rep}'.encode()).digest()[:8],'big') % 2**63
            trial=run_meta_trial(decode_rule(task['rule']),teacher_factory,seed=trial_seed,
                                 steps=steps,opponent_profile=task['opponent_profile'])
            rows.append({'task_id':task['id'],'rule_id':task['rule_id'],'split':task['split'],'rep':rep,
                         **({'family':task['family']} if 'family' in task else {}),**trial})
    result={'summary':summarize_meta(rows),'trials':rows}
    families=sorted({r['family'] for r in rows if 'family' in r})
    if families:
        result['by_family']={f:summarize_meta([r for r in rows if r.get('family')==f]) for f in families}
    return result


def neural_factory(model):
    def factory(*,n,seed,student):
        return NeuralExplorer(model=model,n=n,seed=seed)
    return factory


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def run_meta_experiment(output_dir, *, seeds=(7,17,27),updates=40,trials_per_update=8,steps=12,hidden_size=32,reps=2,
                        catalog='logical',opponent_suite='q_learning'):
    import torch
    from .recurrent import save_checkpoint,load_checkpoint
    from .meta_exploration_rollout import OPPONENT_PROFILES,OPPONENT_SUITES
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(validate_seed(s) is None for s in seeds):
        raise ValueError('seeds must be unique explicit nonempty seeds')
    for k,v in (('updates',updates),('trials_per_update',trials_per_update),('steps',steps),('hidden_size',hidden_size),('reps',reps)):
        positive_integer(v,k)
    if catalog not in ('logical','relational') or opponent_suite not in OPPONENT_SUITES:
        raise ValueError('unknown catalog or opponent suite')
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=False)
    sources=p/'sources';sources.mkdir()
    hashes={}
    for file in sorted(Path(__file__).parent.glob('*.py')):
        data=file.read_bytes();(sources/file.name).write_bytes(data)
        hashes[file.name]=hashlib.sha256(data).hexdigest()
    if catalog=='relational':
        from .relational_catalog import build_relational_manifest,relational_distances
        manifest=build_relational_manifest()
        _write(p/'distances.json',relational_distances(manifest))
    else:
        manifest=build_manifest()
    _write(p/'manifest.json',manifest)
    protocol={'version':2,'catalog':catalog,'opponent_suite':opponent_suite,'seeds':list(seeds),'updates':updates,'trials_per_update':trials_per_update,
              'steps':steps,'hidden_size':hidden_size,'reps':reps,'n':3,'device':'cpu',
              'torch_version':str(torch.__version__),'torch_threads':torch.get_num_threads(),
              'python_version':sys.version,'source_sha256':hashes,
              'opponent_profiles':OPPONENT_PROFILES,'training_profiles':list(OPPONENT_SUITES[opponent_suite]['train']),
              'heldout_profile':OPPONENT_SUITES[opponent_suite]['heldout'],
              'student':{'algorithm':'tabular Q','alpha':.5,'gamma':.95,'assessment_epsilon':0},
              'objective':'assessment_mean minus paired erased-student assessment_mean; no exploration reward',
              'inner_training_gameplay_steps_per_seed':updates*trials_per_update*steps*2,
              'inner_training_control_steps_per_seed':updates*trials_per_update*steps,
              'teacher_training_decisions_per_seed':updates*trials_per_update*steps,
              'selection':'fixed final checkpoint; final task results never select weights',
              'scope':'one learner algorithm and bounded generated distribution, not general meta-exploration proof'}
    _write(p/'protocol.json',protocol)
    tasks={s:meta_tasks(manifest,s,opponent_suite=opponent_suite) for s in SPLITS}
    options={'steps':steps,'reps':reps}
    baselines={}
    reference_factories=[('random',RandomExplorer),('epsilon_greedy',EpsilonExplorer),('repeat',RepeatExplorer),('balanced',BalancedExplorer)]
    if catalog=='relational': reference_factories.append(('change',ChangeExplorer))
    for name,factory in reference_factories:
        print('[baseline '+name+']',file=sys.stderr,flush=True)
        baselines[name]={s:evaluate_meta(factory,t,**options) for s,t in tasks.items()}
    _write(p/'baselines.json',baselines)
    record={'version':2,'protocol':protocol,'baselines':baselines,'runs':[],'comparisons':{},
            'rule_catalog':manifest['rules'] if catalog=='relational' else []}
    for seed in seeds:
        result=train_meta(MetaConfig(seed=seed,updates=updates,trials_per_update=trials_per_update,steps=steps,hidden_size=hidden_size),manifest,opponent_suite=opponent_suite)
        d=p/f'seed{seed}';d.mkdir()
        save_checkpoint(result['initial_model'],d/'initial.pt');save_checkpoint(result['model'],d/'policy.pt')
        initial,trained=load_checkpoint(d/'initial.pt'),load_checkpoint(d/'policy.pt')
        evaluation={name:{s:evaluate_meta(neural_factory(model),t,**options) for s,t in tasks.items()}
                    for name,model in (('initial',initial),('trained',trained))}
        row={'seed':seed,'config':result['config'],'history':result['history'],'evaluation':evaluation,
             'checkpoint':f'seed{seed}/policy.pt','initial_checkpoint':f'seed{seed}/initial.pt'}
        _write(d/'evaluation.json',row);record['runs'].append(row)
        print('[finished seed '+str(seed)+'] '+', '.join(s+' G='+f'{evaluation["trained"][s]["summary"]["meta_reward"]:.4f}' for s in SPLITS),file=sys.stderr,flush=True)
    for s in SPLITS:
        record['comparisons'][s]={}
        for metric in ('meta_reward','assessment_mean'):
            values=[r['evaluation']['trained'][s]['summary'][metric] for r in record['runs']]
            record['comparisons'][s][metric]={'trained_seed_values':values,'trained_mean':statistics.mean(values),
                                             'learning_seed_sd':statistics.stdev(values) if len(values)>1 else None}
            for reference in ('initial',*baselines):
                diffs=[]
                for row in record['runs']:
                    target=row['evaluation']['trained'][s]
                    control=row['evaluation']['initial'][s] if reference=='initial' else baselines[reference][s]
                    a={(t['task_id'],t['rep']):t for t in target['trials']}
                    b={(t['task_id'],t['rep']):t for t in control['trials']}
                    assert set(a)==set(b)
                    diffs.append(statistics.mean(a[k][metric]-b[k][metric] for k in a))
                record['comparisons'][s][metric]['minus_'+reference]={'seed_differences':diffs,'mean':statistics.mean(diffs)}
    save_meta_experiment(p/'experiment.json',record)
    (p/'report.html').write_text(render_meta_report(record),encoding='utf-8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    parser.add_argument('--catalog',choices=('logical','relational'),default='logical')
    parser.add_argument('--opponent-suite',choices=('q_learning','mixed_learning'),default='q_learning')
    for name,value in (('updates',40),('trials-per-update',8),('steps',12),('hidden-size',32),('reps',2)):
        parser.add_argument('--'+name,type=int,default=value)
    args=parser.parse_args()
    try:
        run_meta_experiment(args.output_dir,seeds=args.seeds,updates=args.updates,trials_per_update=args.trials_per_update,
                            steps=args.steps,hidden_size=args.hidden_size,reps=args.reps,
                            catalog=args.catalog,opponent_suite=args.opponent_suite)
    except (TypeError,ValueError,OSError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__=='__main__': main()
