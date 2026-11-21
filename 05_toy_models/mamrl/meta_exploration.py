"""Learn exploration from downstream adaptation against ONLINE-learning opponents."""

import argparse
from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from .common import positive_integer, validate_n, validate_seed
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
    device: str = 'cpu'
    batch_size: int = 1
    assessment_steps: int | None = None
    train_rule_limit: int | None = None
    objective: str = 'retained_gain'

    def __post_init__(self):
        if self.objective not in ('retained_gain','assessment_return'):raise ValueError('unknown exploration objective')
        if self.train_rule_limit is not None:positive_integer(self.train_rule_limit,'train_rule_limit')
        if self.assessment_steps is not None:positive_integer(self.assessment_steps,'assessment_steps')
        if validate_seed(self.seed) is None:
            raise TypeError('seed must be explicit')
        for name in ('updates','trials_per_update','steps','hidden_size','epochs','batch_size'):
            positive_integer(getattr(self,name),name)
        if isinstance(self.learning_rate,bool) or not math.isfinite(self.learning_rate) or self.learning_rate<=0:
            raise ValueError('learning_rate must be finite and positive')


class NeuralExplorer:
    """Observation-only teacher. No student table or task metadata in forward."""
    def __init__(self, *, model, n, seed):
        import torch
        self.model,self.n=model,n
        self.generator=torch.Generator(device=model.device).manual_seed(seed % 2**63)
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


class SystematicExplorer:
    """Seeded label cycle; no hidden role or rule metadata."""
    def __init__(self,*,n,seed,student):self.n=n;self.action=random.Random(seed).randrange(n)
    def act(self,observation):
        action=self.action;self.action=(action+1)%self.n;return action


@dataclass(frozen=True)
class ExplorationRollout:
    events: tuple[Observation,...]
    actions: tuple[int,...]
    old_log_probs: tuple[float,...]
    old_values: tuple[float,...]
    meta_reward: float
    trial: dict


def collect_exploration(model, rule, *, seed, steps=12, opponent_profile='steady',assessment_steps=None):
    from .meta_exploration_rollout import run_meta_trial
    teachers=[]
    def factory(*,n,seed,student):
        teacher=NeuralExplorer(model=model,n=n,seed=seed)
        teachers.append(teacher)
        return teacher
    trial=run_meta_trial(rule,factory,seed=seed,steps=steps,opponent_profile=opponent_profile,assessment_steps=assessment_steps)
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


def meta_targets(rollouts, objective='retained_gain'):
    """No exploration rewards here: a terminal downstream target for each action."""
    import torch
    if not rollouts:
        raise ValueError('meta batch must be nonempty')
    if objective not in ('retained_gain','assessment_return'):raise ValueError('unknown exploration objective')
    values=[(r.meta_reward if objective=='retained_gain' else r.trial['assessment_mean']) for r in rollouts for _ in r.actions]
    result=torch.tensor(values,dtype=torch.float32)
    if not result.numel() or not torch.isfinite(result).all():
        raise ValueError('meta targets must be nonempty and finite')
    return result


def train_meta(config, manifest=None, *, opponent_suite='q_learning', checkpoint_path=None,
               resume=False, stop_after_update=None, checkpoint_every=1, checkpoint_callback=None):
    import torch
    from time import perf_counter
    from copy import deepcopy
    from .devices import resolve_device
    from .meta_batch import collect_batch, replay_batch
    from .meta_training_state import context_digest, save_training_state, restore_training_state
    from .action_set import ActionSetActorCritic
    from .ppo import ppo_loss
    from .meta_exploration_rollout import OPPONENT_SUITES
    if opponent_suite not in OPPONENT_SUITES:
        raise ValueError('unknown opponent suite')
    profiles=OPPONENT_SUITES[opponent_suite]['train']
    manifest=build_manifest() if manifest is None else manifest
    rules=[r for r in manifest['rules'] if r['split']=='train']
    if config.train_rule_limit is not None:
        if config.train_rule_limit>len(rules):raise ValueError('training rule limit exceeds catalog')
        if manifest.get('catalog')=='probe_template' and config.train_rule_limit%2:raise ValueError('probe training limits must preserve theta pairs')
        rules=rules[:config.train_rule_limit]
    from .rules import compile_rule
    rule_tensors={entry['id']:compile_rule(decode_rule(entry['rule'])) for entry in rules}
    device=resolve_device(config.device)
    positive_integer(checkpoint_every, 'checkpoint_every')
    if stop_after_update is not None: positive_integer(stop_after_update, 'stop_after_update')
    if resume and checkpoint_path is None: raise ValueError('resume requires checkpoint_path')
    model=ActionSetActorCritic(hidden_size=config.hidden_size,seed=config.seed).to(device)
    initial=deepcopy(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    rng=random.Random(config.seed)
    history=[]
    context=context_digest(asdict(config),manifest,opponent_suite,device)
    if resume:
        history=restore_training_state(checkpoint_path,model=model,initial=initial,
                                       optimizer=optimizer,sampler=rng,context=context)
    limit=config.updates if stop_after_update is None else min(config.updates,stop_after_update)
    timing={'collection_seconds':0.0,'optimization_seconds':0.0,'updates_this_invocation':0}
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for update in range(len(history)+1,limit+1):
        synchronize();started=perf_counter()
        specifications=[]
        for _ in range(config.trials_per_update):
            entry=rules[rng.randrange(len(rules))]
            profile=profiles[rng.randrange(len(profiles))]
            trial_seed=rng.getrandbits(63)
            specifications.append((entry,profile,trial_seed))
        batch=[]
        for offset in range(0,len(specifications),config.batch_size):
            group=specifications[offset:offset+config.batch_size]
            if config.batch_size==1:
                entry,profile,trial_seed=group[0]
                batch.append(collect_exploration(model,rule_tensors[entry['id']],seed=trial_seed,
                                               steps=config.steps,opponent_profile=profile,assessment_steps=config.assessment_steps))
            else:
                batch.extend(collect_batch(model,[(rule_tensors[e['id']],seed,p) for e,p,seed in group],steps=config.steps,assessment_steps=config.assessment_steps))
        provenance=[{'rule_id':entry['id'],'opponent_profile':profile,'seed':trial_seed,
                     'meta_reward':rollout.meta_reward,'assessment_mean':rollout.trial['assessment_mean'],
                     'erased_mean':rollout.trial['erased_mean']}
                    for (entry,profile,trial_seed),rollout in zip(specifications,batch)]
        synchronize();timing['collection_seconds']+=perf_counter()-started;started=perf_counter()
        target=meta_targets(batch,config.objective).to(device)
        old_values=torch.tensor([v for r in batch for v in r.old_values],dtype=torch.float32,device=device)
        advantage=target-old_values
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        old_logs=torch.tensor([v for r in batch for v in r.old_log_probs],dtype=torch.float32,device=device)
        metrics=[]
        for _ in range(config.epochs):
            replay=[replay_exploration(model,r) for r in batch] if config.batch_size==1 else [
                replay_batch(model,batch[i:i+config.batch_size]) for i in range(0,len(batch),config.batch_size)]
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
        synchronize();timing['optimization_seconds']+=perf_counter()-started
        timing['updates_this_invocation']+=1
        history.append({'update':update,**{k:statistics.mean(x[k] for x in metrics) for k in metrics[0]},
                        'mean_meta_reward':statistics.mean(r.meta_reward for r in batch),
                        'mean_assessment':statistics.mean(r.trial['assessment_mean'] for r in batch),
                        'mean_teacher_target':float(target.mean()),
                        'mean_exploration':statistics.mean(r.trial['exploration_mean'] for r in batch),
                        'opponent_policy_l1_change':statistics.mean(r.trial['opponent_final']['policy_l1_change'] for r in batch),
                        'trials':provenance})
        if checkpoint_path is not None and (update%checkpoint_every==0 or update==limit):
            save_training_state(checkpoint_path,model=model,initial=initial,optimizer=optimizer,
                                sampler=rng,history=history,context=context)
            if checkpoint_callback is not None:checkpoint_callback()
        if update==1 or update%10==0 or update==config.updates:
            print(f'[seed={config.seed} update={update}] meta={history[-1]["mean_meta_reward"]:.4f} assessment={history[-1]["mean_assessment"]:.4f}',file=sys.stderr,flush=True)
    return {'initial_model':initial,'model':model,'history':history,'config':asdict(config),
            'completed':len(history)==config.updates,'timing':timing}


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
            'opponent_nonuniform_action_fraction':sum(r['opponent_nonuniform_action_events'] for r in rows)/sum(len(r['exploration'])+len(r['assessment']) for r in rows)}
    result['assessment_early']=statistics.mean(result['assessment_curve'][:min(3,len(result['assessment_curve']))])
    result['assessment_late']=statistics.mean(result['assessment_curve'][-min(3,len(result['assessment_curve'])):])
    result['total_gameplay_mean']=statistics.mean(((r['exploration_mean']+r['assessment_mean'])/2 if len(r['exploration'])==len(r['assessment']) else (len(r['exploration'])*r['exploration_mean']+len(r['assessment'])*r['assessment_mean'])/(len(r['exploration'])+len(r['assessment']))) for r in rows)
    curve=result['assessment_curve']
    budgets=sorted({min(k,len(curve)) for k in (1,3,6,len(curve))})
    result['assessment_prefix_auc']={str(k):statistics.mean(curve[:k]) for k in budgets}
    center=(len(curve)-1)/2
    denominator=sum((i-center)**2 for i in range(len(curve)))
    result['assessment_slope']=sum((i-center)*v for i,v in enumerate(curve))/denominator if denominator else None
    return result


def evaluate_meta(teacher_factory, tasks, *, steps=12,reps=2,seed=918273,batch_size=1,assessment_steps=None):
    from .meta_exploration_rollout import run_meta_trial
    positive_integer(reps,'reps')
    positive_integer(batch_size,'batch_size')
    positive_integer(steps,'steps')
    if validate_seed(seed) is None:
        raise TypeError('evaluation seed must be explicit')
    tasks=tuple(tasks)
    if not tasks:
        raise ValueError('evaluation tasks must be nonempty')
    rows=[]
    jobs=[]
    for task in tasks:
        for rep in range(reps):
            trial_seed=int.from_bytes(hashlib.sha256(f'{seed}/{task["id"]}/{rep}'.encode()).digest()[:8],'big') % 2**63
            jobs.append((task,rep,trial_seed))
    model=getattr(teacher_factory,'model',None)
    for offset in range(0,len(jobs),batch_size):
        group=jobs[offset:offset+batch_size]
        if model is not None and batch_size>1:
            from .meta_batch import collect_batch
            trials=[r.trial for r in collect_batch(model,[(decode_rule(t['rule']),s,t['opponent_profile'])
                                                     for t,rep,s in group],steps=steps,assessment_steps=assessment_steps)]
        else:
            trials=[run_meta_trial(decode_rule(t['rule']),teacher_factory,seed=s,
                                   steps=steps,opponent_profile=t['opponent_profile'],assessment_steps=assessment_steps) for t,rep,s in group]
        for (task,rep,trial_seed),trial in zip(group,trials):
            rows.append({'task_id':task['id'],'rule_id':task['rule_id'],'split':task['split'],'rep':rep,
                         **({'family':task['family']} if 'family' in task else {}),
                         **({'seen_rule':task['seen_in_training']} if 'seen_in_training' in task else {}),**trial})
    result={'summary':summarize_meta(rows),'trials':rows}
    families=sorted({r['family'] for r in rows if 'family' in r})
    if families:
        result['by_family']={f:summarize_meta([r for r in rows if r.get('family')==f]) for f in families}
    return result


def neural_factory(model):
    def factory(*,n,seed,student):
        return NeuralExplorer(model=model,n=n,seed=seed)
    factory.model=model
    return factory


def _write(path,value):
    from .run_files import write_json_exclusive
    write_json_exclusive(path,value)


def run_meta_experiment(output_dir, *, seeds=(7,17,27),updates=40,trials_per_update=8,steps=12,hidden_size=32,reps=2,
                        catalog='logical',opponent_suite='q_learning',device='cpu',batch_size=1,resume=False,
                        stop_after_update=None,checkpoint_every=10,backup_dir=None,assessment_steps=None,
                        catalog_seed=None,train_rules=8,fresh_rules=8,structural_rules=8,manifest_path=None,train_rule_limit=None,objective='retained_gain',
                        n=3,diversity='broad'):
    import torch
    from .recurrent import save_checkpoint,load_checkpoint
    from .devices import resolve_device
    from .run_files import RunMirror
    from time import perf_counter
    from .meta_exploration_rollout import OPPONENT_PROFILES,OPPONENT_SUITES
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(validate_seed(s) is None for s in seeds):
        raise ValueError('seeds must be unique explicit nonempty seeds')
    for k,v in (('updates',updates),('trials_per_update',trials_per_update),('steps',steps),('hidden_size',hidden_size),('reps',reps),('batch_size',batch_size),('checkpoint_every',checkpoint_every)):
        positive_integer(v,k)
    if objective not in ('retained_gain','assessment_return'):raise ValueError('unknown exploration objective')
    if catalog not in ('logical','relational','probe_template','diverse_relational') or opponent_suite not in OPPONENT_SUITES:
        raise ValueError('unknown catalog or opponent suite')
    n=validate_n(n)
    if n not in (3,4,5):raise ValueError('meta catalogs support n=3, 4 or 5')
    if catalog in ('logical','probe_template') and n!=3:
        raise ValueError('logical and probe_template meta catalogs currently require n=3')
    if diversity not in ('focused','broad'):raise ValueError('unknown diversity setting')
    if stop_after_update is not None:positive_integer(stop_after_update,'stop_after_update')
    assessment_steps=steps if assessment_steps is None else positive_integer(assessment_steps,'assessment_steps')
    selected_device=resolve_device(device)
    p=Path(output_dir)
    mirror=RunMirror(p,backup_dir)
    if not resume and mirror.target and mirror.target.exists() and any(mirror.target.iterdir()):
        raise ValueError('backup directory already contains a run; choose a unique path')
    if resume:
        if not (p/'protocol.json').is_file():raise ValueError('resume requires an existing run protocol')
    else:
        p.mkdir(parents=True,exist_ok=False)
    sources=p/'sources'
    hashes={file.name:hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(Path(__file__).parent.glob('*.py'))}
    if not resume:
        sources.mkdir()
        for file in sorted(Path(__file__).parent.glob('*.py')):(sources/file.name).write_bytes(file.read_bytes())
    if catalog_seed is None:
        catalog_seed={'probe_template':20261116,'diverse_relational':20261118}.get(catalog,20261114)
    if manifest_path is not None:
        manifest=json.loads(Path(manifest_path).read_text())
        if catalog=='probe_template':
            from .probe_templates import validate_probe_manifest
            validate_probe_manifest(manifest)
        elif catalog=='relational':
            from .relational_catalog import validate_relational_manifest
            validate_relational_manifest(manifest)
        elif catalog=='diverse_relational':
            from .diverse_catalog import validate_diverse_manifest
            validate_diverse_manifest(manifest)
            if manifest['diversity']!=diversity:raise ValueError('manifest diversity does not match requested diversity')
        else:
            from .generated_tasks import validate_manifest
            validate_manifest(manifest)
        catalog_seed=manifest['seed']
    elif catalog=='probe_template':
        from .probe_templates import build_probe_manifest,exact_information_audit
        manifest=build_probe_manifest(seed=catalog_seed,train_rules=train_rules,fresh_rules=fresh_rules,structural_rules=structural_rules)
        if not resume:_write(p/'information_audit.json',{
            'positive':exact_information_audit(),
            'short_horizon':exact_information_audit(horizon=1),
            'dominant_safe':exact_information_audit(safe_reward=4),
            'scope':'raw reward fixed-opponent controls; actual MA policies learn'})
    elif catalog=='relational':
        from .relational_catalog import build_relational_manifest,relational_distances
        manifest=build_relational_manifest(n=n,seed=catalog_seed)
        if not resume:_write(p/'distances.json',relational_distances(manifest))
    elif catalog=='diverse_relational':
        from .diverse_catalog import build_diverse_manifest
        manifest=build_diverse_manifest(n=n,seed=catalog_seed,train_rules=train_rules,
                                       fresh_rules=fresh_rules,structural_rules=structural_rules,diversity=diversity)
    else:
        manifest=build_manifest(seed=catalog_seed)
    if manifest['n']!=n:raise ValueError('manifest n does not match requested n')
    selected_training=[r['id'] for r in manifest['rules'] if r['split']=='train']
    if train_rule_limit is not None:
        positive_integer(train_rule_limit,'train_rule_limit')
        if train_rule_limit>len(selected_training):raise ValueError('training limit exceeds catalog')
        if catalog=='probe_template' and train_rule_limit%2:raise ValueError('probe training limits must preserve theta pairs')
        selected_training=selected_training[:train_rule_limit]
    if not resume:_write(p/'manifest.json',manifest)
    elif manifest!=json.loads((p/'manifest.json').read_text()):raise ValueError('resume catalog mismatch')
    protocol={'version':2,'catalog':catalog,'catalog_seed':catalog_seed,'training_rule_ids':selected_training,'train_rule_limit':train_rule_limit,'catalog_counts':dict(Counter(r['split'] for r in manifest['rules'])),'opponent_suite':opponent_suite,'seeds':list(seeds),'updates':updates,'trials_per_update':trials_per_update,
              'steps':steps,'assessment_steps':assessment_steps,'hidden_size':hidden_size,'reps':reps,'n':n,'device':str(selected_device),'batch_size':batch_size,
              'torch_version':str(torch.__version__),'torch_threads':torch.get_num_threads(),
              'python_version':sys.version,'source_sha256':hashes,
              'opponent_profiles':OPPONENT_PROFILES,'training_profiles':list(OPPONENT_SUITES[opponent_suite]['train']),
              'heldout_profile':OPPONENT_SUITES[opponent_suite]['heldout'],
              'student':{'algorithm':'tabular Q','alpha':.5,'gamma':.95,'assessment_epsilon':0},
              'objective':objective,
              'objective_definition':('assessment_mean minus paired erased-student assessment_mean' if objective=='retained_gain' else 'absolute assessment_mean')+'; no exploration reward',
              'inner_training_gameplay_steps_per_seed':updates*trials_per_update*(steps+assessment_steps),
              'inner_training_control_steps_per_seed':updates*trials_per_update*assessment_steps,
              'teacher_training_decisions_per_seed':updates*trials_per_update*steps,
              'selection':'fixed final checkpoint; final task results never select weights',
              'scope':'one learner algorithm and bounded generated distribution, not general meta-exploration proof'}
    if catalog=='diverse_relational':protocol['diversity']=manifest['diversity']
    if resume:
        saved=json.loads((p/'protocol.json').read_text())
        if saved!=protocol:raise ValueError('resume protocol/software/source mismatch; use original run settings')
        if mirror.target and (mirror.target/'protocol.json').is_file():
            if json.loads((mirror.target/'protocol.json').read_text())!=protocol:
                raise ValueError('backup protocol differs from resumed run')
        if (p/'experiment.json').exists():
            result=load_meta_experiment(p/'experiment.json')
            (p/'report.html').write_text(render_meta_report(result),encoding='utf-8');mirror()
            return result
    else:_write(p/'protocol.json',protocol)
    mirror()
    tasks={s:meta_tasks(manifest,s,opponent_suite=opponent_suite) for s in SPLITS}
    if train_rule_limit is not None:
        tasks={s:tuple({**t,'seen_in_training':t['rule_id'] in selected_training} for t in rows) for s,rows in tasks.items()}
    options={'steps':steps,'reps':reps,'batch_size':batch_size,'assessment_steps':assessment_steps}
    baselines={}
    reference_factories=[('random',RandomExplorer),('epsilon_greedy',EpsilonExplorer),('repeat',RepeatExplorer),('balanced',BalancedExplorer)]
    if catalog in ('relational','probe_template','diverse_relational'): reference_factories.append(('change',ChangeExplorer))
    if catalog=='probe_template':reference_factories.append(('systematic',SystematicExplorer))
    if resume and (p/'baselines.json').exists():
        baselines=json.loads((p/'baselines.json').read_text())
    else:
        for name,factory in reference_factories:
            print('[baseline '+name+']',file=sys.stderr,flush=True)
            baselines[name]={s:evaluate_meta(factory,t,**options) for s,t in tasks.items()}
        _write(p/'baselines.json',baselines)
    mirror()
    record={'version':2,'protocol':protocol,'baselines':baselines,'runs':[],'comparisons':{},
            'rule_catalog':manifest['rules'] if catalog in ('relational','probe_template','diverse_relational') else []}
    for seed in seeds:
        d=p/f'seed{seed}'
        if resume and (d/'evaluation.json').is_file():
            record['runs'].append(json.loads((d/'evaluation.json').read_text()));continue
        d.mkdir(exist_ok=True)
        checkpoint=d/'training_state.pt'
        started=perf_counter()
        result=train_meta(MetaConfig(seed=seed,updates=updates,trials_per_update=trials_per_update,
                                    steps=steps,hidden_size=hidden_size,device=str(selected_device),batch_size=batch_size,assessment_steps=assessment_steps,train_rule_limit=train_rule_limit,objective=objective),
                          manifest,opponent_suite=opponent_suite,checkpoint_path=checkpoint,
                          resume=resume and checkpoint.exists(),stop_after_update=stop_after_update,
                          checkpoint_every=checkpoint_every,checkpoint_callback=mirror)
        timing={**result['timing'],'wall_seconds':perf_counter()-started,'device':str(selected_device)}
        timing['gameplay_steps']=timing['updates_this_invocation']*trials_per_update*(steps+assessment_steps)
        timing['control_steps']=timing['updates_this_invocation']*trials_per_update*assessment_steps
        timing['gameplay_steps_per_second']=timing['gameplay_steps']/max(timing['wall_seconds'],1e-12)
        # Timing is operational metadata, separate from deterministic learning history.
        from time import time_ns
        _write(d/f'timing-{time_ns()}.json',timing)
        if not result['completed']:
            mirror()
            return {'paused':True,'seed':seed,'completed_updates':len(result['history']),
                    'checkpoint':str(checkpoint)}
        if not (d/'initial.pt').exists():save_checkpoint(result['initial_model'],d/'initial.pt')
        if not (d/'policy.pt').exists():save_checkpoint(result['model'],d/'policy.pt')
        initial,trained=load_checkpoint(d/'initial.pt',device=str(selected_device)),load_checkpoint(d/'policy.pt',device=str(selected_device))
        evaluation={name:{s:evaluate_meta(neural_factory(model),t,**options) for s,t in tasks.items()}
                    for name,model in (('initial',initial),('trained',trained))}
        row={'seed':seed,'config':result['config'],'history':result['history'],'evaluation':evaluation,
             'checkpoint':f'seed{seed}/policy.pt','initial_checkpoint':f'seed{seed}/initial.pt'}
        _write(d/'evaluation.json',row);record['runs'].append(row);mirror()
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
    mirror()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    parser.add_argument('--catalog',choices=('logical','relational','probe_template','diverse_relational'),default='logical')
    parser.add_argument('--n',type=int,default=3,help='action count (3, 4 or 5; catalog-specific limits apply)')
    parser.add_argument('--diversity',choices=('focused','broad'),default='broad')
    parser.add_argument('--objective',choices=('retained_gain','assessment_return'),default='retained_gain')
    parser.add_argument('--manifest',type=Path,help='validated common catalog; generation counts are ignored')
    parser.add_argument('--train-rule-limit',type=int,help='nested prefix subset of the common training rules')
    parser.add_argument('--catalog-seed',type=int)
    parser.add_argument('--train-rules',type=int,default=8)
    parser.add_argument('--fresh-rules',type=int,default=8)
    parser.add_argument('--structural-rules',type=int,default=8)
    parser.add_argument('--opponent-suite',choices=('q_learning','mixed_learning'),default='q_learning')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--assessment-steps',type=int,help='assessment/control horizon; default equals --steps')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-update',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=10)
    parser.add_argument('--backup-dir',type=Path)
    for name,value in (('updates',40),('trials-per-update',8),('steps',12),('hidden-size',32),('reps',2)):
        parser.add_argument('--'+name,type=int,default=value)
    args=parser.parse_args()
    try:
        result=run_meta_experiment(args.output_dir,seeds=args.seeds,updates=args.updates,trials_per_update=args.trials_per_update,
                            steps=args.steps,hidden_size=args.hidden_size,reps=args.reps,
                            catalog=args.catalog,opponent_suite=args.opponent_suite,device=args.device,batch_size=args.batch_size,
                            resume=args.resume,stop_after_update=args.stop_after_update,checkpoint_every=args.checkpoint_every,backup_dir=args.backup_dir,assessment_steps=args.assessment_steps,catalog_seed=args.catalog_seed,train_rules=args.train_rules,
                            fresh_rules=args.fresh_rules,structural_rules=args.structural_rules,manifest_path=args.manifest,train_rule_limit=args.train_rule_limit,objective=args.objective,n=args.n,diversity=args.diversity)
        if result.get('paused'):print(json.dumps(result))
    except (TypeError,ValueError,OSError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__=='__main__': main()
