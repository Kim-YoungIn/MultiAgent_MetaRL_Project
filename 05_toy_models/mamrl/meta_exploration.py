"""Learn exploration from downstream adaptation against ONLINE-learning opponents."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
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


SPLITS=('train_rule_reference','fresh_rules','structural','new_learning_profile')


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


def train_meta(config, manifest=None):
    import torch
    from copy import deepcopy
    from .action_set import ActionSetActorCritic
    from .ppo import ppo_loss
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
            profile=('steady','rapid')[rng.randrange(2)]
            trial_seed=rng.getrandbits(63)
            rollout=collect_exploration(model,rule_from_dict(entry['rule']),seed=trial_seed,
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


def meta_tasks(manifest, split):
    if split not in SPLITS:
        raise ValueError('unknown meta-exploration split')
    source='train' if split in ('train_rule_reference','new_learning_profile') else split
    profiles=('heldout',) if split=='new_learning_profile' else ('steady','rapid')
    return tuple({'id':r['id']+'/'+profile,'rule_id':r['id'],'rule':r['rule'],
                  'opponent_profile':profile,'split':split}
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
            'opponent_strategy_changed_trials':sum(r['opponent_final']['policy_l1_change']>1e-12 for r in rows)}
    result['assessment_early']=statistics.mean(result['assessment_curve'][:min(3,len(result['assessment_curve']))])
    result['assessment_late']=statistics.mean(result['assessment_curve'][-min(3,len(result['assessment_curve'])):])
    result['total_gameplay_mean']=statistics.mean((r['exploration_mean']+r['assessment_mean'])/2 for r in rows)
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
            trial=run_meta_trial(rule_from_dict(task['rule']),teacher_factory,seed=trial_seed,
                                 steps=steps,opponent_profile=task['opponent_profile'])
            rows.append({'task_id':task['id'],'rule_id':task['rule_id'],'split':task['split'],'rep':rep,**trial})
    return {'summary':summarize_meta(rows),'trials':rows}


def neural_factory(model):
    def factory(*,n,seed,student):
        return NeuralExplorer(model=model,n=n,seed=seed)
    return factory


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def render_meta_report(record):
    esc=lambda x:html.escape(str(x),quote=True)
    num=lambda x:'—' if x is None else f'{x:.4f}'
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>메타 탐색과 학습하는 상대</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:28px auto;padding:0 18px;background:#f5f7fb;color:#152538}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{border:1px solid #cbd4df;padding:8px;text-align:left}pre{white-space:pre-wrap}svg{background:white;width:100%}</style>',
           '<h1>메타 탐색 → 적응 학습 → 학습하는 상대</h1>',
           '<p>탐색자가 만든 경험을 같은 온라인 Q 학습기가 활용합니다. 상대도 자기 경험으로 Q와 전략을 갱신합니다.</p>',
           f'<p>탐색 {record["protocol"]["steps"]}회 + 실제 평가 {record["protocol"]["steps"]}회 + 경험 제거 대조 {record["protocol"]["steps"]}회. 탐색자의 meta reward는 평가−경험 제거 대조입니다. 탐색 중 보상을 더하지 않습니다.</p>',
           '<p><a href="experiment.json">전체 원자료</a> · <a href="protocol.json">프로토콜</a> · <a href="manifest.json">생성 규칙</a></p>']
    labels={'initial':'학습 전 탐색','trained':'메타 학습 탐색','random':'랜덤 탐색','epsilon_greedy':'ε-greedy 탐색','repeat':'반복 탐색'}
    colors=('#8a50b0','#0068b5','#777','#d26900','#268344')
    for split in SPLITS:
        parts.append('<h2>'+esc(split)+'</h2><table><tr><th>탐색 전략</th><th>경험 유지 이득 G</th><th>실제 후속 성능</th><th>경험 제거 성능</th><th>탐색 보상</th><th>실제 게임 전체</th><th>상대 전략 변화</th></tr>')
        curves={}
        for method in labels:
            if method in record['baselines']:
                stats=record['baselines'][method][split]['summary']
            else:
                samples=[r['evaluation'][method][split]['summary'] for r in record['runs']]
                stats={k:statistics.mean(s[k] for s in samples) for k in ('meta_reward','assessment_mean','erased_mean','exploration_mean','total_gameplay_mean','mean_opponent_policy_l1_change')}
                stats['assessment_curve']=[statistics.mean(s['assessment_curve'][i] for s in samples) for i in range(len(samples[0]['assessment_curve']))]
            curves[method]=stats['assessment_curve']
            parts.append('<tr><td>'+labels[method]+'</td>'+''.join('<td>'+num(stats[k])+'</td>' for k in ('meta_reward','assessment_mean','erased_mean','exploration_mean','total_gameplay_mean','mean_opponent_policy_l1_change'))+'</tr>')
        parts.append('</table><svg viewBox="0 0 700 160" role="img" aria-label="후속 학습 성능 곡선"><line x1="20" x2="680" y1="80" y2="80" stroke="#bbb"/>')
        for (method,curve),color in zip(curves.items(),colors):
            pts=' '.join(f'{20+660*i/max(1,len(curve)-1):.2f},{80-65*v:.2f}' for i,v in enumerate(curve))
            parts.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="2"><title>{labels[method]}</title></polyline>')
        parts.append('</svg><p>'+ ' · '.join(f'<span style="color:{c}">{labels[m]}</span>' for m,c in zip(curves,colors))+'</p>')
        parts.append('<details><summary>학습 seed별 결과</summary><pre>'+esc(json.dumps(record['comparisons'][split],ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<h2>구성과 측정의 한계</h2><ul><li>G는 경험을 보존한 Q와 지운 Q의 성능 차이입니다. 순수 정보량이 아니며 학습 능력·상태 방문·상대에 대한 영향도 포함합니다.</li></ul>')
    parts.append('<p>초기·최종 탐색 가중치는 평가 중 고정하고 학생과 상대의 Q는 갱신합니다. 최종 평가 규칙을 메타 학습에 쓰지 않습니다. 상대 초기 조건은 맞추되 탐색에 반응해 다르게 학습하는 것은 허용합니다.</p>')
    parts.append('<p>새 학습 profile은 같은 Q 학습법의 미사용 파라미터이며, 새로운 학습 알고리즘이 아닙니다. 생성 분포도 현재 행동 일치 관계를 공통으로 포함하는 제한된 분포입니다. 제한된 seed의 기술 통계이며 H1/H2 확증이 아닙니다.</p></html>')
    return ''.join(parts)


def run_meta_experiment(output_dir, *, seeds=(7,17,27),updates=40,trials_per_update=8,steps=12,hidden_size=32,reps=2):
    import torch
    from .recurrent import save_checkpoint,load_checkpoint
    from .meta_exploration_rollout import OPPONENT_PROFILES
    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(validate_seed(s) is None for s in seeds):
        raise ValueError('seeds must be unique explicit nonempty seeds')
    for k,v in (('updates',updates),('trials_per_update',trials_per_update),('steps',steps),('hidden_size',hidden_size),('reps',reps)):
        positive_integer(v,k)
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=False)
    sources=p/'sources';sources.mkdir()
    hashes={}
    for file in sorted(Path(__file__).parent.glob('*.py')):
        data=file.read_bytes();(sources/file.name).write_bytes(data)
        hashes[file.name]=hashlib.sha256(data).hexdigest()
    manifest=build_manifest()
    _write(p/'manifest.json',manifest)
    protocol={'version':1,'seeds':list(seeds),'updates':updates,'trials_per_update':trials_per_update,
              'steps':steps,'hidden_size':hidden_size,'reps':reps,'n':3,'device':'cpu',
              'torch_version':str(torch.__version__),'source_sha256':hashes,
              'opponent_profiles':OPPONENT_PROFILES,'training_profiles':['steady','rapid'],
              'student':{'algorithm':'tabular Q','alpha':.5,'gamma':.95,'assessment_epsilon':0},
              'objective':'assessment_mean minus paired erased-student assessment_mean; no exploration reward',
              'inner_training_gameplay_steps_per_seed':updates*trials_per_update*steps*2,
              'inner_training_control_steps_per_seed':updates*trials_per_update*steps,
              'teacher_training_decisions_per_seed':updates*trials_per_update*steps,
              'selection':'fixed final checkpoint; final task results never select weights',
              'scope':'one learner algorithm and bounded generated distribution, not general meta-exploration proof'}
    _write(p/'protocol.json',protocol)
    tasks={s:meta_tasks(manifest,s) for s in SPLITS}
    options={'steps':steps,'reps':reps}
    baselines={}
    for name,factory in (('random',RandomExplorer),('epsilon_greedy',EpsilonExplorer),('repeat',RepeatExplorer)):
        print('[baseline '+name+']',file=sys.stderr,flush=True)
        baselines[name]={s:evaluate_meta(factory,t,**options) for s,t in tasks.items()}
    _write(p/'baselines.json',baselines)
    record={'version':1,'protocol':protocol,'baselines':baselines,'runs':[],'comparisons':{}}
    for seed in seeds:
        result=train_meta(MetaConfig(seed=seed,updates=updates,trials_per_update=trials_per_update,steps=steps,hidden_size=hidden_size),manifest)
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
            for reference in ('initial','random','epsilon_greedy','repeat'):
                diffs=[]
                for row in record['runs']:
                    target=row['evaluation']['trained'][s]
                    control=row['evaluation']['initial'][s] if reference=='initial' else baselines[reference][s]
                    a={(t['task_id'],t['rep']):t for t in target['trials']}
                    b={(t['task_id'],t['rep']):t for t in control['trials']}
                    assert set(a)==set(b)
                    diffs.append(statistics.mean(a[k][metric]-b[k][metric] for k in a))
                record['comparisons'][s][metric]['minus_'+reference]={'seed_differences':diffs,'mean':statistics.mean(diffs)}
    _write(p/'experiment.json',record)
    (p/'report.html').write_text(render_meta_report(record),encoding='utf-8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    for name,value in (('updates',40),('trials-per-update',8),('steps',12),('hidden-size',32),('reps',2)):
        parser.add_argument('--'+name,type=int,default=value)
    args=parser.parse_args()
    try:
        run_meta_experiment(args.output_dir,seeds=args.seeds,updates=args.updates,trials_per_update=args.trials_per_update,
                            steps=args.steps,hidden_size=args.hidden_size,reps=args.reps)
    except (TypeError,ValueError,OSError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__=='__main__': main()
