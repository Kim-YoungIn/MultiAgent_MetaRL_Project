"""Run a complete generated-game, controlled-opponent transfer pilot on CPU."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import sys

from .common import positive_integer, validate_n, validate_seed
from .environment import MemoryOneEnv
from .generated_tasks import (CONDITIONS, SPLITS, build_manifest, validate_manifest,
                              tasks_from_manifest, distance_manifest)
from .generated_information import information_witness
from .opponents import make_opponent
from .rules import compile_rule


def opponent_probabilities(kind, n, previous, other_positive=False, start=False):
    """P1 distribution in GLOBAL action coordinates; researcher/reference only."""
    own, other = previous
    if kind == 'random' or (kind == 'win_stay' and start):
        return (1/n,) * n
    if kind == 'copy_previous':
        return tuple(float(b == own) for b in range(n))
    if kind == 'avoid_previous':
        return tuple(0.0 if b == own else 1/(n-1) for b in range(n))
    if kind == 'win_stay':
        if other_positive:
            return tuple(float(b == other) for b in range(n))
        return tuple(0.0 if b == other else 1/(n-1) for b in range(n))
    raise ValueError('unsupported reference opponent')


def best_response_value(task, *, max_steps=12):
    """Known-task finite-horizon DP expectation per step, averaged over reset.

    This reference additionally knows opponent type and horizon; it is neither
    learner-achievable evidence nor an equilibrium/Nash-regret calculation.
    """
    positive_integer(max_steps,'max_steps')
    tensor, n = compile_rule(task.rule), task.rule.n

    @lru_cache(None)
    def value(left, p0, p1, positive, start):
        if not left:
            return 0.0
        probs = opponent_probabilities(task.opponent_kind,n,(p0,p1),positive,start)
        return max(sum(prob*(tensor[0,p0,p1,a,b] + value(left-1,a,b,
                           tensor[1,p0,p1,a,b]>0,False))
                       for b,prob in enumerate(probs) if prob) for a in range(n))
    return sum(value(max_steps,a,b,False,True) for a in range(n) for b in range(n)) / (n*n*max_steps)


class SimpleLearner:
    def __init__(self, *, mode, family, n, seed):
        self.mode, self.n = mode, n
    def act(self, obs):
        return obs.previous_own_action if self.mode=='repeat' else obs.previous_opponent_action
    def observe(self,*args): pass
    def transition(self,*args): pass
    def episode_boundary(self,*args): pass
    def reset_memory(self): pass


class PosteriorLearner:
    """Strong informed finite-catalog prior, NOT a learned equal-prior baseline.

    Each hypothesis contains a full rule and opponent model, but the TRUE task
    identity is never supplied. Greedy posterior reward maximization does not
    claim Bayes-optimal or actively optimal exploration.
    """
    def __init__(self, *, hypotheses, family, n, seed):
        self.hypotheses = tuple(hypotheses)
        self.n, self.rng = n, random.Random(seed)
        self.tensors = tuple(compile_rule(t.rule) for t in self.hypotheses)
        self.reset_memory()

    def reset_memory(self):
        self.weights = [1/len(self.hypotheses)] * len(self.hypotheses)
        self.other_positive = [False] * len(self.hypotheses)

    def act(self, observation):
        p0,p1 = observation.previous_own_action,observation.previous_opponent_action
        scores = [0.0]*self.n
        for i,(task,tensor) in enumerate(zip(self.hypotheses,self.tensors)):
            if not self.weights[i]:
                continue
            probs = opponent_probabilities(task.opponent_kind,self.n,(p0,p1),self.other_positive[i],observation.episode_start_flag)
            for a in range(self.n):
                scores[a] += self.weights[i] * sum(prob*tensor[0,p0,p1,a,b] for b,prob in enumerate(probs))
        top = max(scores)
        return self.rng.choice([a for a,s in enumerate(scores) if abs(s-top)<1e-10])

    def observe(self, observation, action, reward, final_next_obs):
        p0,p1 = observation.previous_own_action,observation.previous_opponent_action
        b = final_next_obs.previous_opponent_action
        for i,(task,tensor) in enumerate(zip(self.hypotheses,self.tensors)):
            if not self.weights[i]:
                continue
            likelihood = opponent_probabilities(task.opponent_kind,self.n,(p0,p1),self.other_positive[i],observation.episode_start_flag)[b]
            self.weights[i] *= likelihood if abs(tensor[0,p0,p1,action,b]-reward)<1e-9 else 0.0
            self.other_positive[i] = tensor[1,p0,p1,action,b]>0
        total = sum(self.weights)
        if total <= 0:
            raise ValueError('experience inconsistent with informed finite-catalog prior')
        self.weights = [w/total for w in self.weights]

    def transition(self,*args): pass
    def episode_boundary(self,*args): pass


def run_task(task, learner_factory, *, seed, episodes_per_trial=2, max_steps=12, reset_memory=False):
    positive_integer(episodes_per_trial,'episodes_per_trial')
    positive_integer(max_steps,'max_steps')
    if validate_seed(seed) is None:
        raise TypeError('trial seed must be explicit')
    if type(reset_memory) is not bool:
        raise TypeError('reset_memory must be bool')
    seeds = random.Random(seed)
    learner_seed = seeds.getrandbits(63)
    environment_seeds = [seeds.getrandbits(63) for _ in range(episodes_per_trial)]
    opponent_seeds = [seeds.getrandbits(63) for _ in range(episodes_per_trial)]
    # Exactly the visible API: no task/opponent/rule metadata in factory arguments.
    learner = learner_factory(family='generated',n=task.rule.n,seed=learner_seed)
    tensor = compile_rule(task.rule)
    envs = [MemoryOneEnv(tensor,max_steps=max_steps) for _ in environment_seeds]
    starts = [e.reset(seed=s) for e,s in zip(envs,environment_seeds)]
    rewards, actions, other_actions, observations_record = [], [], [], []
    episode_returns, episode_other_returns = [], []
    for index,env in enumerate(envs):
        observations = starts[index]
        opponent = make_opponent(task.opponent_kind)
        state = opponent.initial_state(n=task.rule.n,seed=opponent_seeds[index])
        total, other_total = 0.0, 0.0
        for step in range(max_steps):
            own = observations[0]
            a = learner.act(own)
            other = opponent.act(observations[1],state,n=task.rule.n)
            state = other.state
            result = env.step(a,other.action)
            learner.observe(own,a,result.rewards[0],result.observations[0])
            last = step == max_steps-1
            end = last and index == episodes_per_trial-1
            next_action_obs = starts[index+1][0] if last and not end else result.observations[0]
            learner.transition(own,a,result.rewards[0],next_action_obs,end)
            rewards.append(result.rewards[0])
            actions.append(a)
            other_actions.append(other.action)
            observations_record.append(asdict(own))
            total += result.rewards[0]
            other_total += result.rewards[1]
            observations = result.observations
        episode_returns.append(total)
        episode_other_returns.append(other_total)
        if index < episodes_per_trial-1:
            learner.episode_boundary(observations[0])
            if reset_memory:
                learner.reset_memory()
    return {'seed':seed,'learner_seed':learner_seed,'environment_seeds':environment_seeds,
            'opponent_seeds':opponent_seeds,'rewards':rewards,'actions':actions,
            'opponent_actions':other_actions,'observations':observations_record,
            'episode_returns':episode_returns,'opponent_episode_returns':episode_other_returns}


def _summary(trials, steps):
    means = [sum(t['rewards'])/len(t['rewards']) for t in trials]
    episodes = len(trials[0]['episode_returns'])
    return {'trials':len(trials),'mean_reward_per_step':statistics.mean(means),
            'trial_standard_error':statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else None,
            'first_action':statistics.mean(t['rewards'][0] for t in trials),
            'early_reward':statistics.mean(statistics.mean(t['rewards'][:min(3,len(t['rewards']))]) for t in trials),
            'late_reward':statistics.mean(statistics.mean(t['rewards'][-min(3,len(t['rewards'])):]) for t in trials),
            'mean_reward_by_turn':[statistics.mean(t['rewards'][i] for t in trials) for i in range(len(trials[0]['rewards']))],
            'mean_reward_by_episode':[statistics.mean(t['episode_returns'][e]/steps for t in trials) for e in range(episodes)]}


def evaluate_tasks(learner_factory, tasks, *, reps=2, seed=918273, episodes_per_trial=2,max_steps=12,reset_memory=False):
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError('evaluation tasks must be nonempty')
    positive_integer(reps,'reps')
    if validate_seed(seed) is None:
        raise TypeError('evaluation seed must be explicit')
    rows, all_trials = [], []
    for task in tasks:
        trials = []
        for rep in range(reps):
            # Stable by TASK ID, not traversal order or training condition.
            task_seed = int.from_bytes(hashlib.sha256(f'{seed}/{task.task_id}/{rep}'.encode()).digest()[:8],'big') % 2**63
            trials.append(run_task(task,learner_factory,seed=task_seed,episodes_per_trial=episodes_per_trial,
                                   max_steps=max_steps,reset_memory=reset_memory))
        all_trials.extend(trials)
        rows.append({'task_id':task.task_id,'rule_id':task.rule_id,'opponent_kind':task.opponent_kind,
                     'split':task.split,'summary':_summary(trials,max_steps),'trials':trials})
    return {'summary':_summary(all_trials,max_steps),'tasks':rows}


def _write(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def annotate_record(record, manifest):
    """Research-only familiarity labels; no change to evaluation trajectories."""
    by_id={t['id']:t for t in manifest['tasks']}
    for row in record['runs']:
        training=[by_id[i] for i in row['training_task_ids']]
        seen_rules={t['rule_id'] for t in training}
        seen_opponents={t['opponent_kind'] for t in training}
        row['test_familiarity']={}
        for split in SPLITS:
            row['test_familiarity'][split]={t['task_id']:{
                'seen_rule':t['rule_id'] in seen_rules,
                'seen_opponent':t['opponent_kind'] in seen_opponents,
                'seen_pair':t['task_id'] in row['training_task_ids']}
                for t in row['evaluations']['trained'][split]['tasks']}
    record['protocol']['familiarity_note']='combination is pure novel pairing only in high_rules_high_opponents; see per-run seen flags'
    record['protocol']['curve_note']='early/late includes state occupancy changes, not a pure adaptation estimate'
    record['protocol']['standard_error_note']='overall trial_standard_error is pooled descriptive, not learning-seed or stratified Monte Carlo SE'
    record['protocol']['structure_note']='nested changes predicate and reward-expression composition together, not depth alone'
    record['protocol']['witness_selection']='analytically found fixed pair after catalog generation, frozen before neural results; existence example only'
    record['protocol']['generator_bias']='equality-of-current-actions appears in every sampled payoff; bounded distribution, not unrestricted logical games'
    record['protocol']['budget_scope']='decision_budget_per_condition counts optimizer training rollouts; frozen validation and final evaluation are separate'
    return record


def render_report(record):
    esc = lambda value:html.escape(str(value),quote=True)
    def number(value): return '—' if value is None else f'{value:.4f}'
    labels={CONDITIONS[0]:'규칙4 · 상대1',CONDITIONS[1]:'규칙12 · 상대1',
            CONDITIONS[2]:'규칙4 · 상대3',CONDITIONS[3]:'규칙12 · 상대3'}
    def rule_text(node):
        op=node['op']
        if op=='const': return f'{node["value"]:g}'
        if op=='scale': return f'{node["value"]:g} × ({rule_text(node["children"][0])})'
        if op=='add': return ' + '.join('('+rule_text(c)+')' for c in node['children'])
        if op=='if': return '조건 ['+rule_text(node['condition'])+']이면 '+rule_text(node['children'][0])+', 아니면 '+rule_text(node['children'][1])
        if op in ('eq','ne'):
            names={'own':'현재 자기 행동','other':'현재 상대 행동','prev_own':'직전 자기 행동','prev_other':'직전 상대 행동'}
            def operand(value): return names[value] if isinstance(value,str) else f'행동{value}'
            return operand(node['left'])+(' = ' if op=='eq' else ' ≠ ')+operand(node['right'])
        if op=='not': return '아님('+rule_text(node['children'][0])+')'
        return (' 그리고 ' if op=='and' else ' 또는 ').join('('+rule_text(c)+')' for c in node['children'])
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>생성 논리 게임 전이 파일럿</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 18px;background:#f6f8fb;color:#182438}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{padding:8px;border:1px solid #ccd4df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{width:100%;background:white}small{color:#445}</style>',
             '<h1>생성 논리 게임 · 첫 전이 파일럿</h1>',
             '<p>하나의 정책을 게임·상대 분포에서 학습하고, 모든 조건에 같은 미학습 과제를 평가했습니다. H1/H2 성공을 전제하지 않습니다.</p>',
             f'<p>학습 seeds: {esc(record["protocol"]["seeds"])} · 조건당 {record["protocol"]["updates"]} updates · CPU · 실제 생성 규칙24개</p>',
             '<p><a href="experiment.json">평가 원자료</a> · <a href="manifest.json">규칙·분할 manifest</a> · <a href="protocol.json">실행 조건</a> · <a href="distances.json">거리 자료</a></p>',
             '<h2>학습 분포</h2><table><tr><th>조건</th><th>서로 다른 규칙</th><th>상대 종류</th><th>게임×상대 조합</th></tr>']
    for name,stats in record['condition_counts'].items():
        parts.append(f'<tr><td>{esc(labels[name])}</td><td>{stats["rules"]}</td><td>{stats["opponents"]}</td><td>{stats["tasks"]}</td></tr>')
    parts.append('</table><p>optimizer 학습 rollout의 행동 예산은 같습니다. 별도의 고정 가중치 검증·평가 상호작용은 포함하지 않습니다. 과제당 노출·학습-평가 거리·혼합 비율은 달라지므로 다양성 자체의 분리된 인과 효과로 해석하지 않습니다.</p>')
    for split in SPLITS:
        parts.append(f'<h2>{esc(split)}</h2><table><tr><th>학습 조건</th><th>초기</th><th>학습 평균</th><th>학습 seed SD</th><th>첫 행동</th><th>초기3턴</th><th>마지막3턴</th><th>episode 기억 유지−reset</th></tr>')
        for condition in CONDITIONS:
            stats=record['aggregates'][condition][split]
            parts.append('<tr><td>'+esc(labels[condition])+'</td>'+''.join('<td>'+number(stats[k])+'</td>' for k in ('initial','trained','learning_seed_sd','first_action','early_reward','late_reward','memory_delta'))+'</tr>')
        parts.append('</table><p>고정 비교 기준: '+', '.join(esc(name)+' '+number(value[split]['summary']['mean_reward_per_step']) for name,value in record['references'].items())+
                     ' · task-aware DP 기대 상한 '+number(record['oracle_by_split'][split])+'</p>')
        parts.append('<details><summary>조건별 평가 과제의 익숙함</summary><p>seen_rule/seen_opponent가 모두 참이면 본 요소의 새 조합입니다. 다른 조건에서는 새 규칙·새 상대가 함께 포함될 수 있습니다.</p><table><tr><th>조건</th><th>본 규칙/본 상대</th><th>본 규칙/새 상대</th><th>새 규칙/본 상대</th><th>새 규칙/새 상대</th></tr>')
        for condition in CONDITIONS:
            row=next(r for r in record['runs'] if r['condition']==condition)
            flags=list(row.get('test_familiarity',{}).get(split,{}).values())
            cells=[sum(f['seen_rule']==a and f['seen_opponent']==b for f in flags) for a,b in ((True,True),(True,False),(False,True),(False,False))]
            parts.append('<tr><td>'+esc(labels[condition])+'</td>'+''.join(f'<td>{x}</td>' for x in cells)+'</tr>')
        parts.append('</table></details>')
        colors=('#0068b5','#d14900','#15854b','#8d46ab')
        parts.append('<svg viewBox="0 0 700 160" role="img" aria-label="턴별 평균 보상"><line x1="25" x2="680" y1="80" y2="80" stroke="#bbb"/>')
        for condition,color in zip(CONDITIONS,colors):
            curves=[r['evaluations']['trained'][split]['summary']['mean_reward_by_turn'] for r in record['runs'] if r['condition']==condition]
            curve=[statistics.mean(c[i] for c in curves) for i in range(len(curves[0]))]
            pts=' '.join(f'{25+655*i/max(1,len(curve)-1):.2f},{80-65*v:.2f}' for i,v in enumerate(curve))
            parts.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="2" fill="none"><title>{esc(condition)}</title></polyline>')
        parts.append('</svg><small>가로: trial의 행동 순서. 세로: 공통 척도 보상(−1~1). 각 선은 학습 seed 평균.</small><p>'+ ' · '.join(f'<span style="color:{color}">{esc(labels[c])}</span>' for c,color in zip(CONDITIONS,colors))+'</p>')
    parts.append('<h2>정보를 얻는 행동의 제한된 가치 확인</h2><p>고정된 두 규칙 prior와 알려진 copy 상대, 초기 상태(1,0), 4턴 조건의 별도 Bayes 계산입니다. 즉시 보상이 가장 높은 행동과 전체 기대 보상을 최대화하는 행동이 다릅니다. 신경망 성공이나 전체 분포의 능동 탐색 증거는 아닙니다.</p><pre>'+esc(json.dumps(record['information_witness'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<h2>조건 차이와 해석</h2><pre>'+esc(json.dumps(record['contrasts'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<p>차이는 동일 학습 seed의 동일 평가 과제 결과로 계산했습니다. 제한된 학습 seed의 기술 통계이며 유의성·H1/H2 성립 판정은 하지 않습니다. 곡선은 상태 방문 변화도 포함하며 초기−후기 차이만으로 적응을 주장하지 않습니다. JSON의 전체 trial_standard_error는 과제 차이를 섞은 기술적 지표로, 학습 seed SD나 층화 Monte Carlo 오차가 아닙니다.</p>')
    parts.append('<h2>규칙과 조건부 보상</h2><p>평가 원자료의 rule_id는 manifest의 명세와 연결됩니다. 아래 명세는 연구자용이며 정책 입력이 아닙니다. 첫 sampling 분포는 모든 보상식에 현재 두 행동의 일치 관계를 포함하는 제한된 분포입니다.</p>')
    for entry in record['rule_summaries']:
        parts.append(f'<details><summary>{esc(entry["id"])} / {esc(entry["split"])} / {esc(entry["structure"])}</summary><p>{esc(rule_text(entry["source_rule"]["root"]))}</p><details><summary>원본 규칙 명세</summary><pre>{esc(json.dumps(entry["source_rule"],ensure_ascii=False,indent=2))}</pre></details></details>')
    parts.append('<h2>주의사항</h2><ul><li>신경망은 평가 중 가중치 고정, Q는 평가 중 표 갱신입니다. posterior 기준은 평가 규칙까지 포함한 유한 catalog prior를 알아 더 강한 사전지식을 갖습니다.</li><li>DP는 규칙·상대·남은 길이를 아는 기대 상한이며 Nash regret나 learner 식별 가능성의 증거가 아닙니다. 개별 유한 표본은 기대 상한보다 높을 수 있습니다.</li><li>보상 정규화는 플레이할 task 정의의 일부입니다. 특히 win_stay의 0 임계값 때문에 원래 보상에서의 상대 행동을 보존하지 않을 수 있습니다.</li><li>nested 평가는 같은 관계 어휘의 미학습 조합이며 새로운 primitive 발견을 뜻하지 않습니다. 단순 보상 상승과 경험 기반 적응을 구분해야 합니다.</li><li>이 파일럿은 능동 탐색의 순가치나 H1/H2를 확증하지 않습니다. 기억 초기화 차이는 episode 간 효과만 측정합니다.</li></ul></html>')
    return ''.join(parts)


def run_experiment(output_dir, *, seeds=(7,17,27), updates=40, trials_per_update=8,
                   max_steps=12, hidden_size=32, reps=2):
    import torch
    from .benchmarks import RandomLearner, QLearner, NeuralLearner
    from .recurrent import save_checkpoint, load_checkpoint
    from .training import TrainConfig, train

    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(validate_seed(s) is None for s in seeds):
        raise ValueError('seeds must be nonempty, unique explicit seeds')
    for name,value in (('updates',updates),('trials_per_update',trials_per_update),
                       ('max_steps',max_steps),('hidden_size',hidden_size),('reps',reps)):
        positive_integer(value,name)
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=False)
    source_dir=directory/'sources';source_dir.mkdir()
    sources={}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        data=path.read_bytes()
        (source_dir/path.name).write_bytes(data)
        sources[path.name]=hashlib.sha256(data).hexdigest()
    manifest=build_manifest()
    _write(directory/'manifest.json',manifest)
    distances=distance_manifest(manifest)
    _write(directory/'distances.json',distances)
    witness=information_witness(manifest)
    _write(directory/'information_witness.json',witness)
    protocol={'version':1,'seeds':list(seeds),'updates':updates,'trials_per_update':trials_per_update,
              'n':3,'episodes_per_trial':2,'max_steps':max_steps,'hidden_size':hidden_size,
              'architecture':'action_set_v1','reps':reps,'evaluation_seed':918273,
              'conditions':list(CONDITIONS),'source_sha256':sources,'torch_version':str(torch.__version__),
              'torch_threads':torch.get_num_threads(),'device':'cpu',
              'decision_budget_per_condition':updates*trials_per_update*2*max_steps,
              'reference_prior':'posterior knows entire finite catalog incl heldout; DP knows true task and horizon',
              'selection':'fixed final checkpoint; training-only validation; no heldout tuning',
              'normalization':'actual task definition, not joint-task strategy preservation',
              'scope':'first transfer pilot; no active-exploration-value or H1/H2 confirmation'}
    _write(directory/'protocol.json',protocol)
    splits={s:tasks_from_manifest(manifest,split=s) for s in SPLITS}
    all_tasks=tasks_from_manifest(manifest)
    options=dict(reps=reps,episodes_per_trial=2,max_steps=max_steps)
    references={}
    factories={'random':RandomLearner,'online_q':QLearner,
               'repeat':lambda **kw:SimpleLearner(mode='repeat',**kw),
               'copy':lambda **kw:SimpleLearner(mode='copy',**kw),
               'catalog_posterior':lambda **kw:PosteriorLearner(hypotheses=all_tasks,**kw)}
    for name,factory in factories.items():
        print(f'[reference {name}]',file=sys.stderr,flush=True)
        references[name]={split:evaluate_tasks(factory,tasks,**options) for split,tasks in splits.items()}
    _write(directory/'references.json',references)
    print('[task-aware finite-horizon DP]',file=sys.stderr,flush=True)
    oracle={t.task_id:best_response_value(t,max_steps=max_steps) for ts in splits.values() for t in ts}
    _write(directory/'oracle.json',oracle)
    counts={name:{'tasks':len(tasks_from_manifest(manifest,condition=name)),
                  'rules':len({t.rule_id for t in tasks_from_manifest(manifest,condition=name)}),
                  'opponents':len({t.opponent_kind for t in tasks_from_manifest(manifest,condition=name)})}
            for name in CONDITIONS}
    record={'version':1,'protocol':protocol,'condition_counts':counts,'references':references,'information_witness':witness,
            'oracle':oracle,'oracle_by_split':{s:statistics.mean(oracle[t.task_id] for t in ts) for s,ts in splits.items()},
            'rule_summaries':[{k:r[k] for k in ('id','split','structure','source_rule')} for r in manifest['rules']],
            'runs':[],'aggregates':{},'contrasts':{}}
    for seed in seeds:
        for condition in CONDITIONS:
            print(f'[seed={seed} {condition}] training {updates} updates',file=sys.stderr,flush=True)
            tasks=tasks_from_manifest(manifest,condition=condition)
            def sample(task_seed):
                task=tasks[random.Random(task_seed).randrange(len(tasks))]
                return task.rule,task.opponent_kind
            config=TrainConfig(n=3,seed=seed,architecture='action_set_v1',updates=updates,
                               trials_per_update=trials_per_update,episodes_per_trial=2,
                               max_steps=max_steps,hidden_size=hidden_size,evaluation_trials=4,
                               evaluation_interval=max(1,updates//4))
            result=train(config,task_sampler=sample,evaluation_tasks=[(t.rule,t.opponent_kind) for t in tasks])
            run_dir=directory/f'{condition}_seed{seed}';run_dir.mkdir()
            save_checkpoint(result.initial_model,run_dir/'initial.pt')
            save_checkpoint(result.model,run_dir/'policy.pt')
            final=load_checkpoint(run_dir/'policy.pt')
            initial=load_checkpoint(run_dir/'initial.pt')
            evaluations={}
            for label,model,reset in (('initial',initial,False),('trained',final,False),('episode_reset',final,True)):
                def factory(**kw): return NeuralLearner(model=model,**kw)
                evaluations[label]={split:evaluate_tasks(factory,ts,reset_memory=reset,**options) for split,ts in splits.items()}
            row={'seed':seed,'condition':condition,'config':asdict(config),'training_task_ids':[t.task_id for t in tasks],
                 'parameter_count':sum(p.numel() for p in final.parameters()),'hidden_elements':3*hidden_size,
                 'history':result.history,'training_only_validation':result.evaluations,'evaluations':evaluations,
                 'checkpoint':str((run_dir/'policy.pt').relative_to(directory))}
            _write(run_dir/'evaluation.json',row)
            record['runs'].append(row)
            print('  '+', '.join(s+'='+f'{evaluations["trained"][s]["summary"]["mean_reward_per_step"]:.4f}' for s in SPLITS),file=sys.stderr,flush=True)
    for condition in CONDITIONS:
        rows=[r for r in record['runs'] if r['condition']==condition]
        record['aggregates'][condition]={}
        for split in SPLITS:
            trained=[r['evaluations']['trained'][split]['summary'] for r in rows]
            values=[s['mean_reward_per_step'] for s in trained]
            record['aggregates'][condition][split]={
                'initial':statistics.mean(r['evaluations']['initial'][split]['summary']['mean_reward_per_step'] for r in rows),
                'trained':statistics.mean(values),'learning_seed_sd':statistics.stdev(values) if len(values)>1 else None,
                'memory_delta':statistics.mean(r['evaluations']['trained'][split]['summary']['mean_reward_per_step']-
                                               r['evaluations']['episode_reset'][split]['summary']['mean_reward_per_step'] for r in rows),
                **{k:statistics.mean(t[k] for t in trained) for k in ('first_action','early_reward','late_reward')}}
    comparisons={'more_rules_low_opponents':(CONDITIONS[1],CONDITIONS[0]),
                 'more_rules_high_opponents':(CONDITIONS[3],CONDITIONS[2]),
                 'more_opponents_low_rules':(CONDITIONS[2],CONDITIONS[0]),
                 'more_opponents_high_rules':(CONDITIONS[3],CONDITIONS[1])}
    for name,(high,low) in comparisons.items():
        record['contrasts'][name]={}
        for split in SPLITS:
            diffs=[]
            for seed in seeds:
                def score(condition):
                    row=next(r for r in record['runs'] if r['seed']==seed and r['condition']==condition)
                    return row['evaluations']['trained'][split]['summary']['mean_reward_per_step']
                diffs.append(score(high)-score(low))
            record['contrasts'][name][split]={'seed_differences':diffs,'mean':statistics.mean(diffs),
                                            'sd':statistics.stdev(diffs) if len(diffs)>1 else None}
    annotate_record(record,manifest)
    # These notes are fixed reporting semantics, not selection based on scores.
    (directory/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for row in record['runs']:
        (directory/f'{row["condition"]}_seed{row["seed"]}'/'evaluation.json').write_text(
            json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    _write(directory/'experiment.json',record)
    with (directory/'report.html').open('x',encoding='utf-8') as stream:
        stream.write(render_report(record))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    for name,default in (('updates',40),('trials-per-update',8),('max-steps',12),('hidden-size',32),('reps',2)):
        parser.add_argument('--'+name,type=int,default=default)
    args=parser.parse_args()
    try:
        run_experiment(args.output_dir,seeds=args.seeds,updates=args.updates,trials_per_update=args.trials_per_update,
                       max_steps=args.max_steps,hidden_size=args.hidden_size,reps=args.reps)
    except (TypeError,ValueError,OSError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
