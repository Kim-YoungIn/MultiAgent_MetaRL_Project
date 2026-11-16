"""Separate collected experience, student representation and opponent-state effects.

The natural MA trial is retained. Conditional replay uses one reference opponent
state for all histories; such recombinations are diagnostics, not natural play.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics

from .adaptive_agents import OnlineQ
from .environment import Observation
from .meta_exploration import (RandomExplorer,EpsilonExplorer,BalancedExplorer,RepeatExplorer,
                               neural_factory,decode_rule,meta_tasks,SystematicExplorer)
from .meta_exploration_rollout import run_meta_trial,assess_pair,_make_env
from .rule_metrics import stable_digest
from .common import positive_integer,finite_reward


class RepresentationQ(OnlineQ):
    def __init__(self,*args,representation='actions',**kwargs):
        super().__init__(*args,**kwargs)
        if representation not in ('actions','binned'):raise ValueError('unknown compressed representation')
        self.representation=representation
    def _key(self,observation):
        own,other,reward,start=super()._key(observation)
        reward=0.0 if self.representation=='actions' else float((reward>0)-(reward<0))
        return own,other,reward,start
    def clone(self):
        result=super().clone();result.representation=self.representation;return result


def replay_history(events,*,n,seed,representation='full'):
    if representation not in ('full','actions','binned'):raise ValueError('unknown student representation')
    student=OnlineQ(n,seed) if representation=='full' else RepresentationQ(n,seed,representation=representation)
    for event in events:
        student.update(Observation(**event['observations'][0]),event['actions'][0],event['rewards'][0],
                       Observation(**event['learning_next_observations'][0]),terminal=event['terminal'])
    return student


def collect_history_context(rule,factory,*,seed,steps,assessment_steps,opponent_profile='steady'):
    capture={}
    trial=run_meta_trial(rule,factory,seed=seed,steps=steps,assessment_steps=assessment_steps,
                         opponent_profile=opponent_profile,state_capture=capture)
    return trial,capture


def threshold_time(curve,*,threshold,window=3,sustain=2):
    positive_integer(window,'window');positive_integer(sustain,'sustain')
    threshold=finite_reward(threshold)
    consecutive=0
    for end in range(window,len(curve)+1):
        consecutive=consecutive+1 if statistics.mean(curve[end-window:end])>=threshold else 0
        if consecutive>=sustain:return {'step':end,'reached':True,'right_censored':False,'horizon':len(curve)}
    return {'step':None,'reached':False,'right_censored':True,'horizon':len(curve)}


def score_history(trial,capture,reference_capture,representation='full',window=3,threshold=None):
    events=trial['exploration'];n=capture['tensor'].n
    student=replay_history(events,n=n,seed=capture['seeds']['student'],representation=representation)
    before=student.snapshot()
    if representation=='full' and before!=capture['student'].snapshot():
        raise RuntimeError('history replay did not reconstruct the original student')
    opponent=reference_capture['opponent'].clone()
    env,start=_make_env(capture['tensor'],capture['assessment_steps'],reference_capture['seeds']['assessment_reset'])
    assessed=assess_pair(env,start,student,opponent,capture['assessment_steps'])
    curve=[e['rewards'][0] for e in assessed]
    seen={student._key(Observation(**e['observations'][0])) for e in events}
    overlap=sum(student._key(Observation(**e['observations'][0])) in seen for e in assessed)/len(assessed)
    result={'history_sha256':stable_digest(events),'assessment_mean':statistics.mean(curve),'assessment_curve':curve,
            'assessment_prefix_auc':{str(k):statistics.mean(curve[:k]) for k in sorted({1,min(3,len(curve)),min(6,len(curve)),len(curve)})},
            'student_before':before,'student_after':student.snapshot(),
            'opponent_before':reference_capture['opponent'].snapshot(),
            'unique_exploration_states':len(seen),'exploration_revisit_fraction':1-len(seen)/max(1,len(events)),
            'assessment_state_overlap':overlap,'student_updates_from_history':before['updates'],
            'control_gameplay_steps':len(assessed),
            'time_to_threshold':threshold_time(curve,threshold=threshold,window=window) if threshold is not None else None}
    return result


def paired_identifiability(events,candidates):
    """Researcher-only, known finite prior. Never supplied to a policy/student."""
    from .rules import compile_rule
    candidates=[compile_rule(decode_rule(e['rule'])) for e in candidates]
    counts=[len(candidates)]
    for event in events:
        obs=event['observations'][0];a,b=event['actions'];reward=event['rewards'][0]
        candidates=[t for t in candidates if abs(t[0,obs['previous_own_action'],obs['previous_opponent_action'],a,b]-reward)<1e-9]
        counts.append(len(candidates))
    return {'candidate_counts':counts,'first_unique_step':next((i for i,c in enumerate(counts) if c==1),None),
            'final_entropy_bits':math.log2(counts[-1]) if counts[-1] else None,
            'scope':'known paired-rule prior; history identifiability, not a decoded neural belief'}


def run_history_study(manifest,*,output_dir,checkpoint=None,initial_checkpoint=None,steps=12,
                      assessment_steps=48,reps=2,seed=918273,device='cpu',threshold=None,split='fresh_rules',
                      opponent_suite='mixed_learning',reference_low=None,reference_high=None,threshold_fraction=.8):
    from .run_files import write_json_exclusive
    from .probe_templates import validate_probe_manifest
    validate_probe_manifest(manifest)
    positive_integer(reps,'reps');positive_integer(steps,'steps');positive_integer(assessment_steps,'assessment_steps')
    reference_definition=None
    if reference_low is not None or reference_high is not None:
        if threshold is not None or reference_low is None or reference_high is None:
            raise ValueError('supply either absolute threshold or both reference bounds')
        low,high,fraction=map(finite_reward,(reference_low,reference_high,threshold_fraction))
        if high<=low or not 0<fraction<=1:raise ValueError('reference gap/fraction must be positive')
        threshold=low+fraction*(high-low)
        reference_definition={'low':low,'high':high,'fraction':fraction,'scope':'declared reference, not an assumed attainable maximum'}
    if threshold is not None:threshold=finite_reward(threshold)
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=False)
    factories={'random':RandomExplorer,'epsilon_greedy':EpsilonExplorer,'balanced':BalancedExplorer,
               'repeat':RepeatExplorer,'systematic':SystematicExplorer}
    if checkpoint or initial_checkpoint:
        from .recurrent import load_checkpoint
        for name,path in [('learned',checkpoint),('initial',initial_checkpoint)]:
            if path:factories[name]=neural_factory(load_checkpoint(path,device=device))
    # Preserve the actual source and portable teachers used for replay.
    import shutil
    source_dir=p/'sources';source_dir.mkdir()
    source_hashes={}
    for source in sorted(Path(__file__).parent.glob('*.py')):
        data=source.read_bytes();(source_dir/source.name).write_bytes(data)
        source_hashes[source.name]=hashlib.sha256(data).hexdigest()
    teacher_hashes={}
    for name,path in [('teacher',checkpoint),('initial',initial_checkpoint)]:
        if path:
            shutil.copyfile(path,p/f'{name}.pt');teacher_hashes[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    tasks=meta_tasks(manifest,split,opponent_suite=opponent_suite)
    rows=[];histories={};cost=0
    by_seed=defaultdict(list)
    for entry in manifest['rules']:by_seed[(entry['split'],entry['seed'])].append(entry)
    for task in tasks:
        for rep in range(reps):
            trial_seed=int.from_bytes(hashlib.sha256(f'{seed}/{task["id"]}/{rep}'.encode()).digest()[:8],'big')%2**63
            collected={name:collect_history_context(decode_rule(task['rule']),factory,seed=trial_seed,steps=steps,
                        assessment_steps=assessment_steps,opponent_profile=task['opponent_profile']) for name,factory in factories.items()}
            reference=collected['random'][1]
            for name,(trial,capture) in collected.items():
                history_id=f'{task["id"]}/{rep}/{name}'
                histories[history_id]=trial['exploration'];cost+=trial['environment_steps_with_control']
                identity=paired_identifiability(trial['exploration'],by_seed[(task['split'],next(e['seed'] for e in manifest['rules'] if e['id']==task['rule_id']))])
                for representation in ('full','actions','binned'):
                    score=score_history(trial,capture,reference,representation=representation,threshold=threshold)
                    cost+=assessment_steps
                    rows.append({'task_id':task['id'],'rep':rep,'explorer':name,'representation':representation,
                                 'history_id':history_id,'identifiability':identity,
                                 'natural_assessment_mean':trial['assessment_mean'],'natural_G':trial['meta_reward'],**score})
                # Empty history shares the same opponent/reset; it is a matched
                # control, not exploration with K free interactions removed.
                empty=OnlineQ(capture['tensor'].n,capture['seeds']['student'])
                env,start=_make_env(capture['tensor'],assessment_steps,reference['seeds']['assessment_reset'])
                assessed=assess_pair(env,start,empty,reference['opponent'].clone(),assessment_steps)
                empty_score=statistics.mean(e['rewards'][0] for e in assessed);cost+=assessment_steps
                for row in rows[-3:]:row['conditional_minus_empty']=row['assessment_mean']-empty_score
    summary={}
    for name in factories:
        summary[name]={}
        for representation in ('full','actions','binned'):
            group=[r for r in rows if r['explorer']==name and r['representation']==representation]
            summary[name][representation]={key:statistics.mean(r[key] for r in group) for key in
                ('assessment_mean','conditional_minus_empty','natural_assessment_mean','assessment_state_overlap','unique_exploration_states')}
            if threshold is not None:
                summary[name][representation]['threshold_reach_fraction']=statistics.mean(r['time_to_threshold']['reached'] for r in group)
    record={'version':1,'protocol':{'K':steps,'H':assessment_steps,'reps':reps,'seed':seed,'split':split,
             'opponent_suite':opponent_suite,'opponent_reference':'random-exploration end state','natural_student':'full exact-reward Q',
             'threshold':threshold,'threshold_reference':reference_definition,'source_sha256':source_hashes,'teacher_sha256':teacher_hashes,'threshold_window':3,'threshold_sustain':2,
             'simulation_steps_including_all_controls':cost,
             'scope':'conditional replay and natural MA scores; not pure information causality'},
             'summary':summary,'rows':rows}
    write_json_exclusive(p/'manifest.json',manifest);write_json_exclusive(p/'histories.json',histories)
    write_json_exclusive(p/'study.json',record)
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>History와 학생 비교</title>',
           '<h1>탐색 History × 학생 표현</h1><p>학습 상대는 유지합니다. 조건부 비교에서는 랜덤 탐색이 만든 같은 상대 상태를 사용합니다. 자연스러운 공동 적응 점수와 구분합니다.</p>',
           '<a href="study.json">지표 원자료</a> · <a href="histories.json">공유 History</a>',
           '<table border="1"><tr><th>탐색</th><th>학생 표현</th><th>동일 상대 후속</th><th>빈 경험 대비</th><th>자연 MA(full 학생)</th><th>상태 재사용</th></tr>']
    for name,values in summary.items():
        for representation,value in values.items():
            parts.append('<tr><td>'+html.escape(name)+'</td><td>'+representation+'</td>'+''.join(f'<td>{value[k]:.4f}</td>' for k in ('assessment_mean','conditional_minus_empty','natural_assessment_mean','assessment_state_overlap'))+'</tr>')
    parts.append('</table><p>paired-rule 식별성은 연구자용 알려진 prior의 진단이며 신경망이 규칙을 내부적으로 분류했다는 증거가 아닙니다. threshold 미도달은 우측 검열이며 최대 return을 임의로 가정하지 않습니다.</p></html>')
    (p/'report.html').write_text(''.join(parts),encoding='utf-8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--initial-checkpoint',type=Path)
    parser.add_argument('--steps',type=int,default=12);parser.add_argument('--assessment-steps',type=int,default=48)
    parser.add_argument('--reps',type=int,default=2);parser.add_argument('--device',default='cpu',choices=('cpu','cuda','auto'))
    parser.add_argument('--reference-low',type=float);parser.add_argument('--reference-high',type=float)
    parser.add_argument('--threshold-fraction',type=float,default=.8)
    parser.add_argument('--threshold',type=float);parser.add_argument('--split',default='fresh_rules',choices=('train_rule_reference','fresh_rules','structural','new_learning_profile'))
    args=parser.parse_args()
    run_history_study(json.loads(args.manifest.read_text()),output_dir=args.output_dir,checkpoint=args.checkpoint,
                      initial_checkpoint=args.initial_checkpoint,steps=args.steps,assessment_steps=args.assessment_steps,
                      reps=args.reps,device=args.device,threshold=args.threshold,split=args.split,
                      reference_low=args.reference_low,reference_high=args.reference_high,threshold_fraction=args.threshold_fraction)

if __name__=='__main__':main()
