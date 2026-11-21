"""Cross collected student histories with learned-opponent states, plus frozen Q."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics

from .common import positive_integer, validate_seed
from .history_study import collect_history_context, replay_history
from .meta_exploration import RandomExplorer, decode_rule, meta_tasks, neural_factory
from .meta_exploration_rollout import _make_env, assess_pair
from .probe_templates import validate_probe_manifest
from .recurrent import load_checkpoint
from .rule_metrics import stable_digest
from .run_files import write_json_exclusive


def run_reference_study(manifest, initial_checkpoint, checkpoint, output_dir, *, steps=12,
                        assessment_steps=48, reps=1, seed=918273, split='fresh_rules',
                        device='cpu', opponent_suite='mixed_learning'):
    validate_probe_manifest(manifest)
    for name,value in [('steps',steps),('assessment_steps',assessment_steps),('reps',reps)]:
        positive_integer(value,name)
    if validate_seed(seed) is None: raise ValueError('explicit seed required')
    factories={'random':RandomExplorer,
               'initial':neural_factory(load_checkpoint(initial_checkpoint,device=device)),
               'learned':neural_factory(load_checkpoint(checkpoint,device=device))}
    tasks=meta_tasks(manifest,split,opponent_suite=opponent_suite)
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    source=root/'sources';source.mkdir()
    hashes={}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        data=path.read_bytes();(source/path.name).write_bytes(data);hashes[path.name]=hashlib.sha256(data).hexdigest()
    checkpoint_hashes={}
    for name,path in [('initial',initial_checkpoint),('learned',checkpoint)]:
        data=Path(path).read_bytes();(root/f'{name}.pt').write_bytes(data);checkpoint_hashes[name]=hashlib.sha256(data).hexdigest()
    rows=[]
    for task in tasks:
        rule=decode_rule(task['rule'])
        for rep in range(reps):
            trial_seed=int.from_bytes(hashlib.sha256(f'{seed}/{task["id"]}/{rep}'.encode()).digest()[:8],'big')%2**63
            collected={name:collect_history_context(rule,factory,seed=trial_seed,steps=steps,
                       assessment_steps=assessment_steps,opponent_profile=task['opponent_profile'])
                       for name,factory in factories.items()}
            for history,(trial,capture) in collected.items():
                student0=replay_history(trial['exploration'],n=rule.n,seed=capture['seeds']['student'])
                if student0.snapshot()!=capture['student'].snapshot():raise RuntimeError('history replay mismatch')
                for reference,(_,reference_capture) in collected.items():
                    for mode in ('online','frozen'):
                        student=student0.clone();opponent=reference_capture['opponent'].clone()
                        before_student=student.snapshot();before_opponent=opponent.snapshot()
                        env,start=_make_env(capture['tensor'],assessment_steps,reference_capture['seeds']['assessment_reset'])
                        events=assess_pair(env,start,student,opponent,assessment_steps,learn_student=(mode=='online'))
                        curve=[e['rewards'][0] for e in events]
                        if history==reference and mode=='online' and events!=trial['assessment']:
                            raise RuntimeError('natural diagonal differs from paired reassessment')
                        if mode=='frozen' and student.snapshot()['updates']!=before_student['updates']:
                            raise RuntimeError('frozen student updated')
                        rows.append({'task_id':task['id'],'rule_id':task['rule_id'],'opponent_profile':task['opponent_profile'],
                                     'rep':rep,'seed':trial_seed,'history':history,'reference':reference,'mode':mode,
                                     'history_sha256':stable_digest(trial['exploration']),
                                     'student_before':before_student,'student_after':student.snapshot(),
                                     'opponent_before':before_opponent,'opponent_after':opponent.snapshot(),
                                     'assessment_curve':curve,'assessment_mean':statistics.mean(curve),
                                     'opponent_nonuniform_events':sum(sum(abs(p-1/rule.n) for p in e['opponent_action_probabilities'])>1e-12 for e in events),
                                     'natural_assessment_mean':trial['assessment_mean'],'natural_G':trial['meta_reward']})
    summary={mode:{history:{reference:statistics.mean(r['assessment_mean'] for r in rows if r['mode']==mode and r['history']==history and r['reference']==reference)
                              for reference in factories} for history in factories} for mode in ('online','frozen')}
    record={'version':1,'protocol':{'K':steps,'H':assessment_steps,'reps':reps,'seed':seed,'split':split,
            'opponent_suite':opponent_suite,'device':device,'source_sha256':hashes,'checkpoint_sha256':checkpoint_hashes,
            'tasks':len(tasks),'simulation_steps':len(tasks)*reps*(3*(steps+2*assessment_steps)+18*assessment_steps),
            'frozen_scope':'student updates disabled; opponent continues online learning',
            'scope':'conditional counterfactual recombinations; natural diagonal is reported, not pure information causality'},
            'summary':summary,'rows':rows}
    write_json_exclusive(root/'manifest.json',manifest);write_json_exclusive(root/'study.json',record)
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>History × 상대 상태</title>',
           '<h1>History × 학습 상대 상태</h1><p>행은 학생이 얻은 History, 열은 탐색 종료 시 상대 상태입니다. '
           '대각선만 자연스러운 조합입니다. Frozen은 학생만 고정하며 상대는 계속 학습합니다. '
           '교차 결과는 조건부 진단이며 순수 정보 인과효과가 아닙니다.</p><a href="study.json">원자료</a>']
    for mode in ('online','frozen'):
        parts.append('<h2>'+html.escape(mode)+'</h2><table border="1"><tr><th>History / 상대</th><th>random</th><th>initial</th><th>learned</th></tr>')
        for history in factories:
            parts.append('<tr><th>'+history+'</th>'+''.join(f'<td>{summary[mode][history][ref]:.4f}</td>' for ref in factories)+'</tr>')
        parts.append('</table>')
    parts.append('</html>');(root/'report.html').write_text(''.join(parts),encoding='utf-8')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--assessment-steps',type=int,default=48)
    p.add_argument('--reps',type=int,default=1);p.add_argument('--seed',type=int,default=918273)
    p.add_argument('--split',default='fresh_rules',choices=('train_rule_reference','fresh_rules','structural','new_learning_profile'))
    p.add_argument('--device',default='cpu',choices=('cpu','cuda','auto'))
    a=p.parse_args()
    run_reference_study(json.loads(a.manifest.read_text()),a.initial_checkpoint,a.checkpoint,a.output_dir,
                        steps=a.steps,assessment_steps=a.assessment_steps,reps=a.reps,seed=a.seed,split=a.split,device=a.device)

if __name__=='__main__':main()
