"""Paired development comparison of G and absolute downstream objectives.

All catalog/learner/opponent/budget settings are held fixed. Predeclared update
checkpoints are reported; no checkpoint is selected using development scores.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import statistics

from .common import positive_integer
from .rule_metrics import stable_digest
from .run_files import write_json_exclusive


def compact_evaluation(value,theta):
    rows=[];conditional=defaultdict(lambda:[0,0])
    for trial in value['trials']:
        events=trial['exploration'];repeat=0
        for step,event in enumerate(events):
            obs=event['observations'][0];is_repeat=event['actions'][0]==obs['previous_own_action'];repeat+=is_repeat
            sign=(obs['previous_own_reward']>0)-(obs['previous_own_reward']<0)
            key=f'theta={theta[trial["rule_id"]]}/opp={trial["opponent_profile"]}/reward_sign={sign}/same={obs["previous_own_action"]==obs["previous_opponent_action"]}/phase={"early" if step<3 else "late"}'
            conditional[key][0]+=int(is_repeat);conditional[key][1]+=1
        k,h=len(events),len(trial['assessment'])
        rows.append({key:trial[key] for key in ('task_id','rule_id','rep','seed','opponent_profile','assessment_mean','meta_reward','erased_mean','exploration_mean')}
                    |{'theta':theta[trial['rule_id']],'repeat_fraction':repeat/k,
                      'total_gameplay_mean':(k*trial['exploration_mean']+h*trial['assessment_mean'])/(k+h)})
    return {'summary':value['summary'],'by_family':value.get('by_family',{}),'rows':rows,
            'conditional_repeat_counts':dict(conditional)}


def run_objective_study(manifest,*,output_dir,seeds=(7,17,27),updates=200,checkpoints=(50,100,200),
                        trials_per_update=32,steps=12,assessment_steps=48,hidden_size=32,
                        batch_size=32,reps=1,device='cpu',resume=False):
    import torch
    from .devices import resolve_device
    from .meta_exploration import MetaConfig,train_meta,meta_tasks,evaluate_meta,neural_factory,SPLITS
    from .probe_templates import validate_probe_manifest
    from .recurrent import save_checkpoint,load_checkpoint
    validate_probe_manifest(manifest)
    positive_integer(updates,'updates')
    for key,val in [('trials_per_update',trials_per_update),('steps',steps),('assessment_steps',assessment_steps),
                    ('hidden_size',hidden_size),('batch_size',batch_size),('reps',reps)]:positive_integer(val,key)
    seeds=tuple(seeds);checkpoints=tuple(checkpoints)
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('unique nonempty seeds required')
    if not checkpoints or sorted(set(checkpoints))!=list(checkpoints) or checkpoints[-1]!=updates or any(type(v) is not int or v<=0 for v in checkpoints):
        raise ValueError('checkpoints must increase and end at the fixed training budget')
    device=str(resolve_device(device));p=Path(output_dir)
    source_files=sorted(Path(__file__).parent.glob('*.py'))
    protocol={'version':1,'seeds':list(seeds),'updates':updates,'checkpoints':list(checkpoints),
              'trials_per_update':trials_per_update,'K':steps,'H':assessment_steps,'hidden_size':hidden_size,
              'batch_size':batch_size,'reps':reps,'device':device,'torch_version':str(torch.__version__),
              'torch_threads':torch.get_num_threads(),'manifest_sha256':stable_digest(manifest),
              'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
              'opponent_suite':'mixed_learning','selection':'fixed final update, no dev-best selection',
              'scope':'development objective comparison, not diversity H1 or final evaluation',
              'training_simulation_steps':2*len(seeds)*updates*trials_per_update*(steps+2*assessment_steps)}
    if resume:
        if json.loads((p/'protocol.json').read_text())!=protocol:raise ValueError('study resume protocol mismatch')
    else:
        p.mkdir(parents=True,exist_ok=False);(p/'sources').mkdir()
        for file in source_files:(p/'sources'/file.name).write_bytes(file.read_bytes())
        write_json_exclusive(p/'protocol.json',protocol);write_json_exclusive(p/'manifest.json',manifest)
    theta={e['id']:e['parameters']['theta'] for e in manifest['rules']}
    tasks={s:meta_tasks(manifest,s,opponent_suite='mixed_learning') for s in SPLITS}
    records=[]
    for objective in ('retained_gain','assessment_return'):
        for seed in seeds:
            d=p/objective/f'seed{seed}';d.mkdir(parents=True,exist_ok=True)
            cfg=MetaConfig(seed=seed,updates=updates,trials_per_update=trials_per_update,steps=steps,
                           assessment_steps=assessment_steps,hidden_size=hidden_size,batch_size=batch_size,
                           device=device,objective=objective)
            state=d/'training_state.pt'
            for stop in checkpoints:
                evaluation_file=d/f'update{stop}.json'
                if evaluation_file.exists():continue
                if state.exists():
                    saved=torch.load(state,map_location='cpu',weights_only=True)
                    if saved['completed_updates']>stop:raise ValueError('missing evaluation predates restart state; restore its checkpoint before continuing')
                result=train_meta(cfg,manifest,opponent_suite='mixed_learning',checkpoint_path=state,
                                  resume=state.exists(),stop_after_update=stop,checkpoint_every=20)
                if not (d/'initial.pt').exists():save_checkpoint(result['initial_model'],d/'initial.pt')
                if not (d/'update0.json').exists():
                    initial={s:compact_evaluation(evaluate_meta(neural_factory(result['initial_model']),t,steps=steps,
                             assessment_steps=assessment_steps,reps=reps,batch_size=batch_size),theta) for s,t in tasks.items()}
                    write_json_exclusive(d/'update0.json',{'seed':seed,'objective':objective,'update':0,'evaluation':initial})
                weight=d/f'update{stop}.pt'
                if not weight.exists():save_checkpoint(result['model'],weight)
                # Reload the portable artifact so evaluation checks the saved model.
                model=load_checkpoint(weight,device=device)
                evaluation={s:compact_evaluation(evaluate_meta(neural_factory(model),t,steps=steps,
                            assessment_steps=assessment_steps,reps=reps,batch_size=batch_size),theta) for s,t in tasks.items()}
                write_json_exclusive(evaluation_file,{'seed':seed,'objective':objective,'update':stop,'evaluation':evaluation})
                print(f'[objective={objective} seed={seed} evaluated update={stop}]',flush=True)
            # One copy of per-update training history, separate from evaluation snapshots.
            if not (d/'history.json').exists():
                state_data=torch.load(state,map_location='cpu',weights_only=True)
                write_json_exclusive(d/'history.json',state_data['history'])
            records.append({'objective':objective,'seed':seed,'final':f'{objective}/seed{seed}/update{updates}.json'})
    summary={}
    for split in SPLITS:
        summary[split]={}
        for objective in ('retained_gain','assessment_return'):
            finals=[json.loads((p/r['final']).read_text()) for r in records if r['objective']==objective]
            summary[split][objective]={key:{'seed_values':[x['evaluation'][split]['summary'][key] for x in finals],
                                           'mean':statistics.mean(x['evaluation'][split]['summary'][key] for x in finals)}
                                       for key in ('assessment_mean','meta_reward','total_gameplay_mean')}
        for key in ('assessment_mean','meta_reward','total_gameplay_mean'):
            a=summary[split]['assessment_return'][key]['seed_values'];b=summary[split]['retained_gain'][key]['seed_values']
            summary[split][f'paired_absolute_minus_G_{key}']=[x-y for x,y in zip(a,b)]
    record={'protocol':protocol,'runs':records,'summary':summary}
    if not (p/'study.json').exists():write_json_exclusive(p/'study.json',record)
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>목표함수 대조</title>',
           '<h1>동일 조건: G 목표 vs 절대 후속 보상 목표</h1><p>학생·규칙·상대·초기화·예산과 제어 계산량을 고정했습니다. 최종 update는 사전 고정하며 개발 최고 점수로 선택하지 않습니다.</p>',
           '<a href="study.json">요약 원자료</a><table border="1"><tr><th>평가</th><th>목표</th><th>후속</th><th>전체 게임</th><th>G</th></tr>']
    for split,group in summary.items():
        for objective in ('retained_gain','assessment_return'):
            v=group[objective]
            parts.append(f'<tr><td>{html.escape(split)}</td><td>{objective}</td>'+''.join(f'<td>{v[k]["mean"]:.4f}</td>' for k in ('assessment_mean','total_gameplay_mean','meta_reward'))+'</tr>')
    parts.append('</table><p>평가 trajectory는 저장하지 않고 task별 지표와 조건부 반복 횟수를 보존합니다. 고정 평가 난수와 세 훈련 seed는 서로 다른 독립성 수준입니다.</p></html>')
    (p/'report.html').write_text(''.join(parts),encoding='utf-8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,17,27]);parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[50,100,200]);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    for name,default in [('trials-per-update',32),('steps',12),('assessment-steps',48),('hidden-size',32),('batch-size',32),('reps',1)]:
        parser.add_argument('--'+name,type=int,default=default)
    args=parser.parse_args()
    run_objective_study(json.loads(args.manifest.read_text()),output_dir=args.output_dir,seeds=args.seeds,updates=args.updates,
        checkpoints=args.checkpoints,resume=args.resume,device=args.device,trials_per_update=args.trials_per_update,
        steps=args.steps,assessment_steps=args.assessment_steps,hidden_size=args.hidden_size,batch_size=args.batch_size,reps=args.reps)

if __name__=='__main__':main()
