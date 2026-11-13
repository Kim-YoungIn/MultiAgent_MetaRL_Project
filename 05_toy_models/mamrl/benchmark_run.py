"""Train nominal-action policies and evaluate three functional benchmark families."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from .common import positive_integer, validate_seed


def _write(path,record):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def run_pilot(output_dir,*,seeds=(7,17,27),updates=40,evaluation_trials=48,
              trials_per_update=8,max_steps=12,hidden_size=32):
    import torch
    from .action_set import ActionSetActorCritic
    from .benchmarks import (RandomLearner,ReactiveLearner,QLearner,BeliefLearner,NeuralLearner,
                             benchmark_tasks,probe_memory_witness,evaluate_benchmark,
                             paired_benchmark_difference)
    from .benchmark_report import render_benchmark_report
    from .recurrent import RecurrentActorCritic,save_checkpoint,load_checkpoint
    from .training import TrainConfig,train

    seeds=tuple(seeds)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('learning seeds must be nonempty and unique')
    for seed in seeds:
        if seed is None:
            raise TypeError('learning seeds must be explicit')
        validate_seed(seed)
    for name,value in (('updates',updates),('evaluation_trials',evaluation_trials),
                       ('trials_per_update',trials_per_update),('max_steps',max_steps),('hidden_size',hidden_size)):
        positive_integer(value,name)
    if evaluation_trials%6:
        raise ValueError('evaluation_trials must be a multiple of 6 for all strata')
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=False)
    sources=directory/'sources';sources.mkdir()
    hashes={}
    for name in ('action_set.py','recurrent.py','training.py','opponents.py','benchmarks.py',
                 'benchmark_run.py','benchmark_report.py','rules.py','tensor.py','environment.py','ppo.py'):
        contents=Path(__file__).with_name(name).read_bytes()
        (sources/name).write_bytes(contents)
        hashes[name]=hashlib.sha256(contents).hexdigest()
    architectures=('action_set_v1','action_set_ff_v1')
    families=('hidden_action','probe_memory','opponent_response')
    old=RecurrentActorCritic(hidden_size=hidden_size,seed=0)
    nominal=ActionSetActorCritic(hidden_size=hidden_size,seed=0)
    protocol={'version':1,'seeds':list(seeds),'updates':updates,'families':list(families),
              'architectures':list(architectures),'n':3,'episodes_per_trial':2,'max_steps':max_steps,
              'trials_per_update':trials_per_update,'hidden_size':hidden_size,
              'evaluation_trials':evaluation_trials,'evaluation_seed':918273,
              'python_version':platform.python_version(),'torch_version':str(torch.__version__),
              'torch_threads':torch.get_num_threads(),'device':'cpu',
              'ordinal_parameter_count':sum(p.numel() for p in old.parameters()),
              'action_set_parameter_count':sum(p.numel() for p in nominal.parameters()),
              'ordinal_hidden_elements':hidden_size,'action_set_hidden_elements':3*hidden_size,
              'reference_priors':'belief and reactive references know family likelihoods, not task parameters',
              'test_updates':'Q table updates online; neural weights frozen',
              'checkpoint_compatibility':'ordinal v1 retained; nominal recurrent/reactive v2',
              'scope':'functional pilot; no H1/H2 or unseen-rule generalization claim',
              'source_sha256':hashes}
    _write(directory/'protocol.json',protocol)
    record={'schema_version':1,'protocol':protocol,
            'task_checks':{'memory_witness':probe_memory_witness(),
                           'bounds':{'hidden_action':1.,'probe_memory':1.,
                                     'opponent_copy':1.,'opponent_avoid':.5}},
            'references':{},'runs':[],'aggregates':{}}
    evaluation_options=dict(seed=918273,episodes_per_trial=2,max_steps=max_steps)
    task_sets={family:benchmark_tasks(family,count=evaluation_trials) for family in families}
    for family in families:
        print(f'[references {family}] same visible observations and decision budgets',file=sys.stderr,flush=True)
        tasks=task_sets[family]
        references={name:evaluate_benchmark(learner,tasks,**evaluation_options)
                    for name,learner in (('random',RandomLearner),('reactive',ReactiveLearner),
                                         ('q_learning',QLearner),('informed',BeliefLearner))}
        references['informed_reset']=evaluate_benchmark(BeliefLearner,tasks,reset_memory_between_episodes=True,
                                                         **evaluation_options)
        record['references'][family]=references
    _write(directory/'task_checks.json',record['task_checks'])
    _write(directory/'references.json',record['references'])
    for seed in seeds:
        for family in families:
            for architecture in architectures:
                run_dir=directory/f'{family}_{architecture}_seed{seed}'
                run_dir.mkdir()
                config=TrainConfig(seed=seed,family='action_bonus' if family=='hidden_action' else family,
                                   architecture=architecture,opponent_mode='mixed' if family=='opponent_response' else 'random',
                                   updates=updates,hidden_size=hidden_size,trials_per_update=trials_per_update,
                                   max_steps=max_steps)
                print(f'[seed={seed} {family} {architecture}] {updates} updates',file=sys.stderr,flush=True)
                result=train(config)
                checkpoint=run_dir/'policy.pt';initial_checkpoint=run_dir/'initial.pt'
                save_checkpoint(result.model,checkpoint);save_checkpoint(result.initial_model,initial_checkpoint)
                model=load_checkpoint(checkpoint)
                def factory(*,family,n,seed):
                    return NeuralLearner(model=model,family=family,n=n,seed=seed)
                def initial_factory(*,family,n,seed):
                    return NeuralLearner(model=result.initial_model,family=family,n=n,seed=seed)
                comparisons={'initial':evaluate_benchmark(initial_factory,task_sets[family],**evaluation_options),
                             'trained':evaluate_benchmark(factory,task_sets[family],**evaluation_options),
                             'episode_reset':evaluate_benchmark(factory,task_sets[family],
                                                               reset_memory_between_episodes=True,**evaluation_options)}
                row={'seed':seed,'family':family,'architecture':architecture,'config':asdict(config),
                     'checkpoint':str(checkpoint.relative_to(directory)),
                     'initial_checkpoint':str(initial_checkpoint.relative_to(directory)),
                     'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     'parameter_count':sum(p.numel() for p in model.parameters()),
                     'hidden_state_elements':model.initial_hidden(n=3).numel(),
                     'comparisons':comparisons,
                     'memory_benefit':paired_benchmark_difference(comparisons['trained'],comparisons['episode_reset']),
                     'history':result.history,'evaluations':result.evaluations,
                     'evaluation_seed':918273,'evaluation_tasks':[asdict(t) for t in task_sets[family]],
                     'raw_record':str((run_dir/'run.json').relative_to(directory))}
                _write(directory/row['raw_record'],row)
                record['runs'].append(row)
                print(f'  reward/step={comparisons["trained"]["mean_reward_per_step"]:.6f}, '
                      f'memory delta={row["memory_benefit"]["mean"]:.6f}',file=sys.stderr,flush=True)
    for family in families:
        record['aggregates'][family]={}
        for architecture in architectures:
            rows=[r for r in record['runs'] if r['family']==family and r['architecture']==architecture]
            values=[r['comparisons']['trained']['mean_reward_per_step'] for r in rows]
            record['aggregates'][family][architecture]={
                'mean_reward':statistics.mean(values),
                'sd_across_learning_seeds':statistics.stdev(values) if len(values)>1 else None,
                'mean_memory_benefit':statistics.mean(r['memory_benefit']['mean'] for r in rows)}
    _write(directory/'benchmark.json',record)
    with (directory/'report.html').open('x',encoding='utf-8') as stream:
        stream.write(render_benchmark_report(record))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    for name,default in (('updates',40),('evaluation_trials',48),('trials_per_update',8),
                         ('max_steps',12),('hidden_size',32)):
        parser.add_argument('--'+name.replace('_','-'),type=int,default=default)
    args=parser.parse_args()
    try:
        import torch
        torch.set_num_threads(1)
        record=run_pilot(**vars(args))
    except (ImportError,TypeError,ValueError,OSError,RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'runs':len(record['runs']),
                      'aggregates':record['aggregates']},sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
