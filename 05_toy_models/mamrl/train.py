"""Run the optional CPU recurrent PPO baseline and save a local learning report."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='new directory for all artifacts')
    parser.add_argument('--family', choices=('action_bonus', 'zero_sum', 'coordination', 'memory_one', 'mixed',
                                          'probe_memory', 'opponent_response'))
    parser.add_argument('--architecture', choices=('ordinal_v1','action_set_v1','action_set_ff_v1'), default='action_set_v1')
    parser.add_argument('--opponent-mode', choices=('random','copy_previous','avoid_previous','mixed'))
    for name in ('n', 'seed', 'fixed_action', 'hidden_size', 'updates', 'trials_per_update', 'episodes_per_trial',
                 'max_steps', 'epochs', 'evaluation_trials', 'evaluation_interval'):
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('learning_rate', 'gamma', 'gae_lambda', 'clip_ratio', 'value_coefficient',
                 'entropy_coefficient', 'max_grad_norm'):
        parser.add_argument('--' + name.replace('_', '-'), type=float)
    args = parser.parse_args()
    try:
        import torch
        from .training import (TrainConfig, collect_trial, evaluate, evaluation_specs,
                               evaluation_opponents, sample_task, train)
        from .recurrent import RecurrentPolicy, save_checkpoint
        from .environment import MemoryOneEnv
        from .policy import RandomPolicy
        from .opponents import make_opponent
        from .recording import build_record, save_run
        from .rollout import run_episode
        from .rules import compile_rule
        from .train_report import render_training_report
    except ImportError as exc:
        parser.error(f'학습 의존성을 설치한 .venv/bin/python을 사용하세요: {exc}')
    try:
        options = {key: value for key, value in vars(args).items()
                   if key != 'output_dir' and value is not None}
        if options.get('family') == 'opponent_response' and 'opponent_mode' not in options:
            options['opponent_mode'] = 'mixed'
        config = TrainConfig(**options)
        # Reserve a fresh output directory before any expensive work.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(1)
        result = train(config)
        specs = evaluation_specs(config)
        evaluation_seed = config.seed + 2_000_000
        eval_options = dict(episodes_per_trial=config.episodes_per_trial, max_steps=config.max_steps,
                            seed=evaluation_seed)
        if config.opponent_mode != 'random':
            eval_options['opponent_kinds'] = evaluation_opponents(config)
        comparisons = {
            'initial': result.evaluations[0] | {},
            'random': evaluate(None, specs, **eval_options),
            'trained': result.evaluations[-1] | {},
            'hidden_reset': evaluate(result.model, specs, reset_hidden=True, **eval_options),
        }
        for name in ('initial', 'trained'):
            comparisons[name].pop('update')
        change = math.sqrt(sum(float((p.detach().double()-q.detach().double()).square().sum())
                               for p, q in zip(result.model.parameters(), result.initial_model.parameters())))
        record = {
            'schema_version': 1, 'kind': 'recurrent_ppo_engineering_baseline',
            'config': asdict(config),
            'runtime': {'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
                        'device': 'cpu', 'threads': torch.get_num_threads()},
            'history': result.history, 'evaluations': result.evaluations,
            'comparisons': comparisons, 'parameter_change_l2': change,
            'evaluation_task_specs': [asdict(spec) for spec in specs], 'evaluation_seed': evaluation_seed,
            'evaluation_opponent_kinds': list(eval_options.get('opponent_kinds',('random',)*len(specs))),
        }
        # Preserve the first complete evaluation trial, including boundary events.
        trial_seed = random.Random(evaluation_seed).getrandbits(63)
        trace = collect_trial(result.model, specs[0], episodes_per_trial=config.episodes_per_trial,
                              max_steps=config.max_steps, seed=trial_seed,
                              opponent_kind=eval_options.get('opponent_kinds',('random',))[0])
        trace_record = {'schema_version': 1, 'trial_seed': trial_seed,
                        'model_checkpoint': 'policy.pt', 'rollout': asdict(trace)}
        # This is a separate single-episode inspection, not the full-trial evaluation.
        sample_seed = config.seed + 3_000_000
        sample_spec = sample_task(config.n, config.family, sample_seed, fixed_action=config.fixed_action)
        sample_tensor = compile_rule(sample_spec)
        sample_kind = 'copy_previous' if config.opponent_mode == 'mixed' else config.opponent_mode
        sample_policies = (RecurrentPolicy(result.model), make_opponent(sample_kind))
        sample_policy_seeds = (sample_seed+1, sample_seed+2)
        sample = run_episode(MemoryOneEnv(sample_tensor, max_steps=config.max_steps), sample_policies,
                              seed=sample_seed, policy_seeds=sample_policy_seeds)
        sample_record = build_record(sample_spec, sample_tensor, sample, family=config.family,
                                     seed=sample_seed, policy_seeds=sample_policy_seeds,
                                     policy_names=tuple(type(policy).__name__ for policy in sample_policies))
        html = render_training_report(record)
        save_checkpoint(result.initial_model, args.output_dir / 'initial.pt')
        save_checkpoint(result.model, args.output_dir / 'policy.pt')
        for name, contents in (('config.json', asdict(config)), ('training.json', record),
                               ('evaluation_trial.json', trace_record)):
            with (args.output_dir / name).open('x', encoding='utf-8') as stream:
                json.dump(contents, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        with (args.output_dir / 'report.html').open('x', encoding='utf-8') as stream:
            stream.write(html)
        save_run(sample_record, args.output_dir / 'sample_episode')
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'updates': config.updates,
                      'comparisons': {key: {'mean_reward_per_step': value['mean_reward_per_step'],
                                            'standard_error': value['standard_error']}
                                      for key, value in comparisons.items()},
                      'parameter_change_l2': change}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
