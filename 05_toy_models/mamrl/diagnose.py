"""Run a predeclared CPU diagnostic matrix without replacing the PPO baseline."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

from .common import positive_integer, validate_n, validate_seed


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _aggregate(rows):
    values = [row['comparisons']['trained']['mean_reward_per_step'] for row in rows]
    return {
        'learning_seed_count': len(values),
        'mean_reward_across_learning_seeds': statistics.mean(values),
        'sd_across_learning_seeds': statistics.stdev(values) if len(values) > 1 else None,
        'mean_paired_vs_random': statistics.mean(row['paired_vs_random']['mean'] for row in rows),
        'mean_paired_memory_benefit': statistics.mean(row['paired_memory_benefit']['mean'] for row in rows),
        'per_seed': [{'seed': row['seed'], 'mean_reward_per_step': row['comparisons']['trained']['mean_reward_per_step']}
                     for row in rows],
    }


def run_study(output_dir, *, seeds=(7, 17, 27), updates=120, short_updates=40,
              supervised_steps=400, evaluation_repeats=16, trials_per_update=8,
              max_steps=12, hidden_size=32, n=3):
    """Persist every planned run; no adaptive seed or hyperparameter selection."""
    import torch
    from .diagnostics import (balanced_specs, behavior_metrics, paired_difference,
                              render_diagnostic_report, supervised_probe)
    from .recurrent import load_checkpoint, save_checkpoint
    from .training import TrainConfig, train

    n = validate_n(n)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('learning seeds must be nonempty and unique')
    for seed in seeds:
        if seed is None:
            raise TypeError('learning seeds must be explicit integers')
        validate_seed(seed)
    for name, value in (('updates', updates), ('short_updates', short_updates),
                        ('supervised_steps', supervised_steps), ('evaluation_repeats', evaluation_repeats),
                        ('trials_per_update', trials_per_update), ('max_steps', max_steps),
                        ('hidden_size', hidden_size)):
        positive_integer(value, name)
    if updates <= short_updates:
        raise ValueError('updates must be greater than short_updates')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {
        'version': 1, 'seeds': list(seeds), 'n': n, 'short_updates': short_updates,
        'long_updates': updates, 'supervised_steps': supervised_steps,
        'evaluation_repeats': evaluation_repeats, 'evaluation_trials': n*evaluation_repeats,
        'evaluation_seed': 918273, 'trials_per_update': trials_per_update,
        'max_steps': max_steps, 'episodes_per_trial': 2, 'hidden_size': hidden_size,
        'device': 'cpu', 'torch_version': str(torch.__version__),
        'design': 'fixed targets separately; variable short/long from the same optimization path',
        'uncertainty': 'target-stratified paired evaluation SE; learning-seed SD reported separately',
    }
    _write_json(directory/'protocol.json', protocol)
    report = {'schema_version': 1, 'protocol': protocol, 'fixed_runs': [],
              'variable_runs': [], 'supervised_runs': [], 'aggregates': {}}
    eval_seed = protocol['evaluation_seed']
    random_cache = {}

    def evaluate_run(result, *, target, update, run_dir, checkpoint, cached_initial=None):
        specs = balanced_specs(n=n, repeats=evaluation_repeats, fixed_action=target)
        options = dict(episodes_per_trial=2, max_steps=max_steps, seed=eval_seed)
        if target not in random_cache:
            random_cache[target] = behavior_metrics(None, specs, **options)
        model = load_checkpoint(checkpoint)
        initial = cached_initial if cached_initial is not None else behavior_metrics(result.initial_model, specs, **options)
        comparisons = {
            'initial': initial, 'random': random_cache[target],
            'trained': behavior_metrics(model, specs, **options),
            'hidden_reset': behavior_metrics(model, specs, reset_hidden=True, **options),
        }
        row = {
            'seed': result.config.seed, 'target': target, 'update': update,
            'config': asdict(result.config), 'comparisons': comparisons,
            'paired_vs_random': paired_difference(comparisons['trained'], comparisons['random']),
            'paired_memory_benefit': paired_difference(comparisons['trained'], comparisons['hidden_reset']),
            'checkpoint': str(checkpoint.relative_to(directory)),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'evaluation_seed': eval_seed, 'evaluation_specs': [asdict(spec) for spec in specs],
            'history': result.history[:update],
            'raw_record': str((run_dir/f'update_{update:04d}.json').relative_to(directory)),
        }
        _write_json(directory/row['raw_record'], row)
        return row

    for seed in seeds:
        print(f'[probe seed={seed}] {supervised_steps} supervised steps', file=sys.stderr, flush=True)
        probe = supervised_probe(seed=seed, steps=supervised_steps, hidden_size=hidden_size, n=n)
        report['supervised_runs'].append(probe)
        _write_json(directory/f'probe_seed{seed}.json', probe)

    for seed in seeds:
        for target in range(n):
            run_dir = directory/f'fixed_target{target}_seed{seed}'
            run_dir.mkdir()
            config = TrainConfig(n=n, seed=seed, family='action_bonus', fixed_action=target,
                                 updates=short_updates, hidden_size=hidden_size,
                                 trials_per_update=trials_per_update, max_steps=max_steps)
            print(f'[fixed seed={seed} target={target}] {short_updates} updates', file=sys.stderr, flush=True)
            def fixed_callback(update, model):
                if update == 0:
                    save_checkpoint(model, run_dir/'initial.pt')
                if update == short_updates:
                    save_checkpoint(model, run_dir/f'update_{update:04d}.pt')
            result = train(config, checkpoint_callback=fixed_callback)
            row = evaluate_run(result, target=target, update=short_updates, run_dir=run_dir,
                               checkpoint=run_dir/f'update_{short_updates:04d}.pt')
            report['fixed_runs'].append(row)
            print(f'  reward/step={row["comparisons"]["trained"]["mean_reward_per_step"]:.6f}', file=sys.stderr, flush=True)

    for seed in seeds:
        run_dir = directory/f'variable_seed{seed}'
        run_dir.mkdir()
        config = TrainConfig(n=n, seed=seed, family='action_bonus', updates=updates,
                             hidden_size=hidden_size, trials_per_update=trials_per_update,
                             max_steps=max_steps)
        print(f'[variable seed={seed}] {updates} updates; snapshots {short_updates}/{updates}', file=sys.stderr, flush=True)
        def variable_callback(update, model):
            if update == 0:
                save_checkpoint(model, run_dir/'initial.pt')
            if update in (short_updates, updates):
                save_checkpoint(model, run_dir/f'update_{update:04d}.pt')
            if update and update % 40 == 0:
                print(f'  update {update}/{updates}', file=sys.stderr, flush=True)
        result = train(config, checkpoint_callback=variable_callback)
        initial = None
        for update in (short_updates, updates):
            row = evaluate_run(result, target=None, update=update, run_dir=run_dir,
                               checkpoint=run_dir/f'update_{update:04d}.pt', cached_initial=initial)
            initial = row['comparisons']['initial']
            report['variable_runs'].append(row)
            print(f'  snapshot {update} reward/step={row["comparisons"]["trained"]["mean_reward_per_step"]:.6f}', file=sys.stderr, flush=True)

    report['aggregates'] = {
        'fixed_by_target': {str(target): _aggregate([row for row in report['fixed_runs'] if row['target'] == target])
                            for target in range(n)},
        'variable_by_update': {str(update): _aggregate([row for row in report['variable_runs'] if row['update'] == update])
                              for update in (short_updates, updates)},
    }
    _write_json(directory/'diagnostic.json', report)
    with (directory/'report.html').open('x', encoding='utf-8') as stream:
        stream.write(render_diagnostic_report(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    for name, default in (('n', 3), ('updates', 120), ('short_updates', 40), ('supervised_steps', 400),
                           ('evaluation_repeats', 16), ('trials_per_update', 8),
                           ('max_steps', 12), ('hidden_size', 32)):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    args = parser.parse_args()
    try:
        import torch
        torch.set_num_threads(1)
        record = run_study(**vars(args))
    except (ImportError, TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'fixed_runs': len(record['fixed_runs']),
                      'variable_snapshots': len(record['variable_runs']),
                      'supervised_runs': len(record['supervised_runs']), 'aggregates': record['aggregates']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
