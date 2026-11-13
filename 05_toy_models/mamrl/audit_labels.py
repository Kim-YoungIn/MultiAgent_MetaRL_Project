"""Audit reward/gradient calculations and controlled cyclic action relabeling."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

from .common import positive_integer, validate_seed


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _summary(rows):
    values = [row['comparisons']['trained']['mean_reward_per_step'] for row in rows]
    return {'mean_reward': statistics.mean(values),
            'sd_across_runs': statistics.stdev(values) if len(values) > 1 else None,
            'run_count': len(rows), 'learning_seed_count': len({row['seed'] for row in rows}),
            'mean_initial_to_trained': statistics.mean(row['initial_to_trained']['mean'] for row in rows)}


def load_physical_model(checkpoint, permutation):
    """Restore internal-coordinate weights together with their physical mapping."""
    from .label_audit import PermutedActorCritic
    from .recurrent import load_checkpoint

    base = load_checkpoint(checkpoint)
    physical = PermutedActorCritic(hidden_size=base.hidden_size, seed=0, permutation=tuple(permutation))
    physical.load_state_dict(base.state_dict())
    return physical


def run_audit(output_dir, *, seeds=(7, 17, 27), updates=40, evaluation_repeats=16,
              trials_per_update=8, max_steps=12, hidden_size=32):
    import torch
    from .diagnostics import balanced_specs, behavior_metrics, paired_difference
    from .label_audit import (PermutedActorCritic, actor_gradient_audit, loss_gradient_audit,
                              render_label_audit_report, reward_audit)
    from .recurrent import RecurrentActorCritic, load_checkpoint, save_checkpoint
    from .training import TrainConfig, train

    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('learning seeds must be nonempty and unique')
    for seed in seeds:
        if seed is None:
            raise TypeError('seeds must be explicit nonnegative integers')
        validate_seed(seed)
    for name, value in (('updates', updates), ('evaluation_repeats', evaluation_repeats),
                        ('trials_per_update', trials_per_update), ('max_steps', max_steps),
                        ('hidden_size', hidden_size)):
        positive_integer(value, name)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    source_dir = directory/'sources'
    source_dir.mkdir()
    source_hashes = {}
    for name in ('audit_labels.py', 'label_audit.py', 'training.py', 'recurrent.py', 'ppo.py',
                 'diagnostics.py', 'environment.py', 'rules.py', 'tensor.py'):
        contents = Path(__file__).with_name(name).read_bytes()
        (source_dir/name).write_bytes(contents)
        source_hashes[name] = hashlib.sha256(contents).hexdigest()
    protocol = {
        'version': 1, 'seeds': list(seeds), 'n': 3, 'updates': updates,
        'physical_targets': [0,1,2], 'shifts': [0,1,2],
        'permutation_definition': 'physical_to_internal[a]=(a+shift)%3',
        'evaluation_repeats': evaluation_repeats, 'evaluation_trials': 3*evaluation_repeats,
        'evaluation_seed': 918273, 'trials_per_update': trials_per_update,
        'max_steps': max_steps, 'episodes_per_trial': 2, 'hidden_size': hidden_size,
        'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
        'torch_threads': torch.get_num_threads(), 'device': 'cpu', 'source_sha256': source_hashes,
        'identity_policy': 'new matched runs, not reused historical checkpoints',
        'checkpoint_coordinates': 'internal; restore using load_physical_model(path,permutation)',
        'scope': 'joint GRU input and actor candidate label intervention; no architecture adoption',
    }
    _json(directory/'protocol.json', protocol)
    record = {'schema_version': 1, 'protocol': protocol, 'reward_audit': reward_audit(),
              'initial_gradient_audits': [], 'loss_gradient_audits': [], 'runs': [], 'aggregates': {}}
    for seed in seeds:
        print(f'[gradients seed={seed}] exact actor and first-batch PPO decomposition', file=sys.stderr, flush=True)
        initial = RecurrentActorCritic(hidden_size=hidden_size, seed=seed)
        record['initial_gradient_audits'].append({'seed': seed, 'audit': actor_gradient_audit(initial)})
        for target in range(3):
            config = TrainConfig(seed=seed, fixed_action=target, updates=updates,
                                 hidden_size=hidden_size, trials_per_update=trials_per_update, max_steps=max_steps)
            record['loss_gradient_audits'].append({'seed': seed, 'physical_target': target,
                                                   'audit': loss_gradient_audit(initial, config)})
    _json(directory/'gradient_audits.json', {'reward_audit': record['reward_audit'],
                                           'initial_gradient_audits': record['initial_gradient_audits'],
                                           'loss_gradient_audits': record['loss_gradient_audits']})
    random_cache = {}
    for seed in seeds:
        for shift in range(3):
            permutation = tuple((a+shift)%3 for a in range(3))
            for target in range(3):
                run_dir = directory/f'seed{seed}_shift{shift}_target{target}'
                run_dir.mkdir()
                config = TrainConfig(seed=seed, fixed_action=target, updates=updates,
                                     hidden_size=hidden_size, trials_per_update=trials_per_update,
                                     max_steps=max_steps)
                print(f'[seed={seed} shift={shift} target={target} internal={permutation[target]}] {updates} updates',
                      file=sys.stderr, flush=True)
                def factory(*, hidden_size, seed, mapping=permutation):
                    return PermutedActorCritic(hidden_size=hidden_size, seed=seed, permutation=mapping)
                result = train(config, model_factory=factory)
                initial_checkpoint = run_dir/'initial_internal_weights.pt'
                checkpoint = run_dir/'internal_weights.pt'
                save_checkpoint(result.initial_model, initial_checkpoint)
                save_checkpoint(result.model, checkpoint)
                restored = load_physical_model(checkpoint, permutation)
                specs = balanced_specs(repeats=evaluation_repeats, fixed_action=target)
                options = dict(episodes_per_trial=2, max_steps=max_steps, seed=918273)
                if target not in random_cache:
                    random_cache[target] = behavior_metrics(None, specs, **options)
                comparisons = {
                    'initial': behavior_metrics(result.initial_model, specs, **options),
                    'random': random_cache[target],
                    'trained': behavior_metrics(restored, specs, **options),
                }
                row = {
                    'seed': seed, 'physical_target': target, 'internal_target': permutation[target],
                    'permutation': list(permutation), 'shift': shift, 'config': asdict(config),
                    'comparisons': comparisons,
                    'initial_to_trained': paired_difference(comparisons['trained'], comparisons['initial']),
                    'checkpoint': str(checkpoint.relative_to(directory)),
                    'initial_checkpoint': str(initial_checkpoint.relative_to(directory)),
                    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'raw_record': str((run_dir/'run.json').relative_to(directory)),
                    'history': result.history, 'evaluation_specs': [asdict(s) for s in specs],
                    'evaluation_seed': 918273,
                    'trained_gradient_audit': actor_gradient_audit(load_checkpoint(checkpoint)),
                }
                _json(directory/row['raw_record'], row)
                record['runs'].append(row)
                print(f'  reward/step={comparisons["trained"]["mean_reward_per_step"]:.6f}', file=sys.stderr, flush=True)
    record['aggregates'] = {
        'by_internal_target': {str(t): _summary([r for r in record['runs'] if r['internal_target'] == t])
                               for t in range(3)},
        'by_physical_target': {str(t): _summary([r for r in record['runs'] if r['physical_target'] == t])
                               for t in range(3)},
    }
    _json(directory/'audit.json', record)
    with (directory/'report.html').open('x', encoding='utf-8') as stream:
        stream.write(render_label_audit_report(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7,17,27])
    for name, default in (('updates',40), ('evaluation_repeats',16), ('trials_per_update',8),
                          ('max_steps',12), ('hidden_size',32)):
        parser.add_argument('--'+name.replace('_','-'), type=int, default=default)
    args = parser.parse_args()
    try:
        import torch
        torch.set_num_threads(1)
        record = run_audit(**vars(args))
    except (ImportError, TypeError, ValueError, OSError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'runs': len(record['runs']),
                      'reward_audit': record['reward_audit'], 'aggregates': record['aggregates']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
