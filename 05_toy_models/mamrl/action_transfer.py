"""Evaluate one fixed exploration checkpoint across action counts.

Comparisons with random exploration are paired within each action count.
Different action counts define different games, not identical paired tasks.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

from .common import positive_integer, validate_seed
from .diverse_catalog import validate_diverse_manifest
from .rule_metrics import stable_digest
from .run_files import write_json_exclusive


def run_action_transfer(checkpoint, manifests, output_dir, *, steps=12,
                        assessment_steps=48, reps=1, seed=918273,
                        device='cpu', batch_size=1):
    from .meta_exploration import evaluate_meta, meta_tasks, neural_factory, RandomExplorer
    from .recurrent import load_checkpoint
    import torch

    for name, value in (('steps', steps), ('assessment_steps', assessment_steps),
                        ('reps', reps), ('batch_size', batch_size)):
        positive_integer(value, name)
    if validate_seed(seed) is None:
        raise ValueError('explicit evaluation seed required')
    if not manifests:
        raise ValueError('at least one action-count manifest required')
    for n, manifest in manifests.items():
        validate_diverse_manifest(manifest)
        if type(n) is not int or n != manifest['n']:
            raise ValueError('manifest key and action count differ')
    model = load_checkpoint(checkpoint, device=device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    weights = Path(checkpoint).read_bytes()
    (root/'policy.pt').write_bytes(weights)
    (root/'sources').mkdir()
    hashes = {}
    for file in sorted(Path(__file__).parent.glob('*.py')):
        data = file.read_bytes()
        (root/'sources'/file.name).write_bytes(data)
        hashes[file.name] = hashlib.sha256(data).hexdigest()
    evaluations = {}
    simulation_steps = 0
    for n, manifest in sorted(manifests.items()):
        write_json_exclusive(root/f'manifest_n{n}.json', manifest)
        evaluations[str(n)] = {}
        for split in ('fresh_rules', 'structural'):
            tasks = meta_tasks(manifest, split, opponent_suite='mixed_learning')
            group = {}
            for method, factory in (('trained', neural_factory(model)), ('random', RandomExplorer)):
                value = evaluate_meta(factory, tasks, steps=steps, assessment_steps=assessment_steps,
                                      reps=reps, seed=seed, batch_size=batch_size)
                rows = []
                for trial in value['trials']:
                    events = trial['exploration']
                    repeat = statistics.mean(e['actions'][0] == e['observations'][0]['previous_own_action']
                                             for e in events)
                    rows.append({key: trial[key] for key in ('task_id', 'rule_id', 'family', 'seed', 'rep',
                                'opponent_profile', 'assessment_mean', 'meta_reward', 'exploration_mean')}
                                | {'repeat_fraction': repeat, 'repeat_minus_uniform': repeat-1/n,
                                   'action_coverage': len({e['actions'][0] for e in events})/n})
                group[method] = {'summary': value['summary'], 'by_family': value['by_family'], 'rows': rows}
                simulation_steps += len(rows)*(steps+2*assessment_steps)
            group['paired_trained_minus_random'] = [a['assessment_mean']-b['assessment_mean']
                for a, b in zip(group['trained']['rows'], group['random']['rows'])]
            evaluations[str(n)][split] = group
    if any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items()):
        raise RuntimeError('evaluation changed checkpoint weights')
    record = {'version': 1, 'protocol': {'K': steps, 'H': assessment_steps, 'reps': reps,
        'seed': seed, 'device': str(model.device), 'batch_size': batch_size,
        'checkpoint_sha256': hashlib.sha256(weights).hexdigest(), 'source_sha256': hashes,
        'manifest_sha256': {str(n): stable_digest(m) for n, m in manifests.items()},
        'simulation_steps': simulation_steps, 'weights_unchanged': True,
        'scope': 'same checkpoint across n; training provenance belongs to original run; no cross-n identical-task claim',
        'opponent_suite': 'mixed_learning'}, 'evaluation': evaluations}
    write_json_exclusive(root/'study.json', record)
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>행동 공간 전이</title>',
             '<h1>같은 탐색 가중치의 행동 공간별 평가</h1><p>학생과 상대는 계속 학습합니다. '
             '행동 수가 다르면 게임도 다르므로 생보상 차이를 행동 수만의 인과효과로 해석하지 않습니다. '
             '일반화 주장은 원래 학습 run의 행동 수와 함께 판단해야 합니다.</p>',
             '<a href="study.json">원자료</a><table border="1"><tr><th>n</th><th>평가</th>'
             '<th>학습 탐색</th><th>랜덤</th><th>paired 차이 평균</th></tr>']
    for n, splits in evaluations.items():
        for split, group in splits.items():
            parts.append(f'<tr><td>{n}</td><td>{html.escape(split)}</td>'
                f'<td>{group["trained"]["summary"]["assessment_mean"]:.4f}</td>'
                f'<td>{group["random"]["summary"]["assessment_mean"]:.4f}</td>'
                f'<td>{statistics.mean(group["paired_trained_minus_random"]):.4f}</td></tr>')
    (root/'report.html').write_text(''.join(parts)+'</table></html>', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    for name, default in (('steps', 12), ('assessment-steps', 48), ('reps', 1),
                           ('seed', 918273), ('batch-size', 1)):
        parser.add_argument('--'+name, type=int, default=default)
    args = parser.parse_args()
    manifests = {}
    for path in args.manifests:
        value = json.loads(path.read_text())
        if value['n'] in manifests:
            parser.error('only one manifest per action count allowed')
        manifests[value['n']] = value
    run_action_transfer(args.checkpoint, manifests, args.output_dir, steps=args.steps,
        assessment_steps=args.assessment_steps, reps=args.reps, seed=args.seed,
        batch_size=args.batch_size, device=args.device)


if __name__ == '__main__':
    main()
