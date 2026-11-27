"""Independently audit D-031 receipts and recompute paired effects from trial curves."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import runpy

import numpy as np

KS = np.array([0, 3, 6, 12, 24])


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_receipt(path):
    expected = json.loads(path.read_text())
    method = path.name.removesuffix('.receipt.json')
    names = {method + '.json.gz', method + '.trials.jsonl.gz'}
    if set(expected) != names:
        raise ValueError(f'unexpected evaluation receipt contents: {path}')
    for name, checksum in expected.items():
        if sha(path.parent / name) != checksum:
            raise ValueError(f'evaluation checksum mismatch: {path.parent / name}')


def trial_values(folder, method):
    outcomes = {}
    path = folder / f'{method}.trials.jsonl.gz'
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] != 'fresh_rules' or row['profile'] not in ('steady', 'sarsa'):
                continue
            key = (row['rule'], row['profile'], row['rep'])
            if key in outcomes:
                raise ValueError(f'duplicate trial {path}: {key}')
            curve = row['curve']
            if [point['k'] for point in curve] != KS.tolist():
                raise ValueError(f'unexpected budgets in {path}: {key}')
            y = np.array([point['return'] for point in curve])
            auc = float(np.diff(KS) @ ((y[:-1] + y[1:]) / 2 - y[0]) / KS[-1])
            if not np.isclose(auc, row['auc'], atol=1e-12, rtol=0):
                raise ValueError(f'trial AUC disagrees with curve in {path}: {key}')
            outcomes[key] = (row['family'], auc, row['seed'], y[0])
    if len(outcomes) != 64 * 2 * 8:
        raise ValueError(f'incomplete fresh-rule evaluation in {path}: {len(outcomes)}')
    return outcomes


def paired_cell(folder):
    m, p = trial_values(folder, 'M'), trial_values(folder, 'P')
    if m.keys() != p.keys():
        raise ValueError(f'unpaired trial keys: {folder}')
    groups = defaultdict(list)
    for key, a in m.items():
        b = p[key]
        if a[0] != b[0] or a[2] != b[2] or not np.isclose(a[3], b[3], atol=1e-12, rtol=0):
            raise ValueError(f'unpaired metadata or zero-shot return: {folder} {key}')
        groups[key[1], a[0]].append((a[1], b[1]))
    if {key[0] for key in groups} != {'steady', 'sarsa'}:
        raise ValueError(f'missing opponent profile: {folder}')
    profiles = []
    for profile in ('steady', 'sarsa'):
        family_means = [np.mean(values, axis=0) for (pname, _), values in groups.items() if pname == profile]
        profiles.append(np.mean(family_means, axis=0))
    means = np.mean(profiles, axis=0)
    return float(means[0]), float(means[1])


def summarize(values):
    if not values:
        return {'models': 0, 'mean': None, 'ci_low': None, 'ci_high': None}
    catalogs = sorted({key[0] for key in values})
    by_catalog = {c: sorted(k for k in values if k[0] == c) for c in catalogs}
    estimate = float(np.mean([np.mean([values[k] for k in by_catalog[c]]) for c in catalogs]))
    if len(values) < 2:
        return {'models': len(values), 'mean': estimate, 'ci_low': None, 'ci_high': None}
    # Same independently implemented catalog/replica resampling as the sealed report.
    rng = np.random.default_rng(71001)
    choices = rng.integers(len(catalogs), size=(10000, len(catalogs)))
    samples = np.empty(len(choices))
    for draw, selected in enumerate(choices):
        cat_means = []
        for index in selected:
            keys = by_catalog[catalogs[index]]
            chosen = rng.choice(len(keys), size=len(keys), replace=True)
            cat_means.append(np.mean([values[keys[i]] for i in chosen]))
        samples[draw] = np.mean(cat_means)
    low, high = np.quantile(samples, [.025, .975])
    return {'models': len(values), 'mean': estimate, 'ci_low': float(low), 'ci_high': float(high)}


def plot_results(output, groups, contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    panels = [
        (groups, [('primary/n3', 'Original condition'), ('rules_low/n3', '8 training rules'),
                  ('rules_high/n3', '128 training rules'), ('focused/n3', 'Focused family'),
                  ('q_only/n3', 'Q opponent'), ('sarsa_only/n3', 'SARSA opponent'),
                  ('n5/n5', '5 actions')], 'Within each training condition', 'M minus P'),
        (contrasts, [('rule_count', '128 minus 8 rules'), ('low_to_primary', '32 minus 8 rules'),
                     ('primary_to_high', '128 minus 32 rules'), ('family', 'Broad minus focused'),
                     ('opponent_diversity', 'Mixed minus single opponents')],
         'Paired changes in the M–P gap', 'Difference in M minus P'),
    ]
    for ax, (table, labels, title, xlabel) in zip(axes, panels, strict=True):
        for i, (key, _) in enumerate(labels):
            point = table[key]
            ax.plot([point['ci_low'], point['ci_high']], [i, i], color='#34697a', linewidth=2)
            ax.plot(point['mean'], i, 'o', color='#214757', markersize=5)
        ax.set_yticks(range(len(labels)), [label for _, label in labels])
        ax.invert_yaxis()
        ax.axvline(0, color='0.45', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel=xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.18)
    fig.savefig(output / 'diversity.png', dpi=160)
    plt.close(fig)


def audit(root, output, allow_incomplete):
    seal = json.loads((root / 'study.json').read_text())
    if seal['protocol'] != 'D-031' or seal['development'] or seal['device'] != 'cpu':
        raise ValueError('expected the sealed full-budget CPU study')
    for name, expected in seal['sources'].items():
        path = root.parent / 'source' / name
        if sha(path) != expected:
            raise ValueError(f'sealed source checksum mismatch: {path}')
    if json.loads((root / 'panel_sha256.json').read_text()) != sha(root / 'panels/panel.json'):
        raise ValueError('sealed panel checksum mismatch')
    expected_cells = runpy.run_path(str(root.parent / 'source/experiment/protocol.py'))['plan'](False)
    cells = []
    missing_fits, missing_evaluations = [], []
    fit_count = receipt_count = 0
    for cell in expected_cells:
        folder = root / 'cells' / cell['condition'] / f'c{cell["catalog"]}' / f'j{cell["replica"]}'
        for method in cell['methods']:
            training = folder / 'training' / method
            ledger_path = training / 'ledger.json'
            if not ledger_path.is_file():
                missing_fits.append(str(ledger_path))
                continue
            ledger = json.loads(ledger_path.read_text())
            # The original first primary fits were dispatched one method at a
            # time. That dispatch list does not change a model's training input.
            dispatch = ledger['cell']['methods']
            saved_cell = dict(ledger['cell'], methods=list(cell['methods']))
            if (ledger['method'] != method or ledger['device'] != 'cpu'
                    or method not in dispatch or not set(dispatch) <= set(cell['methods'])
                    or saved_cell != json.loads(json.dumps(cell))
                    or len(ledger['updates']) != 800
                    or [row['update'] for row in ledger['updates']] != list(range(1, 801))
                    or ledger['real_steps'] != 800 * 32 * 48
                    or ledger['erasure_steps'] != 800 * 32 * 24):
                raise ValueError(f'invalid training ledger: {ledger_path}')
            for filename, key in (('training_state.pt', 'checkpoint_sha256'), ('model_config.pt', 'model_config_sha256')):
                if sha(training / filename) != ledger[key]:
                    raise ValueError(f'training checksum mismatch: {training / filename}')
            fit_count += 1
        for n in ((3, 4, 5) if cell['condition'] in ('primary', 'n5') else (cell['train_n'],)):
            methods = (*cell['methods'], 'random', 'initial', 'epsilon', 'balanced') if n == cell['train_n'] else ('M', 'P')
            for method in methods:
                receipt = folder / f'n{n}' / f'{method}.receipt.json'
                if receipt.exists():
                    verify_receipt(receipt)
                    receipt_count += 1
                else:
                    missing_evaluations.append(str(receipt))
            pair_folder = folder / f'n{n}'
            if all((pair_folder / f'{method}.receipt.json').is_file() for method in ('M', 'P')):
                m, p = paired_cell(pair_folder)
                cells.append(dict(condition=cell['condition'], n=n, catalog=cell['catalog'],
                                  replica=cell['replica'], M=m, P=p, M_minus_P=m-p))
    if (missing_fits or missing_evaluations) and not allow_incomplete:
        raise ValueError(f'incomplete study: {len(missing_fits)} fits and {len(missing_evaluations)} evaluations missing')
    groups = defaultdict(dict)
    for row in cells:
        groups[row['condition'], row['n']][row['catalog'], row['replica']] = row['M_minus_P']
    results = {f'{condition}/n{n}': summarize(values) for (condition, n), values in sorted(groups.items())}
    contrasts = {}
    for name, low, high in (
        ('rule_count', ('rules_low', 3), ('rules_high', 3)),
        ('low_to_primary', ('rules_low', 3), ('primary', 3)),
        ('primary_to_high', ('primary', 3), ('rules_high', 3)),
        ('family', ('focused', 3), ('primary', 3)),
        ('primary_n3_to_n4', ('primary', 3), ('primary', 4)),
        ('primary_n3_to_n5', ('primary', 3), ('primary', 5)),
        ('n5_n5_to_n3', ('n5', 5), ('n5', 3)),
    ):
        left, right = groups[low], groups[high]
        contrasts[name] = summarize({key: right[key] - left[key] for key in left.keys() & right.keys()})
    mixed, q, s = groups['primary', 3], groups['q_only', 3], groups['sarsa_only', 3]
    contrasts['opponent_diversity'] = summarize({key: mixed[key] - (q[key] + s[key]) / 2
                                                 for key in mixed.keys() & q.keys() & s.keys()})
    source_report = json.loads((root / 'report.json').read_text()) if (root / 'report.json').exists() else {}
    comparisons = 0
    for name, calc in results.items():
        source = source_report.get('groups', {}).get(name + '/fresh_rules/mixed/M-P')
        if source is None or source['n_models'] != calc['models']:
            continue
        if not np.isclose(calc['mean'], source['mean'], rtol=0, atol=1e-12):
            raise ValueError(f'report mean disagrees with raw data: {name}')
        if not np.allclose([calc['ci_low'], calc['ci_high']], source['confidence_interval'], rtol=0, atol=1e-12):
            raise ValueError(f'report interval disagrees with raw data: {name}')
        comparisons += 1
    contrast_comparisons = 0
    for name, calc in contrasts.items():
        source = source_report.get('block_contrasts', {}).get(name)
        if source is None or source['n_models'] != calc['models'] or calc['models'] < 2:
            continue
        if not np.isclose(calc['mean'], source['mean'], rtol=0, atol=1e-12):
            raise ValueError(f'report contrast disagrees with raw data: {name}')
        if not np.allclose([calc['ci_low'], calc['ci_high']], source['confidence_interval'], rtol=0, atol=1e-12):
            raise ValueError(f'report contrast interval disagrees with raw data: {name}')
        contrast_comparisons += 1
    if not allow_incomplete:
        # Three additional cross-action comparisons have no entry in the
        # original reporter. The other five and all eleven group results do.
        if source_report.get('missing_required_files') != [] or comparisons != 11 or contrast_comparisons != 5:
            raise ValueError('rebuild the sealed final report before publishing verification')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cells.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('condition', 'n', 'catalog', 'replica', 'M', 'P', 'M_minus_P'))
        writer.writeheader()
        writer.writerows(cells)
    result = dict(study=str(root), study_sha256=sha(root / 'study.json'), panel_sha256=sha(root / 'panels/panel.json'),
                  report_sha256=sha(root / 'report.json') if source_report else None,
                  sealed_python=seal['python'], sealed_torch=seal['torch'], verifier_python=platform.python_version(),
                  verified_source_files=len(seal['sources']),
                  verified_fits=fit_count, expected_fits=375, verified_evaluation_receipts=receipt_count,
                  missing_fits=missing_fits, missing_evaluations=missing_evaluations,
                  paired_cells=len(cells), trial_curves_verified=len(cells) * 2 * 1024,
                  source_report_groups_compared=comparisons, source_report_contrasts_compared=contrast_comparisons,
                  groups=results, contrasts=contrasts)
    (output / 'completion.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not allow_incomplete:
        plot_results(output, results, contrasts)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    audit(args.study, args.output, args.allow_incomplete)


if __name__ == '__main__':
    main()
