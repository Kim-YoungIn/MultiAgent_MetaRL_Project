"""Compute paired block statistics and tabular summaries from saved results."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ('learned', 'blind', 'random', 'greedy')
REGIMES = ('reward_auc', 'controlled_auc', 'pooled_auc', 'zero_auc')


def load(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    cfg = manifest['config']
    files = sorted(root.glob('block_*.json'))
    expected = cfg['blocks']
    if len(files) != expected:
        raise ValueError(f'expected {expected} blocks, found {len(files)}')
    records = []
    for index, path in enumerate(files):
        rec = json.loads(path.read_text())
        weights = path.with_suffix('.pt')
        if rec['block'] != index or not weights.is_file():
            raise ValueError(f'incomplete or misnumbered block {path}')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != rec['weights_sha256']:
            raise ValueError(f'checkpoint checksum mismatch at {weights}')
        for regime in REGIMES:
            result = rec[regime]
            ks = np.asarray(result['ks'])
            for method in METHODS:
                curve = np.asarray(result['curves'][method], dtype=float)
                if not np.isfinite(curve).all() or not np.isclose(
                    np.trapezoid(curve, ks) / ks[-1], result['auc'][method],
                    rtol=1e-8, atol=1e-10
                ):
                    raise ValueError(f'invalid AUC for {path}, {regime}, {method}')
        records.append(rec)
    return manifest, records


def bootstrap_indices(records, draws=20000, seed=882734):
    """Resample blocks independently within each fixed master seed."""
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec.get('master_seed', 'all')].append(i)
    sizes = {len(group) for group in groups.values()}
    if len(sizes) != 1:
        raise ValueError('master seeds must have equal numbers of blocks')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(group, size=(draws, len(group)), replace=True)
                           for group in groups.values()], axis=1)


def summary(values, indices):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('non-finite outcomes')
    ci = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), block_sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
                ci_low=float(ci[0]), ci_high=float(ci[1]))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(root):
    root = Path(root)
    manifest, records = load(root)
    draws = bootstrap_indices(records)
    rows, pairs, raw = [], [], []
    for regime in REGIMES:
        k_grid = records[0][regime]['ks']
        if any(record[regime]['ks'] != k_grid for record in records):
            raise ValueError(f'inconsistent budgets in {regime}')
        for method in METHODS:
            arrays = {
                'absolute_auc': np.array([r[regime]['auc'][method] for r in records]),
                'initial': np.array([r[regime]['curves'][method][0] for r in records]),
            }
            for k_index, k in enumerate(k_grid):
                arrays[f'reward_k{k}'] = np.array([r[regime]['curves'][method][k_index] for r in records])
                arrays[f'total_k{k}'] = np.array([r[regime]['total_return'][method][k_index] for r in records])
                arrays[f'first_assessment_k{k}'] = np.array([r[regime]['assessment_steps'][method][str(k)][0] for r in records])
                if k_index:
                    previous = k_grid[k_index-1]
                    arrays[f'slope_k{previous}_{k}'] = (
                        arrays[f'reward_k{k}']-arrays[f'reward_k{previous}']
                    )/(k-previous)
            for metric, values in arrays.items():
                rows.append(dict(regime=regime,method=method,metric=metric,**summary(values,draws)))
                for i, value in enumerate(values):
                    raw.append(dict(block=records[i]['block'],
                                    master_seed=records[i].get('master_seed', manifest.get('seed')),
                                    seed_block=records[i].get('seed_block', records[i]['block']),
                                    regime=regime,method=method,metric=metric,value=float(value)))
        for method in METHODS[1:]:
            values = np.array([r[regime]['auc']['learned']-r[regime]['auc'][method] for r in records])
            pairs.append(dict(regime=regime,contrast=f'learned_minus_{method}',metric='absolute_auc',
                              **summary(values,draws)))
    # Same teachers and seeds under prior changes, paired within each block.
    for regime in ('pooled_auc', 'zero_auc'):
        for method in METHODS:
            values = np.array([r['reward_auc']['auc'][method]-r[regime]['auc'][method] for r in records])
            pairs.append(dict(regime=regime,contrast=f'trained_prior_minus_{regime.removesuffix("_auc")}_{method}',
                              metric='absolute_auc',**summary(values,draws)))
    report = dict(blocks=len(records), master_seeds=sorted({str(r.get('master_seed',manifest.get('seed'))) for r in records}),
                  bootstrap=dict(draws=20000, seed=882734, unit='independent training block',
                                 stratified_by='master seed', level=.95),
                  paired_contrasts=pairs, measures=rows,
                  zero_shot_equal=all(r['reward_auc']['r0_equal'] for r in records))
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out/'summary.csv',rows)
    write_csv(out/'paired.csv',pairs)
    write_csv(out/'block_metrics.csv',raw)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = make_report(args.output)
    print(json.dumps({'blocks': result['blocks'], 'paired_contrasts': result['paired_contrasts']}, indent=2))
