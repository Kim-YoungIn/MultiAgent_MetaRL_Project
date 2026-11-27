"""Compare prior choices after common random probes, on absolute and gain scales."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.fast_exploration_v2.analyze import bootstrap_indices, file_sha256, statistics, write_csv


def run(root, out):
    rows = []
    for version, directory in (('v1', 'fast_exploration_scale_final'), ('v2', 'fast_exploration_v2_final')):
        source = root / directory
        manifest = json.loads((source / 'manifest.json').read_text())
        complete = json.loads((source / 'complete.json').read_text())
        blocks = manifest['config']['blocks']
        assert complete['blocks'] == blocks == (24 if version == 'v1' else 36)
        records, series = [], defaultdict(list)
        for i in range(blocks):
            path = source / f'block_{i:03d}.npz'
            rec = json.loads(path.with_suffix('.json').read_text())
            assert rec['block'] == i and file_sha256(path) == rec['raw_sha256']
            records.append(rec)
            with np.load(path) as raw:
                methods, ks = list(raw['methods']), list(raw['ks'])
                assert ks == [0, 4, 8, 16, 24]
                if version == 'v1':
                    method = methods.index('random')
                    reward_names = dict(ips='reward_auc', pooled='pooled_auc', zero='zero_auc')
                    rewards = {name: raw[key + '_assessment'][method] for name, key in reward_names.items()}
                    for key in reward_names.values():
                        np.testing.assert_array_equal(raw[key + '_exploration'][method],
                                                      raw['reward_auc_exploration'][method])
                    regimes = {'live': rewards}
                else:
                    names = {name: methods.index(name + '_random') for name in ('ips', 'pooled')}
                    for field in ('probe_actions', 'probe_rewards'):
                        tape = raw[field]
                        np.testing.assert_array_equal(tape[names['ips']], tape[names['pooled']])
                    regimes = {}
                    for regime, key in (('live', 'natural_rewards'), ('frozen', 'controlled_rewards')):
                        tape = raw[key]
                        regimes[regime] = {name: tape[j, ..., 0] for name, j in names.items()}
                for regime, rewards in regimes.items():
                    for name in rewards.keys() - {'pooled'}:
                        delta = rewards[name] - rewards['pooled']
                        for family in ('all', *np.unique(raw['families'])):
                            mask = raw['families'] == family if family != 'all' else np.ones(len(raw['families']), bool)
                            for index, k in enumerate(ks):
                                d = delta[index, mask].mean()
                                change = (delta[index, mask] - delta[0, mask]).mean()
                                series[regime, name, family, k, 'absolute_gap'].append(d)
                                series[regime, name, family, k, 'change_in_gap_from_k0'].append(change)
        picks = bootstrap_indices(records)
        for (regime, name, family, k, metric), values in sorted(series.items()):
            rows.append(dict(version=version, endpoint=regime, contrast=name + '-pooled', family=family,
                             k=k, metric=metric, blocks=blocks, **statistics(values, picks)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'prior.csv', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=Path('research/retrospective/evidence'))
    args = parser.parse_args()
    run(args.runs, args.out)
