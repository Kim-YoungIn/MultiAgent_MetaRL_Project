"""Check exact structured-game fingerprints against available earlier panels."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_maps(files, key):
    values = set()
    for path in files:
        with np.load(path) as data:
            values.update(data[key].tolist())
    return values


def fingerprints(node):
    if isinstance(node,list):
        for row in node:
            yield from fingerprints(row)
    elif isinstance(node,dict):
        if 'fingerprint' in node:
            yield node['fingerprint']
        if 'fingerprints' in node:
            yield from node['fingerprints']
        for name, value in node.items():
            if name not in ('game','rows','probe','checkpoints'):
                yield from fingerprints(value)


def historical(parent):
    schemas = [
        ('active_probe_v1','block_*/*.npz','game_fingerprint'),
        ('probe_stakes_v1','block_*.npz','game_fingerprint'),
        ('opponent_adaptation_v1','block_*/train.npz','fingerprint'),
        ('opponent_adaptation_v1','block_*/test.npz','fingerprint'),
        ('absolute_exploration_dev','block_*/*.npz','game_fingerprint'),
    ]
    panels = {}
    for name,pattern,key in schemas:
        values=read_maps((parent/name).glob(pattern),key)
        if values:
            panels.setdefault(name,set()).update(values)
    for name in ('structured_transfer_v1','structured_scale_v1','structured_oracle_v1'):
        path = parent/name
        files = list(path.glob('block_*/train.json'))+list(path.glob('block_*/test.json'))
        if (path/'pairs.json').exists():
            files.append(path/'pairs.json')
        if files:
            panels[name]=set(fp for p in files for fp in fingerprints(json.loads(p.read_text())))
    return panels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--history',type=Path,default=Path('runs'))
    args = p.parse_args()
    new = read_maps(args.run.glob('block_*/*.npz'),'game_fingerprint')
    expected = json.loads((args.run/'complete.json').read_text())['distinct_games']
    if len(new) != expected:
        raise ValueError('unexpected final fingerprint count')
    old = historical(args.history)
    report = dict(final=len(new), prior_counts={k:len(v) for k,v in old.items()},
                  prior_union=len(set().union(*old.values())),
                  overlap_by_panel={k:len(new&v) for k,v in old.items()})
    if any(report['overlap_by_panel'].values()):
        raise ValueError(report)
    (args.run/'previous_splits.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
