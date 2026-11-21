"""Export paired game/reset records while retaining full per-turn NPZ rewards."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .verify import REGIMES


def export(root):
    cfg = json.loads((root/'manifest.json').read_text())['config']
    result = root/'analysis'/'trials.csv.gz'
    with gzip.open(result,'wt',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['block','master_seed','seed_block','game_seed','reset_seed','family',
                         'regime','method','k','assessment_mean','first_assessment','exploration_sum','total_return'])
        for index in range(cfg['blocks']):
            path = root/f'block_{index:03d}'
            record = json.loads(path.with_suffix('.json').read_text())
            with np.load(path.with_suffix('.npz')) as raw:
                repeats = cfg['test_resets']
                ids = np.repeat(raw['game_seeds'],repeats)
                samples = list(zip(ids,raw['reset_seeds'],raw['families'],strict=True))
                for regime in REGIMES:
                    pay = raw[regime+'_assessment']
                    exploration = raw[regime+'_exploration']
                    for mi, method in enumerate(raw['methods']):
                        for ki,k in enumerate(raw['ks']):
                            turns = pay[mi,ki]
                            writer.writerows(
                                (index,record['master_seed'],record['seed_block'],g,s,f,regime,method,k,m,a,e,t)
                                for (g,s,f),m,a,e,t in zip(samples,turns.mean(1),turns[:,0],
                                    exploration[mi,ki],turns.sum(1)+exploration[mi,ki],strict=True)
                            )
    return result


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('run',type=Path)
    print(export(cli.parse_args().run))
