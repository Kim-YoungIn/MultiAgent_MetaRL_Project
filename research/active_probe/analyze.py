"""Paired return estimates, reward-channel interaction and uncertainty."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REGIMES = ('live', 'rewind')
CHANNELS = ('masked', 'visible')
OPENERS = ('myopic', 'masked', 'visible', 'uniform', 'empirical')
FAMILIES = ('resource', 'sequence', 'reciprocity', 'mixed')
ENDPOINTS = ('live_visible_minus_myopic', 'rewind_reward_interaction')


def outcomes(data):
    terminal = data['terminal']
    opening = data['opening']
    n = len(terminal)
    i = np.arange(n)
    future = np.take_along_axis(data['next_value'], terminal, axis=3)
    result = {}
    for r, regime in enumerate(REGIMES):
        for c, channel in enumerate(CHANNELS):
            for p, policy in enumerate(OPENERS):
                if policy == 'uniform':
                    f = data['first_reward'].mean(1)
                    s = future[:, :, r, c].mean(1)
                    initial = data['first_value'].mean(1)
                    opponent_update = data['opponent_q_update'].mean(1)
                else:
                    if policy == 'empirical':
                        a = data['empirical_opening']
                        t = data['empirical_terminal'][:, c]
                    else:
                        a = opening[:, r, p]
                        t = terminal[i, a, r, c]
                    f = data['first_reward'][i, a]
                    s = data['next_value'][i, a, r, t]
                    initial = data['first_value'][i, a]
                    opponent_update = data['opponent_q_update'][i, a]
                result[regime, channel, policy] = dict(first=f, terminal=s, total=f+s,
                                                    immediate_expected=initial, opponent_update=opponent_update)
    full = result['live','visible','visible']['total'] - result['live','visible','myopic']['total']
    v = result['rewind','visible','visible']['total'] - result['rewind','visible','masked']['total']
    m = result['rewind','masked','visible']['total'] - result['rewind','masked','masked']['total']
    diagnostics = {}
    for r, regime in enumerate(REGIMES):
        a, b, g = opening[:, r, 2], opening[:, r, 1], opening[:, r, 0]
        pay = future[:, :, r, :]
        gain = pay[:, :, 1]-pay[:, :, 0]
        realized = data['first_reward'] + pay[:, :, 1]
        entry = dict(opener_disagreement=(a != b), visible_vs_myopic_disagreement=(a != g),
                     visible_switches_response=(terminal[i,a,r,1] != terminal[i,a,r,0]),
                     uniform_feedback_gain=gain.mean(1), hindsight_gain_spread=gain.max(1)-gain.min(1),
                     hindsight_return_ceiling=realized.max(1), visible_opening_cost=data['first_value'][i,g]-data['first_value'][i,a])
        if regime == 'live':
            no_update = data['next_value_no_update']
            for label, action in [('visible', a), ('myopic', g), ('masked', b)]:
                t = terminal[i, action, r, 1]
                entry['q_update_effect_'+label] = pay[i,action,1]-no_update[i,action,t]
        diagnostics[regime] = entry
    return result, {ENDPOINTS[0]:full, ENDPOINTS[1]:v-m}, diagnostics


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    cfg = json.loads((root/'manifest.json').read_text())['config']
    cells, diagnostics, effects = [], [], []
    for index in range(cfg['blocks']):
        folder = root / f'block_{index:03d}'
        for k in (0, 4, 8):
            with np.load(folder/f'eval_k{k}.npz') as raw:
                data = dict(raw)
            result, contrasts, diag = outcomes(data)
            for fam in ('all',*FAMILIES):
                mask = np.ones(len(data['motif']),bool) if fam=='all' else data['motif']==FAMILIES.index(fam)
                base = dict(block=index, prefix=k, family=fam)
                for (regime,ch,policy), vector in result.items():
                    cells.append(dict(**base, regime=regime, channel=ch, opener=policy,
                                      **{key:float(np.mean(x[mask])) for key,x in vector.items()}))
                for regime, metrics in diag.items():
                    for name, x in metrics.items():
                        diagnostics.append(dict(**base, regime=regime, metric=name, value=float(x[mask].mean())))
                effects.append(dict(**base, **{key:float(x[mask].mean()) for key,x in contrasts.items()}))
    block = np.array([[row[key] for key in ENDPOINTS] for row in effects if row['family']=='all' and row['prefix']==4])
    rng = np.random.default_rng(2817601)
    draws = rng.integers(len(block), size=(20000,len(block)))
    lower, upper = np.quantile(block[draws].mean(axis=1), [.0125,.9875], axis=0)
    summary = dict(blocks=len(block), primary=[dict(name=name, estimate=float(block[:,j].mean()),
                   lower=float(lower[j]), upper=float(upper[j])) for j,name in enumerate(ENDPOINTS)])
    write_csv(root/'cells.csv', cells)
    write_csv(root/'diagnostics.csv', diagnostics)
    write_csv(root/'effects.csv', effects)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(analyze(parser.parse_args().root), indent=2))
