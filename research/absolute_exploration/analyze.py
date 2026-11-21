"""Score absolute utility and the value of the chosen probe's reward signal."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from research.structured_scale.run import FAMILIES

OPENERS = ('return','terminal_only','gain','myopic','donor','uniform')
ENDPOINTS = ('live_return_minus_gain','live_return_minus_myopic','rewind_feedback_return_minus_myopic')


def evaluate(raw):
    achieved = np.take_along_axis(raw['next_value'],raw['terminal_actions'],axis=3)
    components = np.stack([raw['first_reward'],achieved[:,:,0,1],achieved[:,:,0,0],
                           achieved[:,:,1,1],achieved[:,:,1,0]],axis=-1)
    rows = np.arange(len(achieved))[:,None]
    selected = components[rows,raw['openers']]
    all_methods = np.concatenate([selected,components.mean(1,keepdims=True)],axis=1)
    first, visible, masked, rewrite_visible, rewrite_masked = all_methods.transpose(2,0,1)
    return dict(opening_reward=first,terminal_reward=visible,masked_terminal=masked,
                total=first+visible, live_feedback=visible-masked,
                rewind_visible=rewrite_visible,rewind_masked=rewrite_masked,
                rewind_feedback=rewrite_visible-rewrite_masked)


def write_rows(path, rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    count = manifest['config']['blocks']
    effects, summaries, decisions = [], [], []
    for block in range(count):
        with np.load(root/f'block_{block:03d}'/'eval.npz') as raw:
            values = evaluate(raw)
            for family_idx, family in enumerate(('all',*FAMILIES),start=-1):
                selected = raw['motif'] == family_idx if family_idx >= 0 else np.ones(len(raw['motif']),bool)
                means = {metric:pay[selected].mean(0) for metric,pay in values.items()}
                for method_idx,method in enumerate(OPENERS):
                    summaries.append(dict(block=block,family=family,opener=method,
                                          **{key:float(v[method_idx]) for key,v in means.items()}))
                actions = raw['openers'][selected]
                for m, label in enumerate(OPENERS[1:-1],start=1):
                    decisions.append(dict(block=block,family=family,against=label,
                                           differs=float((actions[:,0] != actions[:,m]).mean())))
                if family == 'all':
                    total, rewind = means['total'], means['rewind_feedback']
                    effects.append([total[0]-total[2],total[0]-total[3],rewind[0]-rewind[3]])
    values = np.asarray(effects)
    pick = np.random.default_rng(912853).integers(count,size=(20000,count))
    low, high = np.quantile(values[pick].mean(1),[1/120,119/120],axis=0)
    summary = dict(split=manifest['split'],blocks=count,
                   primary=[dict(name=metric,estimate=float(values[:,i].mean()),lower=float(low[i]),upper=float(high[i]))
                            for i,metric in enumerate(ENDPOINTS)])
    write_rows(root/'cells.csv',summaries)
    write_rows(root/'choices.csv',decisions)
    write_rows(root/'effects.csv',[dict(block=i,**dict(zip(ENDPOINTS,row))) for i,row in enumerate(effects)])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    print(json.dumps(analyze(cli.parse_args().output),indent=2))
