"""Independent raw D-031 check of zero-shot, retained and erased performance."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from experiment.artifacts import verify_evaluation


def raw(path, method):
    verify_evaluation(path,method)
    with gzip.open(path/f'{method}.trials.jsonl.gz','rt') as stream:
        trials = map(json.loads, stream)
        return {(r['rule'],r['profile'],r['rep'],r['seed']): r for r in trials
                if r['split']=='fresh_rules' and r['profile'] in ('steady','sarsa')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    aucs, at_24 = [], []
    for c in range(1,6):
        for r in range(1,6):
            path=args.study/f'cells/primary/c{c}/j{r}/n3'
            paired=tuple(raw(path,method) for method in ('M','P'))
            assert len(paired[0]) == 1024 and paired[0].keys() == paired[1].keys()
            records=[]
            for group in paired:
                rounds=np.array([[p['return'] for p in trial['curve']] for trial in group.values()])
                gains=np.array([[p['G'] for p in trial['curve']] for trial in group.values()])
                auc=(rounds-rounds[:,:1])@np.array([1.5,3,4.5,9,6])/24
                np.testing.assert_allclose(auc,[v['auc'] for v in group.values()],atol=1e-12)
                actual=rounds[:,-1]
                retained=gains[:,-1]
                fixed=np.array([next(p['game'] for p in v['fixed_budget'] if p['k']==24) for v in group.values()])
                records.append(dict(zero=rounds[:,0],retained_gain=retained,assessment=actual,
                                    erased_assessment=actual-retained, fixed_budget_mean=fixed))
            np.testing.assert_array_equal(records[0]['zero'],records[1]['zero'])
            aucs.append(np.mean([paired[0][key]['auc']-paired[1][key]['auc'] for key in paired[0]]))
            at_24.append([{k:float(v.mean()) for k,v in arm.items()} for arm in records])
    original=json.loads((args.study/'report.json').read_text())['primary']
    np.testing.assert_allclose(np.mean(aucs),original['mean'],atol=1e-12)
    result=dict(fits_paired=len(aucs),primary_original=original,
        means={name:{metric:float(np.mean([r[i][metric] for r in at_24])) for metric in at_24[0][i]}
               for i,name in enumerate(('retained_gain_objective','immediate_reward_objective'))})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['means'],indent=2))


if __name__=='__main__':
    main()
