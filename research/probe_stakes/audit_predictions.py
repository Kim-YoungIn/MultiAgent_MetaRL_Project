"""Reconstruct decisions from frozen checkpoints and summarize both weights."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from research.active_probe.policy import opening_features, terminal_features
from research.active_probe.verify import network


def main():
    root = Path('runs/probe_stakes_v1')
    upstream = Path('runs/active_probe_v1')
    checked = []
    means = []
    disagreement = []
    for block in range(16):
        path = root/f'block_{block:03d}.npz'
        original = upstream/f'block_{block:03d}'
        weights = torch.load(original/'weights.pt', map_location='cpu', weights_only=True)
        record = json.loads((original/'record.json').read_text())
        if hashlib.sha256((original/'weights.pt').read_bytes()).hexdigest() != record['sha256']['weights.pt']:
            raise ValueError(f'checkpoint hash mismatch in block {block}')
        with np.load(path) as raw:
            data = dict(raw)
        scores = network(weights['opening'],opening_features(data))
        x = terminal_features(data)
        genuine = network(weights['terminal'],x[:,:,1,:,:])[...,1]
        agree(genuine,data['true_action'],f'genuine block {block}')
        for row, donor in enumerate(data['donors']):
            altered = {**data,'first_reward':data['first_reward'][donor]}
            shuffled_x = terminal_features(altered)
            np.testing.assert_array_equal(x[:,:,0],shuffled_x[:,:,0])
            changed = np.any(x[:,:,1] != shuffled_x[:,:,1],axis=(0,1,2))
            if np.count_nonzero(changed) > 1:
                raise ValueError('sham changed more than the reward feature')
            predicted = network(weights['terminal'],shuffled_x[:,:,1,:,:])[...,1]
            agree(predicted,data['sham_action'][row],f'sham block {block} draw {row}')
        for j,h in enumerate((1,8)):
            for m,pred in enumerate((scores[...,0],scores[...,0]+h*scores[...,3],
                                      scores[...,0]+h*(scores[...,3]+scores[...,4]))):
                agree(pred,data['openings'][j,:,m],f'opening block {block} weight {h} method {m}')
        checked.append(dict(block=block, game_count=len(scores), donors=len(data['donors'])))
        means.append(data['rewards'].mean(1))
        disagreement.append((data['openings'][:,:,1] != data['openings'][:,:,2]).mean(1))
    with (root/'outcomes.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['weight','opener','first','genuine_terminal','sham_terminal','genuine_utility'])
        for h, row in zip((1,8),np.mean(means,axis=0),strict=True):
            for name, result in zip(('myopic','masked','visible'),row,strict=True):
                writer.writerow([h,name,*result])
    report = dict(blocks=checked, decision_score_tolerance=1e-6,
                  specialized_opener_disagreement=np.mean(disagreement,axis=0).tolist())
    (root/'prediction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Checked all frozen-checkpoint choices for {sum(row["game_count"] for row in checked)} games.')


def agree(scores, actions, context):
    chosen=np.take_along_axis(scores,actions[...,None],axis=-1)[...,0]
    np.testing.assert_allclose(chosen,scores.max(-1),atol=1e-6,rtol=0,err_msg=context)


if __name__=='__main__':
    main()
