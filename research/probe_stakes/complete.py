"""Complete uniform-opener and historical-panel checks from retained branches."""
import json
from pathlib import Path
import numpy as np

root=Path('runs/probe_stakes_v1')
previous=set()
for filename in Path('runs/active_probe_v1').glob('block_*/*.npz'):
    with np.load(filename) as data:
        previous.update(data['game_fingerprint'].tolist())

def collect(value):
    if isinstance(value, dict):
        if 'fingerprint' in value:
            previous.add(value['fingerprint'])
        if 'fingerprints' in value:
            previous.update(value['fingerprints'])
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)

for run in ('structured_transfer_v1', 'structured_scale_v1'):
    for filename in (Path('runs') / run).glob('block_*/*.json'):
        if filename.name in ('train.json', 'test.json'):
            collect(json.loads(filename.read_text()))
collect(json.loads(Path('runs/structured_oracle_v1/pairs.json').read_text()))
for filename in Path('runs/opponent_adaptation_v1').glob('block_*/[tt]*.npz'):
    with np.load(filename) as data:
        previous.update(data['fingerprint'].tolist())
blocks=[]
new=set()
for filename in sorted(root.glob('block_*.npz')):
    with np.load(filename) as data:
        fp=set(data['game_fingerprint'].tolist())
        assert len(fp)==256 and not new.intersection(fp)
        new.update(fp)
        i=np.arange(256)[:,None]
        a=np.arange(3)[None,:]
        first=data['first_reward'].mean(axis=1)
        actual=data['next_value'][i,a,1,data['true_action']].mean(axis=1)
        sham=data['next_value'][i[None,:,:],a[None,:,:],1,data['sham_action']].mean(axis=(0,2))
        blocks.append([first.mean(),actual.mean(),sham.mean(),(first+actual).mean(),(first+8*actual).mean()])
assert len(blocks)==16 and not new.intersection(previous)
result={'blocks':len(blocks),'games':len(new),'previous_exact_fingerprints':len(previous),
        'overlap':len(new.intersection(previous)),
        'uniform_mean':dict(zip(('first','genuine_terminal','shuffled_terminal','return_h1','return_h8'),np.mean(blocks,axis=0).tolist())),
        'block_values':blocks,
        'checked_previous':['active_probe_v1','structured_transfer_v1','structured_scale_v1','structured_oracle_v1','opponent_adaptation_v1']}
(root/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['uniform_mean'],indent=2))
print('Prior fingerprints',len(previous),'overlap',result['overlap'])
