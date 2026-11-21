"""Evaluate fixed active-probe policies under genuine and shuffled rewards."""
import json
from pathlib import Path
import numpy as np
import torch
from research.active_probe.data import generate
from research.active_probe import policy

ROOT = Path('runs/probe_stakes_v1')
ROOT.mkdir(parents=True, exist_ok=True)
for block in range(16):
    path = ROOT / f'block_{block:03d}.npz'
    if path.exists():
        continue
    seed = int(np.random.SeedSequence([59183271, block, 1]).generate_state(1)[0])
    rng = np.random.default_rng(np.random.SeedSequence([59183271, block, 2]))
    data = generate(seed, 256, 1)
    checkpoint = torch.load(f'runs/active_probe_v1/block_{block:03d}/weights.pt', map_location='cpu', weights_only=True)
    opening_x = policy.opening_features(data)
    terminal_x = policy.terminal_features(data)
    open_model = policy.Regressor(opening_x.shape[-1], 5)
    terminal_model = policy.Regressor(terminal_x.shape[-1], 2)
    open_model.load_state_dict(checkpoint['opening'])
    terminal_model.load_state_dict(checkpoint['terminal'])
    scores = policy.predict(open_model, opening_x)
    donors = np.empty((32, 256), dtype=int)
    for motif in range(4):
        for other in range(3):
            group = np.flatnonzero((data['motif'] == motif) & (data['first_b'] == other))
            if len(group) < 2:
                raise ValueError('insufficient donor group')
            for j in range(32):
                order = rng.permutation(group)
                donors[j, order] = np.roll(order, int(rng.integers(1, len(order))))
    true_predictions = policy.predict(terminal_model, terminal_x)
    true_action = true_predictions[:,:,1,:,1].argmax(axis=-1)
    rows = np.arange(256)[:, None]
    openings = np.empty((2,256,3), dtype=int)
    rewards = np.empty((2,256,3,4), dtype=float)
    sham_action = np.empty((32,256,3), dtype=int)
    for j in range(32):
        altered = dict(data)
        altered['first_reward'] = data['first_reward'][donors[j]]
        prediction = policy.predict(terminal_model, policy.terminal_features(altered))
        sham_action[j] = prediction[:,:,1,:,1].argmax(axis=-1)
    for ix, h in enumerate((1, 8)):
        a = scores[:,:,0].argmax(1)
        b = (scores[:,:,0]+h*scores[:,:,3]).argmax(1)
        c = (scores[:,:,0]+h*(scores[:,:,3]+scores[:,:,4])).argmax(1)
        openings[ix] = np.stack([a,b,c],axis=1)
        for m, action in enumerate((a,b,c)):
            idx = np.arange(256)
            actual = data['next_value'][idx,action,1,true_action[idx,action]]
            sham = data['next_value'][idx[None,:],action[None,:],1,sham_action[:,idx,action]].mean(0)
            first = data['first_reward'][idx,action]
            rewards[ix,:,m] = np.stack([first,actual,sham,first+h*actual],axis=1)
    np.savez_compressed(path, **data, donors=donors, openings=openings, rewards=rewards, true_action=true_action, sham_action=sham_action)
effects=[]
for block in range(16):
    with np.load(ROOT/f'block_{block:03d}.npz') as d:
        r=d['rewards'][1]
        effects.append(((r[:,2,1]-r[:,2,2])-(r[:,1,1]-r[:,1,2])).mean())
x=np.array(effects)
rng=np.random.default_rng(488021)
idx=rng.integers(16,size=(20000,16))
result={'estimate':float(x.mean()),'interval_95':np.quantile(x[idx].mean(1),[.025,.975]).tolist(),'block_effects':x.tolist()}
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
