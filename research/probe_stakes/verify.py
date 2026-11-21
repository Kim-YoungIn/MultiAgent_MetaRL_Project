"""Verify donor assignments and recover outcomes from saved action arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('runs/probe_stakes_v1')
effects=[]
hashes={}
seen=set()
for block in range(16):
    path=root/f'block_{block:03d}.npz'
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    original=Path(f'runs/active_probe_v1/block_{block:03d}')
    record=json.loads((original/'record.json').read_text())
    assert hashlib.sha256((original/'weights.pt').read_bytes()).hexdigest()==record['sha256']['weights.pt']
    with np.load(path) as d:
        fingerprints=set(d['game_fingerprint'])
        assert len(fingerprints)==256 and not fingerprints.intersection(seen)
        seen.update(fingerprints)
        for donors in d['donors']:
            assert len(set(donors))==256
            assert np.all(donors!=np.arange(256))
            assert np.array_equal(d['motif'][donors],d['motif'])
            assert np.array_equal(d['first_b'][donors],d['first_b'])
        for h in range(2):
            for method in range(3):
                idx=np.arange(256)
                act=d['openings'][h,:,method]
                true=d['next_value'][idx,act,1,d['true_action'][idx,act]]
                sham=d['next_value'][idx[None,:],act[None,:],1,d['sham_action'][:,idx,act]].mean(0)
                np.testing.assert_allclose(true,d['rewards'][h,:,method,1])
                np.testing.assert_allclose(sham,d['rewards'][h,:,method,2])
        r=d['rewards'][1]
        effects.append(((r[:,2,1]-r[:,2,2])-(r[:,1,1]-r[:,1,2])).mean())
x=np.array(effects)
idx=np.random.default_rng(488021).integers(16,size=(20000,16))
summary=json.loads((root/'summary.json').read_text())
np.testing.assert_allclose([summary['estimate'],*summary['interval_95']],[x.mean(),*np.quantile(x[idx].mean(1),[.025,.975])])
(root/'verification.json').write_text(json.dumps({'blocks':16,'games':len(seen),'checkpoint_hashes_verified':True,'donors_verified':True,'payoffs_and_interval_verified':True,'raw_sha256':hashes},indent=2)+'\n')
print('Verified 16 blocks, 4096 unique games, donor permutations and paired estimates.')
