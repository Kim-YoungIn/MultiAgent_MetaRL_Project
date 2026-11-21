"""Run and resume the matched meta-training × opponent-learning study."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter
import numpy as np

from . import data, study
from .analyze import analyze

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONFIG={'dev':dict(seed=181731,blocks=2,games=12),'final':dict(seed=61428759,blocks=24,games=128)}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path,payload):
    path.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')


def sources():
    paths=[HERE/p for p in ('PROTOCOL.md','data.py','study.py','analyze.py','run.py')]
    paths += [ROOT/'research'/p for p in ('structured_scale/run.py','structured_oracle/run.py')]
    paths += [ROOT/'05_toy_models/mamrl'/p for p in ('__init__.py','structured_games.py','environment.py',
               'tensor.py','adaptive_agents.py','common.py','rules.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def block(job):
    output,cfg,index=job
    end=output/f'block_{index:03d}'
    if end.exists():
        hashes=json.loads((end/'record.json').read_text())['sha256']
        if not all(digest(end/p)==v for p,v in hashes.items()):
            raise ValueError(f'corrupt block: {end}')
        return
    start=perf_counter()
    seeds=[int(s) for s in np.random.SeedSequence([cfg['seed'],index]).generate_state(4)]
    with tempfile.TemporaryDirectory(prefix='.pending-',dir=output) as work:
        folder=Path(work)/'block'
        folder.mkdir()
        train=data.sample(seeds[0],cfg['games'])
        w=study.fit(train)
        np.savez_compressed(folder/'train.npz',**train)
        test=data.sample(seeds[1],cfg['games'])
        np.savez_compressed(folder/'test.npz',**test)
        scores=study.evaluate(test,w,np.random.default_rng(seeds[2]))
        live=study.deploy(test,w,np.random.default_rng(seeds[3]))
        np.savez_compressed(folder/'results.npz',w=w,**scores,**live)
        write_json(folder/'record.json',dict(seconds=perf_counter()-start,seeds=seeds,
                   sha256={p.name:digest(p) for p in folder.iterdir()}))
        folder.rename(end)


def execute(root,split,workers):
    cfg=CONFIG[split]
    source=sources()
    record=dict(config=cfg,source=source,numpy=np.__version__,python=platform.python_version(),split=split)
    manifest=root/'manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text())!=record:
            raise ValueError('source or configuration changed; choose a fresh run path')
    else:
        if any(root.glob('block_*')):
            raise ValueError('blocks without manifest')
        for name in source:
            path=root/'source'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,path)
        write_json(manifest,record)
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(block,[(root,cfg,i) for i in range(cfg['blocks'])]))
    seen=set()
    for i in range(cfg['blocks']):
        for name in ('train','test'):
            with np.load(root/f'block_{i:03d}'/f'{name}.npz') as raw:
                fps=set(raw['fingerprint'])
                if len(fps)!=cfg['games'] or seen.intersection(fps):
                    raise ValueError('game overlap')
                seen.update(fps)
    if not (root/'complete.json').exists():
        write_json(root/'complete.json',dict(distinct_games=len(seen),wall_seconds=perf_counter()-start,
                                              workers=workers,platform=platform.platform()))
    print(json.dumps(analyze(root),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--split',choices=CONFIG,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(args.output,args.split,args.workers)


if __name__=='__main__':
    main()
