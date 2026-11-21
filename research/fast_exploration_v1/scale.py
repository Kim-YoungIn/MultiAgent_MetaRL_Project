"""Run the 24-block Fast Exploration protocol with batched simulation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from experiment.artifacts import atomic_path
from research.structured_scale.run import FAMILIES
from . import batch, study
from .figure import render
from .report import make_report

MASTER_SEEDS = (7311903, 8291309, 9416819)
CONFIGS = {
    'dev': dict(blocks=2, blocks_per_master=2, test_games=16, prior_games=32,
                teacher_updates=8, batch=6, prior_steps=25, test_resets=2,
                ks=[0,4,8,16,24], h=32),
    'final': dict(blocks=24, blocks_per_master=8, test_games=128, prior_games=128,
                  teacher_updates=400, batch=32, prior_steps=160, test_resets=2,
                  ks=[0,4,8,16,24], h=32),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_weights(path,value):
    with atomic_path(path) as tmp:
        # A file handle keeps the PyTorch archive's internal root name stable.
        with tmp.open('wb') as stream:
            torch.save(value,stream)


def source_files():
    project = Path(__file__).resolve().parents[2]
    names = set(study.SOURCE_PATHS)
    names.update(f'research/fast_exploration_v1/{name}' for name in
                 ('batch.py','simulator.py','report.py','scale.py','SCALE_PROTOCOL.md'))
    names.update(('05_toy_models/mamrl/adaptive_agents.py','experiment/artifacts.py'))
    return project, {name:digest(project/name) for name in sorted(names)}


def completed(output,index):
    prefix=output/f'block_{index:03d}'
    record=prefix.with_suffix('.json')
    if not record.exists():
        return False
    payload=json.loads(record.read_text())
    if payload['block']!=index:
        raise ValueError(f'block number mismatch at {record}')
    for ext,key in (('.pt','weights_sha256'),('.npz','raw_sha256')):
        if digest(prefix.with_suffix(ext))!=payload[key]:
            raise ValueError(f'invalid {ext} checksum at {record}')
    return True


def fit_teachers(output,index,prior,seed,cfg,device,sampling):
    path=output/f'.block_{index:03d}_progress.pt'
    previous=torch.load(path,map_location=device,weights_only=True) if path.exists() else None
    metrics={name:[] for name in batch.METHODS[:2]} if previous is None else previous['metrics']

    def save(update,actors,optimizers,rng,rewards):
        for name,value in rewards.items():
            metrics[name].append(value)
        if update%25==0 or update==cfg['teacher_updates']:
            save_weights(path,dict(models={m:v.state_dict() for m,v in actors.items()},
                         optimizers={m:v.state_dict() for m,v in optimizers.items()},
                         rng=rng.bit_generator.state,update=update,metrics=metrics))

    models=batch.train_group(prior,seed,cfg,device=device,sampling=sampling,methods=batch.METHODS[:2],
                             history=save,resume=previous)
    return models,metrics


def fit_job(job):
    output,cfg,master,seed_block,index,device,sampling=job
    if completed(output,index):
        return index
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    seeds=np.random.SeedSequence([master,seed_block]).generate_state(3)
    train_seed,prior_seed,test_seed=map(int,seeds)
    prior_data=study.prior_samples(prior_seed,cfg['prior_games'])
    meta=study.fit_return_prior(prior_data,steps=cfg['prior_steps'],seed=train_seed,device=device)
    _, pooled=study.pooled_fit(prior_data)
    t1=time.perf_counter()
    models,traces=fit_teachers(output,index,meta,train_seed,cfg,device,sampling)
    t2=time.perf_counter()
    rng=np.random.default_rng(test_seed)
    game_seeds=rng.integers(2**61,2**62,size=cfg['test_games'])
    resets=cfg['test_resets']
    reset_seeds=rng.integers(2**61,size=len(game_seeds)*resets)
    families=[FAMILIES[i%4] for i in range(len(game_seeds)) for _ in range(resets)]
    episodes=batch.prepare(reset_seeds,families,max(cfg['ks']),cfg['h'],rule_seeds=np.repeat(game_seeds,resets))
    fingerprints=[r.fingerprint() for r in episodes.rules[::resets]]
    if len(set(fingerprints))!=cfg['test_games']:
        raise ValueError('duplicate held-out games')
    if not set(fingerprints).isdisjoint(prior_data['fingerprint']):
        raise ValueError('prior/test overlap')
    scores,raw=batch.evaluate({'reward_auc':meta,'pooled_auc':pooled,'zero_auc':np.zeros_like(meta)},
                               {k:v.cpu().eval() for k,v in models.items()},episodes,cfg)
    t3=time.perf_counter()
    prefix=output/f'block_{index:03d}'
    weights=prefix.with_suffix('.pt')
    archive=prefix.with_suffix('.npz')
    with atomic_path(archive) as path, path.open('wb') as stream:
        np.savez_compressed(stream, **raw, game_seeds=game_seeds, reset_seeds=reset_seeds,
                            families=families, test_fingerprints=fingerprints,
                            prior_fingerprints=prior_data['fingerprint'], ks=cfg['ks'],
                            methods=batch.METHODS, **{f'{m}_training_return':v for m,v in traces.items()})
    save_weights(weights,{name:actor.state_dict() for name,actor in models.items()})
    record=dict(block=index,master_seed=master,seed_block=seed_block,seeds=list(map(int,seeds)),
                trained_prior=meta.tolist(),pooled_prior=pooled.tolist(),**scores,
                weights_sha256=digest(weights),raw_sha256=digest(archive),
                seconds=dict(prior=t1-t0,teacher=t2-t1,evaluation=t3-t2,total=time.perf_counter()-t0))
    save_json(prefix.with_suffix('.json'),record)
    (output/f'.block_{index:03d}_progress.pt').unlink(missing_ok=True)
    return index


def execute(output,split,device,sampling,workers,only=None):
    cfg=CONFIGS[split]
    masters=MASTER_SEEDS if split=='final' else (8925115,)
    chosen=study.resolve_device(device)
    actor_device=study.resolve_device(sampling)
    project,sources=source_files()
    identity=dict(protocol='fast-exploration-scale-v1',split=split,config=cfg,master_seeds=masters,
                  device=str(chosen),sampling=str(actor_device),python=platform.python_version(),
                  torch=str(torch.__version__),numpy=np.__version__,cuda_runtime=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(chosen) if chosen.type=='cuda' else None,source=sources)
    identity['master_seeds']=list(identity['master_seeds'])
    manifest=output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:
            raise ValueError('sealed configuration differs; use a fresh output path')
    else:
        if any(output.glob('block_*')):
            raise ValueError('block artifacts without a manifest')
        for name,checksum in sources.items():
            dest=output/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            content=(project/name).read_bytes()
            if hashlib.sha256(content).hexdigest()!=checksum:
                raise RuntimeError('source changed during preparation')
            dest.write_bytes(content)
        save_json(manifest,identity)
    all_pairs=[(s,b) for s in masters for b in range(cfg['blocks_per_master'])]
    jobs=[(output,cfg,s,b,i,str(chosen),str(actor_device)) for i,(s,b) in enumerate(all_pairs)
          if (only is None or i==only) and not completed(output,i)]
    start=time.perf_counter()
    if workers==1:
        for job in jobs:
            index=fit_job(job)
            print(f'block {index:03d} saved',flush=True)
    elif jobs:
        # A fresh interpreter is required before CUDA initialization. CPU
        # workers fork without inheriting a CUDA context and share code pages.
        ctx=multiprocessing.get_context('spawn' if 'cuda' in (str(chosen),str(actor_device)) else 'fork')
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as pool:
            for result in as_completed([pool.submit(fit_job,job) for job in jobs]):
                print(f'block {result.result():03d} saved',flush=True)
    if only is None:
        seen=set()
        for i in range(cfg['blocks']):
            with np.load(output/f'block_{i:03d}.npz') as raw:
                fps=set(raw['test_fingerprints'])
            if seen.intersection(fps):
                raise ValueError(f'held-out fingerprint reused at block {i}')
            seen.update(fps)
        report=make_report(output)
        render(output)
        save_json(output/'complete.json',dict(blocks=report['blocks'],test_games=len(seen),
                    completing_invocation_seconds=time.perf_counter()-start,workers=workers,
                    platform=platform.platform()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=CONFIGS,required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--sampling',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--only-block',type=int)
    args=parser.parse_args()
    if args.workers<1 or (args.only_block is not None and not 0<=args.only_block<CONFIGS[args.split]['blocks']):
        parser.error('invalid worker count or block index')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args.output,args.split,args.device,args.sampling,args.workers,args.only_block)


if __name__=='__main__':
    main()
