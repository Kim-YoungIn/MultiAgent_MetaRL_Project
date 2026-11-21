"""Run a fixed split with independent CPU blocks and versioned checkpoints."""
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
import time

import numpy as np
import torch

from . import policy
from .analyze import analyze
from .data import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIGS = {
    'dev': dict(seed=591302, blocks=2, train_games=40, train_resets=2, test_games=32, shift_games=8),
    'final': dict(seed=67021843, blocks=16, train_games=512, train_resets=4, test_games=256, shift_games=64),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [HERE/f for f in ('PROTOCOL.md','run.py','data.py','policy.py','analyze.py')]
    paths += [ROOT/'research'/p for p in ('structured_oracle/run.py','structured_scale/run.py')]
    paths += [p for p in (ROOT/'05_toy_models/mamrl').glob('*.py') if p.name in
              ('structured_games.py','adaptive_agents.py','environment.py','tensor.py','rules.py',
               'common.py','policy.py','rollout.py','__init__.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2)+'\n')


def assemble(args):
    root, cfg, index = args
    folder = root/f'block_{index:03d}'
    if folder.exists():
        rec = json.loads((folder/'record.json').read_text())
        for filename, h in rec['sha256'].items():
            if digest(folder/filename) != h:
                raise ValueError(f'checksum mismatch: {folder/filename}')
        return
    start = time.perf_counter()
    seeds = [int(s) for s in np.random.SeedSequence([cfg['seed'],index]).generate_state(7)]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(dir=root, prefix='.pending-') as pending:
        stage = Path(pending)/'block'
        stage.mkdir()
        data = generate(seeds[0], cfg['train_games'], cfg['train_resets'])
        terminal, term_loss = policy.fit_terminal(data, seeds[2])
        np.savez_compressed(stage/'terminal_train.npz', **data)
        data = generate(seeds[1], cfg['train_games'], cfg['train_resets'])
        opening, open_loss = policy.fit_opening(data, terminal, seeds[3])
        np.savez_compressed(stage/'opening_train.npz', **data)
        torch.save(dict(terminal=terminal.state_dict(),opening=opening.state_dict()), stage/'weights.pt')
        for j,k in enumerate((4,0,8)):
            games = cfg['test_games'] if k==4 else cfg['shift_games']
            test = generate(seeds[4+j], games, 2, prefix=k)
            scores = policy.predict(opening, policy.opening_features(test))
            ac, follow = policy.empirical_actions(test)
            np.savez_compressed(stage/f'eval_k{k}.npz', **test, opening_scores=scores,
                                opening=policy.opening_actions(scores), terminal=policy.terminal_actions(terminal,test),
                                empirical_opening=ac, empirical_terminal=follow)
        write_json(stage/'record.json',dict(block=index, seconds=time.perf_counter()-start, seeds=seeds,
                   terminal_loss=term_loss, opening_loss=open_loss,
                   sha256={p.name:digest(p) for p in sorted(stage.iterdir())}))
        stage.rename(folder)


def check_splits(root, cfg):
    seen = set()
    for index in range(cfg['blocks']):
        for f in (root/f'block_{index:03d}').glob('*.npz'):
            with np.load(f) as data:
                fps, counts = np.unique(data['game_fingerprint'],return_counts=True)
                expected = cfg['train_resets'] if 'train' in f.name else 2
                if not (counts == expected).all() or seen.intersection(fps):
                    raise ValueError(f'duplicate game: {f}')
                seen.update(fps)
    return len(seen)


def execute(root, split, workers):
    cfg = CONFIGS[split]
    identity = dict(split=split,config=cfg,source=sources(),numpy=np.__version__,
                    torch=str(torch.__version__),python=platform.python_version())
    path = root/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('run configuration/source differs; use a fresh path')
    else:
        if any(root.glob('block_*')):
            raise ValueError('existing blocks without a manifest')
        for name in identity['source']:
            dest = root/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        write_json(path,identity)
    start = time.perf_counter()
    ctx = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as pool:
        list(pool.map(assemble,[(root,cfg,i) for i in range(cfg['blocks'])]))
    games = check_splits(root,cfg)
    if not (root/'complete.json').exists():
        write_json(root/'complete.json',dict(distinct_games=games,wall_seconds=time.perf_counter()-start,
                   workers=workers,platform=platform.platform()))
    print(json.dumps(analyze(root), indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--split',choices=CONFIGS,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(args.output,args.split,args.workers)


if __name__=='__main__':
    main()
