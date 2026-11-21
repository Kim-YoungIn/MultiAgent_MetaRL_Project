"""Execute or resume independent training/evaluation blocks, using NumPy on CPU."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .evaluation import evaluate, panel
from .game import training


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(f.name, path)


def run_block(args):
    root, seed, index, reps = args
    dest = root / f'block_{index:03d}'
    if dest.exists():
        record = json.loads((dest / 'record.json').read_text())
        for name, expected in record['sha256'].items():
            if digest(dest / name) != expected:
                raise ValueError(f'block checksum mismatch: {dest / name}')
        return record['seconds']
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root, prefix=f'.pending_{index:03d}_') as temp:
        path = Path(temp) / 'block'
        path.mkdir()
        fits = {}
        for count in (32, 128):
            for p in (100, 75, 50):
                for speed in ('slow', 'mix'):
                    arm = f'g{count}_p{p}_{speed}'
                    rng = np.random.default_rng([seed, 1, index, count])
                    row = training(rng, count, p/100, .5 if speed == 'mix' else 0)
                    np.savez_compressed(path / f'train_{arm}.npz', **row)
                    fits[arm] = row['fit']
                    if arm == 'g128_p75_mix':
                        coef = row['pooled']
        tasks = panel(np.random.default_rng([seed, 2, index]), reps)
        for name, local in [('metrics', False), ('shift', True)]:
            result = evaluate(tasks, fits, coef, local)
            np.savez_compressed(path / f'{name}.npz', **result)
        record = dict(index=index, seconds=time.monotonic()-start,
                      sha256={f.name: digest(f) for f in sorted(path.glob('*.npz'))})
        write_json(path / 'record.json', record)
        path.rename(dest)
    return record['seconds']


def run(root, config, workers):
    here = Path(__file__).parent
    sources = {name: digest(here/name) for name in
               ('PROTOCOL.md', 'game.py', 'evaluation.py', 'run.py', 'analyze.py')}
    identity = dict(config=config, sources=sources, numpy=np.__version__, python=platform.python_version())
    manifest = root / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError('existing run has a different configuration, source, or runtime')
    else:
        if any(root.glob('block_*')):
            raise ValueError('missing manifest for existing blocks')
        archive = root / 'source'
        archive.mkdir(exist_ok=True)
        for name in sources:
            shutil.copyfile(here/name, archive/name)
        write_json(manifest, dict(**identity, system=platform.uname()._asdict(),
                   base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                   first_workers=workers))
    start = time.monotonic()
    jobs = [(root, config['seed'], i, config['reps']) for i in range(config['blocks'])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        seconds = list(pool.map(run_block, jobs))
    summary = root / 'completion.json'
    if not summary.exists():
        write_json(summary, dict(blocks=len(jobs), sum_block_seconds=sum(seconds),
                                 completing_invocation_seconds=time.monotonic()-start,
                                 workers=workers, mean_block_seconds=float(np.mean(seconds))))
    print(summary.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', choices=('dev', 'final', 'custom'), required=True)
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--reps', type=int, help='evaluation games per sign/rate stratum')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    presets = {'dev': (2, 2, 721409), 'final': (32, 8, 41927031)}
    if args.split in presets:
        if any(v is not None for v in (args.blocks, args.reps, args.seed)):
            parser.error('use custom split for a new configuration')
        blocks, reps, seed = presets[args.split]
    else:
        blocks, reps, seed = args.blocks, args.reps, args.seed
        if None in (blocks, reps, seed) or seed in (721409, 41927031):
            parser.error('supply blocks, reps and a distinct seed')
    if not (2 <= blocks <= 1000 and reps >= 1 and seed >= 0 and args.workers >= 1):
        parser.error('invalid block count [2,1000], replicates, seed or worker count')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args.output, dict(split=args.split, blocks=blocks, reps=reps, seed=seed), args.workers)


if __name__ == '__main__':
    main()
