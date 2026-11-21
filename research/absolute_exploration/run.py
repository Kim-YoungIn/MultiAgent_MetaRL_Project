"""Train and compare matched opening policies without altering historic studies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import torch

from research.active_probe import policy
from research.active_probe.data import generate
from research.active_probe.run import ROOT, digest, sources, write_json
from .analyze import analyze

METHODS = ('return', 'terminal_only', 'gain', 'myopic')
CONFIGS = {
    'dev': dict(seed=8420129, blocks=2, terminal_games=32, opening_games=48, train_resets=2, test_games=24),
    'final': dict(seed=29380741, blocks=24, terminal_games=512, opening_games=768, train_resets=4, test_games=256),
}


def targets(data, terminal_actions):
    values = policy.achieved(data, terminal_actions)
    visible = values[:, :, 0, 1]
    masked = values[:, :, 0, 0]
    immediate = data['first_value']
    return np.stack([immediate + visible, visible, visible - masked, immediate], axis=-1)


def history_donors(data):
    """Use the next distinct game in each motif, aligned by reset slot."""
    samples = len(data['motif'])
    ids, totals = np.unique(data['game_seed'], return_counts=True)
    resets = totals[0]
    if not np.all(totals == resets):
        raise ValueError('donor assignment needs an equal number of resets for each game')
    groups = np.arange(samples).reshape(len(ids), resets)
    if not np.all(np.ptp(data['game_seed'][groups], axis=1) == 0):
        raise ValueError('resets of a game must be contiguous')
    motif = data['motif'][groups[:, 0]]
    donor = np.empty(samples, dtype=int)
    for f in range(4):
        rows = groups[motif == f]
        if len(rows) < 2:
            raise ValueError('need at least two games per motif')
        donor[rows.ravel()] = np.roll(rows, -1, axis=0).ravel()
    return donor


def fit_block(args):
    root, cfg, index = args
    final = root / f'block_{index:03d}'
    if final.exists():
        verify_block(final)
        return
    start = time.perf_counter()
    seeds = [int(x) for x in np.random.SeedSequence([cfg['seed'], index]).generate_state(5)]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(dir=root, prefix='.pending-') as temp:
        stage = Path(temp) / 'block'
        stage.mkdir()
        terminal_data = generate(seeds[0], cfg['terminal_games'], cfg['train_resets'])
        terminal, terminal_loss = policy.fit_terminal(terminal_data, seeds[3])
        np.savez_compressed(stage/'terminal_train.npz', **terminal_data)
        opener_data = generate(seeds[1], cfg['opening_games'], cfg['train_resets'])
        actions = policy.terminal_actions(terminal, opener_data)
        labels = targets(opener_data, actions)
        features = policy.opening_features(opener_data)
        weights, losses = {'terminal': terminal.state_dict()}, {'terminal': terminal_loss}
        test = generate(seeds[2], cfg['test_games'], 2)
        test_features = policy.opening_features(test)
        predicted = []
        for j, method in enumerate(METHODS):
            model, losses[method] = policy.fit(features, labels[..., j], seeds[4], 600, 256)
            weights[method] = model.state_dict()
            predicted.append(policy.predict(model, test_features)[..., 0])
        scores = np.stack(predicted, axis=-1)
        donor = history_donors(test)
        openers = scores.argmax(axis=1)
        openers = np.column_stack([openers, openers[donor, 0]])
        np.savez_compressed(stage/'opening_train.npz', **opener_data, targets=labels, terminal_actions=actions)
        np.savez_compressed(stage/'eval.npz', **test, scores=scores, openers=openers, donor=donor,
                            terminal_actions=policy.terminal_actions(terminal, test))
        torch.save(weights, stage/'weights.pt')
        write_json(stage/'record.json', dict(index=index, seeds=seeds, losses=losses,
                   seconds=time.perf_counter()-start, sha256={p.name:digest(p) for p in sorted(stage.iterdir())}))
        stage.rename(final)


def verify_block(folder):
    record = json.loads((folder/'record.json').read_text())
    if set(record['sha256']) != {'terminal_train.npz','opening_train.npz','eval.npz','weights.pt'}:
        raise ValueError(f'incomplete block manifest in {folder}')
    for filename, expected in record['sha256'].items():
        if digest(folder/filename) != expected:
            raise ValueError(f'checksum mismatch at {folder/filename}')


def check_splits(root, cfg):
    seen = set()
    for index in range(cfg['blocks']):
        folder = root / f'block_{index:03d}'
        for name, resets, games in [
            ('terminal_train', cfg['train_resets'], cfg['terminal_games']),
            ('opening_train', cfg['train_resets'], cfg['opening_games']),
            ('eval', 2, cfg['test_games']),
        ]:
            with np.load(folder/f'{name}.npz') as raw:
                ids, counts = np.unique(raw['game_fingerprint'], return_counts=True)
            if len(ids) != games or not np.all(counts == resets) or seen.intersection(ids):
                raise ValueError(f'bad game count or split overlap: {folder/name}')
            seen.update(ids)
    return len(seen)


def execute(root, split, workers):
    cfg = CONFIGS[split]
    source = sources()
    base = Path(__file__).resolve().parent
    source.update({str(path.relative_to(ROOT)):digest(path)
                   for path in (base/'PROTOCOL.md', base/'run.py', base/'analyze.py')})
    identity = dict(split=split,config=cfg,source=source,numpy=np.__version__,
                    torch=str(torch.__version__),python=platform.python_version())
    manifest = root/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('source or configuration differs; use a fresh path')
    else:
        if any(root.glob('block_*')):
            raise ValueError('run blocks without a manifest')
        for name in source:
            target = root/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            staged = Path(temp)/'manifest.json'
            write_json(staged,identity)
            staged.replace(manifest)
    start = time.perf_counter()
    context = multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        list(pool.map(fit_block, [(root,cfg,i) for i in range(cfg['blocks'])]))
    total = check_splits(root,cfg)
    if not (root/'complete.json').exists():
        write_json(root/'complete.json',dict(distinct_games=total, workers=workers,
                   wall_seconds=time.perf_counter()-start, platform=platform.platform()))
    print(json.dumps(analyze(root),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--split',choices=CONFIGS,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(args.output,args.split,args.workers)


if __name__ == '__main__':
    main()
