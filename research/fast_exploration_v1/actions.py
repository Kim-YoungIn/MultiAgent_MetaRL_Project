"""Replay sealed CPU evaluations and record the executed joint actions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from experiment.artifacts import atomic_path
from . import batch, study
from .scale import completed, digest, save_json, source_files

ROLE_NAMES = ('invest', 'harvest', 'signal')
PRIORS = ('reward_auc', 'pooled_auc', 'zero_auc')
REGIMES = ('reward_auc', 'controlled_auc', 'pooled_auc', 'zero_auc')


class RecordedEpisode:
    """Observe actions at the simulator boundary without touching its state."""
    def __init__(self, env):
        self.env = env
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, focal, opponent, gate_uniforms):
        rewards = self.env.step(focal, opponent, gate_uniforms)
        self.actions.append(np.column_stack((focal, opponent)).astype(np.int8))
        return rewards

    def trajectory(self):
        return np.stack(self.actions, axis=1)


class RecordedEpisodes:
    def __init__(self, episodes):
        self.episodes = episodes
        self.probe = None
        self.assessments = []

    def __getattr__(self, name):
        return getattr(self.episodes, name)

    def episode(self, seeds, horizon):
        trace = RecordedEpisode(self.episodes.episode(seeds, horizon))
        if seeds is self.episodes.explore_seed:
            self.probe = trace
        elif seeds is self.episodes.assess_seed:
            self.assessments.append(trace)
        else:
            raise ValueError('unexpected rollout phase')
        return trace


def replay_block(job):
    run, output, cfg, index = job
    stem = f'block_{index:03d}'
    if not completed(run, index):
        raise ValueError(f'no sealed result for {stem}')
    record = json.loads((run / f'{stem}.json').read_text())
    checkpoint = torch.load(run / f'{stem}.pt', map_location='cpu', weights_only=True)
    actors = {}
    for method in batch.METHODS[:2]:
        actor = study.BeliefActor()
        actor.load_state_dict(checkpoint[method])
        actors[method] = actor.eval()
    priors = (record['trained_prior'], record['pooled_prior'], np.zeros_like(record['trained_prior']))
    ks, h = cfg['ks'], cfg['h']
    fields = ('game_seeds', 'reset_seeds', 'families', 'test_fingerprints', 'methods', 'ks')
    with np.load(run / f'{stem}.npz') as raw:
        meta = {name: raw[name] for name in fields}
        np.testing.assert_array_equal(meta['methods'], batch.METHODS)
        np.testing.assert_array_equal(meta['ks'], ks)
        seeds = np.repeat(meta['game_seeds'], cfg['test_resets'])
        episodes = batch.prepare(meta['reset_seeds'], meta['families'], max(ks), h, rule_seeds=seeds)
        np.testing.assert_array_equal(
            [r.fingerprint() for r in episodes.rules[::cfg['test_resets']]], meta['test_fingerprints']
        )
        # The inverse permutation maps an executed raw action to its game-specific role.
        roles = np.array([r.roles for r in episodes.rules], dtype=np.int8)
        np.testing.assert_array_equal(np.sort(roles, axis=1), np.tile(np.arange(3), (len(roles), 1)))
        initial_q = batch.warm_up(episodes)
        shape = (len(batch.METHODS), len(ks), len(seeds))
        actions = {}
        for regime in REGIMES:
            actions[f'{regime}_assessment'] = np.empty((*shape, h, 2), dtype=np.int8)
        for regime, prior in zip(PRIORS, priors, strict=True):
            controlled = regime == 'reward_auc'
            exploration = actions[f'{regime}_exploration'] = np.full((*shape, max(ks), 2), -1, dtype=np.int8)
            assessment = actions[f'{regime}_assessment']
            expected_exploration = raw[f'{regime}_exploration']
            expected_assessment = raw[f'{regime}_assessment']
            if controlled:
                expected_controlled = raw['controlled_auc_assessment']
                np.testing.assert_array_equal(expected_exploration, raw['controlled_auc_exploration'])
                frozen = actions['controlled_auc_assessment']

            for ki, k in enumerate(ks):
                for mi, method in enumerate(batch.METHODS):
                    trace = RecordedEpisodes(episodes)
                    rewards, probe, _, focal_actions = batch.collect(
                        trace, initial_q, prior, k, h, method=method,
                        teacher=actors.get(method), controlled=controlled,
                    )
                    np.testing.assert_array_equal(probe.sum(axis=1), expected_exploration[mi, ki])
                    np.testing.assert_array_equal(rewards['live'], expected_assessment[mi, ki])
                    if k:
                        joint = trace.probe.trajectory()
                        np.testing.assert_array_equal(joint[..., 0], focal_actions)
                        exploration[mi, ki, :, :k] = joint
                    # collect creates the live assessment reset before the controlled fork.
                    assessment[mi, ki] = trace.assessments[0].trajectory()
                    if controlled:
                        np.testing.assert_array_equal(rewards['controlled'], expected_controlled[mi, ki])
                        frozen[mi, ki] = trace.assessments[1].trajectory()

    dest = output / f'{stem}.npz'
    with atomic_path(dest) as temp, temp.open('wb') as stream:
        np.savez_compressed(stream, **meta, **actions, action_to_role=roles.argsort(axis=1),
                            role_names=ROLE_NAMES, master_seed=record['master_seed'],
                            seed_block=record['seed_block'])
    return stem, dict(sha256=digest(dest), input_raw_sha256=record['raw_sha256'],
                      checkpoint_sha256=record['weights_sha256'])


def replay(run, output, workers):
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['config']
    if (manifest['protocol'], manifest['sampling']) != ('fast-exploration-scale-v1', 'cpu'):
        raise ValueError('expected a scaled v1 run with CPU sampling')
    runtime = platform.python_version(), str(torch.__version__), np.__version__
    if runtime != tuple(manifest[key] for key in ('python', 'torch', 'numpy')):
        raise ValueError('replay requires the experiment runtime versions')
    _, sources = source_files()
    if sources != manifest['source']:
        raise ValueError('replay requires the original sealed sources')
    if json.loads((run / 'complete.json').read_text())['blocks'] != cfg['blocks']:
        raise ValueError('run is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jobs = [(run, output, cfg, index) for index in range(cfg['blocks'])]
    if workers == 1:
        receipts = dict(map(replay_block, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            receipts = dict(pool.map(replay_block, jobs))
    save_json(output / 'complete.json', dict(
        kind='replayed_joint_actions', blocks=cfg['blocks'], source_manifest_sha256=digest(manifest_path),
        replay_code_sha256=digest(Path(__file__)), sampling='cpu', rewards_exact_match=True,
        axes=['method', 'budget', 'game_reset', 'turn', 'player'], players=['focal', 'opponent'],
        padding=-1, role_names=ROLE_NAMES, controlled_exploration='reward_auc_exploration',
        test_games=cfg['blocks'] * cfg['test_games'], test_resets=cfg['test_resets'], receipts=receipts,
    ))


def write_game_csv(path, game, stream):
    with np.load(path) as data:
        games = len(data['game_seeds'])
        if not 0 <= game < games:
            raise ValueError(f'game index must be in [0, {games})')
        repeats = len(data['reset_seeds']) // games
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['block', 'master_seed', 'game_index', 'game_seed', 'reset_index', 'reset_seed',
                         'family', 'regime', 'method', 'k', 'phase', 'turn', 'focal_controller',
                         'focal_action', 'opponent_action', 'focal_role', 'opponent_role'])
        role_names = data['role_names']
        master = int(data['master_seed'])
        game_seed = int(data['game_seeds'][game])
        for regime in REGIMES:
            prior = 'reward_auc' if regime == 'controlled_auc' else regime
            probes = data[f'{prior}_exploration']
            assessment = data[f'{regime}_assessment']
            for reset in range(repeats):
                sample = game * repeats + reset
                action_roles = role_names[data['action_to_role'][sample]]
                prefix = (path.stem, master, game, game_seed, reset, int(data['reset_seeds'][sample]),
                          data['families'][sample], regime)
                for mi, method in enumerate(data['methods']):
                    for ki, k in enumerate(data['ks']):
                        phases = (
                            ('exploration', str(method), probes[mi, ki, sample, :k]),
                            ('assessment', 'student', assessment[mi, ki, sample]),
                        )
                        for phase, controller, actions in phases:
                            for turn, (focal, opponent) in enumerate(actions, start=1):
                                writer.writerow((*prefix, method, k, phase, turn, controller, focal,
                                                 opponent, action_roles[focal], action_roles[opponent]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('replay', help='rebuild turn-by-turn action archives')
    run.add_argument('run', type=Path)
    run.add_argument('output', type=Path)
    run.add_argument('--workers', type=int, default=1)
    csv_command = commands.add_parser('csv', help='print a recorded game and its resets as CSV')
    csv_command.add_argument('block', type=Path)
    csv_command.add_argument('--game', type=int, default=0, help='zero-based game index within the block')
    args = parser.parse_args()
    if args.command == 'replay':
        if args.workers < 1:
            parser.error('--workers must be positive')
        replay(args.run, args.output, args.workers)
        print(args.output / 'complete.json')
    else:
        write_game_csv(args.block, args.game, sys.stdout)


if __name__ == '__main__':
    main()
