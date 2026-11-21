"""Rebuild block scores from saved rewards and replay samples with StructuredEnv."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import batch, study
from .scale import digest

REGIMES = ('reward_auc','controlled_auc','pooled_auc','zero_auc')


def verify(output, replay=True):
    manifest=json.loads((output/'manifest.json').read_text())
    cfg=manifest['config']
    ks=np.array(cfg['ks'])
    h=cfg['h']
    methods=list(batch.METHODS)
    n_games, n_resets=cfg['test_games'],cfg['test_resets']
    samples=n_games*n_resets
    expected_shape=(len(methods),len(ks),samples,h)
    seen_test,seen_prior=set(),set()
    assessments=[]
    checked_replays=0
    for file, checksum in manifest['source'].items():
        if digest(output/'source'/file)!=checksum:
            raise ValueError(f'source hash mismatch: {file}')
    for index in range(cfg['blocks']):
        path=output/f'block_{index:03d}'
        record=json.loads(path.with_suffix('.json').read_text())
        expected_master=manifest['master_seeds'][index//cfg['blocks_per_master']]
        assert record['master_seed']==expected_master
        assert record['seed_block']==index%cfg['blocks_per_master']
        assert record['seeds']==np.random.SeedSequence([expected_master,record['seed_block']]).generate_state(3).tolist()
        assert digest(path.with_suffix('.npz'))==record['raw_sha256']
        assert digest(path.with_suffix('.pt'))==record['weights_sha256']
        with np.load(path.with_suffix('.npz')) as file:
            raw=dict(file)
        assert list(raw['methods'])==methods
        np.testing.assert_array_equal(raw['ks'],ks)
        assert len(set(raw['game_seeds']))==n_games
        assert len(set(raw['reset_seeds']))==samples
        assert len(set(raw['test_fingerprints']))==n_games
        assert len(set(raw['prior_fingerprints']))==cfg['prior_games']
        assert seen_test.isdisjoint(raw['test_fingerprints'])
        seen_test.update(raw['test_fingerprints'])
        seen_prior.update(raw['prior_fingerprints'])
        for name in methods[:2]:
            assert raw[f'{name}_training_return'].shape==(cfg['teacher_updates'],)
            assert np.isfinite(raw[f'{name}_training_return']).all()
        for regime in REGIMES:
            rewards=raw[regime+'_assessment']
            explore=raw[regime+'_exploration']
            assert rewards.shape==expected_shape and explore.shape==expected_shape[:-1]
            assert np.isfinite(rewards).all() and np.isfinite(explore).all()
            assert np.all(explore[:,0]==0)
            assert np.all(rewards[:,0]==rewards[0,0])
            expected=record[regime]
            per_turn=rewards.mean(axis=2)
            mean=per_turn.mean(axis=2)
            totals=(rewards.sum(3)+explore).mean(2)
            auc=np.trapezoid(mean,x=ks,axis=1)/ks[-1]
            for j,method in enumerate(methods):
                np.testing.assert_allclose(expected['curves'][method],mean[j],atol=1e-12)
                np.testing.assert_allclose(expected['total_return'][method],totals[j],atol=1e-12)
                np.testing.assert_allclose(expected['auc'][method],auc[j],atol=1e-12)
                np.testing.assert_allclose([expected['assessment_steps'][method][str(k)] for k in ks],
                                           per_turn[j],atol=1e-12)
            if regime=='reward_auc':
                assessments.append(auc)
        np.testing.assert_array_equal(raw['reward_auc_exploration'],raw['controlled_auc_exploration'])
        if replay:
            checked_replays+=replay_one_per_family(path,raw,record,ks,h,n_resets)
    assert seen_prior.isdisjoint(seen_test)
    aucs=np.array(assessments)
    diff=aucs[:,:1]-aucs[:,1:]
    rng=np.random.default_rng(882734)
    size=cfg['blocks_per_master']
    groups=[np.arange(start,start+size) for start in range(0,len(aucs),size)]
    draws=np.concatenate([rng.choice(group,(20000,size),replace=True) for group in groups],axis=1)
    intervals=np.quantile(diff[draws].mean(axis=1),[.025,.975],axis=0).T
    result_record=json.loads((output/'analysis'/'report.json').read_text())
    paired=[r for r in result_record['paired_contrasts'] if r['regime']=='reward_auc']
    np.testing.assert_allclose([r['mean'] for r in paired],diff.mean(0),atol=1e-12)
    np.testing.assert_allclose([[r['ci_low'],r['ci_high']] for r in paired],intervals,atol=1e-12)
    result=dict(independent_blocks=len(aucs),test_games=len(seen_test),resets_per_game=n_resets,
                exact_test_prior_overlap=0,checked_scalar_trajectories=checked_replays,
                raw_payoff_and_source_hashes_match=True,absolute_auc=aucs.mean(0).tolist(),
                paired_effects=diff.mean(0).tolist(),paired_ci95=intervals.tolist())
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def replay_one_per_family(path,raw,record,ks,h,resets):
    original_game = study.game
    checkpoint=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
    actors={}
    for name,state in checkpoint.items():
        actors[name]=study.BeliefActor()
        actors[name].load_state_dict(state)
    prior=np.array(record['trained_prior'])
    for i in range(4):
        j=i*resets
        seed, family = int(raw['reset_seeds'][j]),str(raw['families'][j])
        game_seed=int(raw['game_seeds'][i])
        rule=original_game(game_seed,family,3,3)
        assert rule.fingerprint()==raw['test_fingerprints'][i]
        with patch.object(study,'game',return_value=rule),torch.no_grad():
            for method_id,method in enumerate(batch.METHODS):
                k_index=1 if i%2==0 else len(ks)-1
                k=int(ks[k_index])
                for regime in ('reward_auc','controlled_auc'):
                    frozen=regime=='controlled_auc'
                    run=study.rollout(seed,family,prior,k,method=method,teacher=actors.get(method),h=h,
                                      freeze_assessment=frozen,neutralize_opponent=frozen,
                                      opponent_learning=not frozen)
                    np.testing.assert_allclose(run['assessment_rewards'],
                        raw[regime+'_assessment'][method_id,k_index,j],atol=1e-12,rtol=0,
                        err_msg=f'{path}: {regime} {method} {family} {k}')
                    np.testing.assert_allclose(run['explore_sum'],
                        raw[regime+'_exploration'][method_id,k_index,j],atol=1e-12,rtol=0)
    return 4*len(batch.METHODS)*2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--skip-scalar',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(verify(args.output,replay=not args.skip_scalar),indent=2))
