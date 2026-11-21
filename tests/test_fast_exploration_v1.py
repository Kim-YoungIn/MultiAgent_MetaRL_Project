"""Focused checks for the *new* study. Legacy evidence is not rewritten."""
import unittest
import tempfile
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import torch

from research.fast_exploration_v1.prior import auc_weights, absolute_auc, fit_return_prior
from research.fast_exploration_v1.study import KS, BeliefActor, rollout, evaluate, train_teacher
from research.opponent_adaptation.data import sample
from research.structured_scale.run import FAMILIES, D, Model


class FastExplorationTests(unittest.TestCase):
    def test_absolute_auc_not_gain(self):
        self.assertAlmostEqual(sum(auc_weights(KS)),1.)
        self.assertAlmostEqual(absolute_auc(KS,[5.]*len(KS)),5.)
        self.assertGreater(absolute_auc(KS,[4.]*len(KS)),
                           absolute_auc(KS,[0.,1.,2.,3.,4.]))
        with self.assertRaises(ValueError):
            auc_weights((1,2))

    def test_prior_has_finite_coefficients(self):
        data=sample(93814,4)
        mu=fit_return_prior(data,steps=3,seed=2)
        self.assertEqual(mu.shape,(D,))
        self.assertTrue(np.isfinite(mu).all())

    def test_identical_zero_shot_and_replay(self):
        prior=np.zeros(D)
        actor=BeliefActor(seed=3).eval()
        seed=1577
        methods=('learned','blind','random','greedy')
        a=[rollout(seed,FAMILIES[0],prior,0,teacher=actor if m in ('learned','blind') else None,method=m,h=4)
           for m in methods]
        self.assertEqual(len({x['total'] for x in a}),1)
        one=rollout(seed,FAMILIES[2],prior,4,method='random',h=4)
        two=rollout(seed,FAMILIES[2],prior,4,method='random',h=4)
        self.assertEqual(one['total'],two['total'])

    def test_teacher_explicitly_reads_posterior(self):
        actor=BeliefActor(seed=19).eval()
        student=Model(3,np.zeros(D))
        obs=SimpleNamespace(previous_own_action=0,previous_opponent_action=1,
                            previous_own_reward=0.,episode_start_flag=False)
        x=np.zeros((3,D))
        x[:,0]=1
        x[np.arange(3),5+np.arange(3)]=1
        with torch.no_grad():
            blind,_,_=actor.forward_observation(obs,actor.initial_hidden(n=3),
                                                 student=student,x=x,belief=False)
            informed,_,_=actor.forward_observation(obs,actor.initial_hidden(n=3),
                                                    student=student,x=x,belief=True)
        self.assertFalse(torch.allclose(blind,informed))

    def test_training_and_controlled_evaluation(self):
        prior=np.zeros(D)
        cfg={'teacher_updates':1,'batch':4}
        teacher=train_teacher(prior,611,cfg)
        self.assertTrue(all(torch.isfinite(p.detach()).all().item()
                            for p in teacher.parameters()))
        result=evaluate(prior,teacher,[111,222,333,444],h=4,controlled=True,blind_teacher=teacher)
        self.assertTrue(result['r0_equal'])
        for method, curve in result['curves'].items():
            self.assertEqual(len(curve),len(KS))
            for k in KS:
                self.assertEqual(len(result['assessment_steps'][method][str(k)]),4)
            self.assertAlmostEqual(curve[KS.index(8)],
                np.mean(result['assessment_steps'][method]['8']))

    def test_manifest_tracks_environment_and_evaluation_sources(self):
        from research.fast_exploration_v1.study import SOURCE_PATHS, source_fingerprints
        self.assertIn('research/structured_scale/run.py', SOURCE_PATHS)
        self.assertIn('research/opponent_adaptation/study.py', SOURCE_PATHS)
        self.assertIn('05_toy_models/mamrl/structured_games.py', SOURCE_PATHS)
        self.assertEqual(set(source_fingerprints()), set(SOURCE_PATHS))

    def test_figure_outputs_and_auc_integrity(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Plotting is optional until matplotlib is installed')
        from research.fast_exploration_v1.figure import render
        from research.fast_exploration_v1.prior import absolute_auc
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for block in range(2):
                curves={m:[float(block+k/16+0.01*i) for k in KS]
                        for i,m in enumerate(('learned','blind','random','greedy'))}
                data={'reward_auc':{'ks':list(KS),'curves':curves,
                      'auc':{m:absolute_auc(KS,v) for m,v in curves.items()},
                      'assessment_steps':{m:{str(k):[float(v)]*4 for k,v in zip(KS,curve)}
                                          for m,curve in curves.items()}}}
                (root/f'block_{block:03d}.json').write_text(json.dumps(data))
            written=render(root,k_for_steps=8)
            self.assertEqual(len(written['figures']),3)
            for file in written['figures']:
                self.assertTrue(Path(file).is_file())
            self.assertTrue((root/'figures'/'adaptation_curve.csv').is_file())
            self.assertTrue((root/'figures'/'auc.csv').is_file())
            self.assertTrue((root/'figures'/'adaptation_slopes.csv').is_file())
            # If a manifest is present, never graph an incomplete sealed run.
            (root/'manifest.json').write_text(json.dumps({'config': {'blocks': 3}}))
            with self.assertRaisesRegex(ValueError,'incomplete run'):
                render(root)
            (root/'manifest.json').unlink()
            # Detect data corruption: source AUC must match numeric curve area.
            path=root/'block_000.json'
            bad=json.loads(path.read_text())
            bad['reward_auc']['auc']['learned']+=1
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'saved AUC'):
                render(root)


# CUDA regression tests are appended separately so CPU-only CI still exercises
# device selection, batched GRU autograd, and explicit no-silent-fallback policy.
class CudaExecutionTests(unittest.TestCase):
    def test_device_selection(self):
        from research.fast_exploration_v1.study import resolve_device
        self.assertEqual(str(resolve_device('cpu')),'cpu')
        self.assertEqual(str(resolve_device('auto')),
                         'cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError,'CUDA requested'):
                resolve_device('cuda')

    def test_batched_replay_gradients(self):
        from research.fast_exploration_v1.study import _train_loss
        model=BeliefActor(seed=12)
        record=dict(features=[np.ones((3,7),dtype=np.float32)]*2,
                    actions=[0,1], assessment=0.42)
        loss=_train_loss(model,[record,record])
        loss.backward()
        self.assertTrue(np.isfinite(loss.item()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable in CI')
    def test_cuda_prior_and_teacher_training(self):
        from research.fast_exploration_v1.study import resolve_device
        from research.fast_exploration_v1.prior import fit_return_prior
        from research.opponent_adaptation.data import sample
        device=resolve_device('cuda')
        self.assertEqual(device.type,'cuda')
        data=sample(7431,4)
        mu=fit_return_prior(data,steps=2,seed=7,device='cuda')
        self.assertTrue(np.isfinite(mu).all())
        actor=train_teacher(mu,seed=17,cfg={'teacher_updates':1,'batch':2},device='cuda')
        self.assertEqual(next(actor.parameters()).device.type,'cuda')
        self.assertTrue(all(torch.isfinite(p).all().item() for p in actor.parameters()))
        # Regression: a CPU Generator cannot sample a CUDA probability tensor.
        # Direct GPU-backed evaluation must work as well as execute()'s CPU copy.
        with torch.no_grad():
            scores=evaluate(mu,actor,[17001],h=3,controlled=True,blind_teacher=actor)
        self.assertTrue(scores['r0_equal'])
        self.assertEqual(next(actor.parameters()).device.type,'cuda')


if __name__=='__main__':
    unittest.main()
