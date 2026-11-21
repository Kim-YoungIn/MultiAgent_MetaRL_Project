import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'torch required')
class TrainingContract(unittest.TestCase):
    def test_context_actor_equivariance_and_batch(self):
        from experiment.model import ContextActor
        from mamrl.environment import Observation
        model = ContextActor(seed=7)
        obs = Observation(0, 2, .5, False)
        hidden = torch.randn(3, 32)
        permutation = [2, 0, 1]
        inverse = [permutation.index(i) for i in range(3)]
        other = Observation(inverse[obs.previous_own_action], inverse[obs.previous_opponent_action], .5, False)
        logits, value, next_hidden = model.forward_observation(obs, hidden, n=3)
        result = model.forward_observations([obs, other], torch.stack([hidden, hidden[permutation]]), n=3)
        self.assertTrue(torch.allclose(logits[permutation], result[0][1], atol=1e-6))
        self.assertTrue(torch.allclose(value, result[1][1], atol=1e-6))
        self.assertTrue(torch.allclose(next_hidden[permutation], result[2][1], atol=1e-6))

    def test_exact_resume(self):
        from experiment.training import train
        from mamrl.diverse_catalog import build_diverse_manifest
        manifest = build_diverse_manifest(train_rules=2, fresh_rules=2, structural_rules=2)
        entries = [r for r in manifest['rules'] if r['split'] == 'train']
        with tempfile.TemporaryDirectory() as root:
            for method in ('M', 'P', 'O'):
                kw = dict(method=method, seed=7, profiles=('steady', 'sarsa'), updates=2, trials=2, k=2, h=2)
                uninterrupted = train(root+'/'+method+'a', entries, **kw)
                train(root+'/'+method+'b', entries, stop=1, **kw)
                resumed = train(root+'/'+method+'b', entries, **kw)
                for key, value in uninterrupted.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[key]))
