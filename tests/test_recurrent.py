"""Recurrent policy/model contract, RNG boundaries and safe checkpoint checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.environment import Observation
    from mamrl.recurrent import RecurrentActorCritic, RecurrentPolicy, save_checkpoint, load_checkpoint


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class RecurrentContract(unittest.TestCase):
    def test_observation_and_candidate_features(self):
        model = RecurrentActorCritic(hidden_size=8, seed=20)
        captures = {}
        cells = [m for m in model.modules() if isinstance(m, torch.nn.GRUCell)]
        self.assertEqual(len(cells), 1)
        def capture_gru(_module, inputs):
            captures['obs'] = inputs[0].detach().clone()
        handle = cells[0].register_forward_pre_hook(capture_gru)
        actor = next(m for m in model.modules()
                     if isinstance(m, torch.nn.Linear) and m.in_features == 8+3)
        def capture_actor(_module, inputs):
            captures['actor'] = inputs[0].detach().clone()
        actor_handle = actor.register_forward_pre_hook(capture_actor)
        try:
            model.forward_observation(Observation(1,2,-4,True), model.initial_hidden(), n=3)
        finally:
            handle.remove()
            actor_handle.remove()
        expected = torch.tensor([.5, 1., 0., math.tanh(-1), 1., 1/3])
        torch.testing.assert_close(captures['obs'], expected)
        candidates = torch.tensor([[0.,0.,0.], [.5,1.,0.], [1.,0.,1.]])
        torch.testing.assert_close(captures['actor'][:,-3:], candidates)

    def test_n_independent_shape_finite_and_initial_rng_preservation(self):
        state_before = torch.random.get_rng_state().clone()
        model = RecurrentActorCritic(hidden_size=8, seed=20)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))
        shape0 = {name: p.shape for name,p in model.named_parameters()}
        hidden = model.initial_hidden()
        self.assertEqual(hidden.shape, (8,))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertTrue(torch.equal(hidden, torch.zeros(8)))
        for n in (3,4,5):
            obs = Observation(1,2,-3.5,False)
            logits, value, next_hidden = model.forward_observation(obs, hidden, n=n)
            self.assertEqual(logits.shape, (n,))
            self.assertEqual(value.shape, ())
            self.assertEqual(next_hidden.shape, (8,))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(next_hidden).all())
            self.assertEqual({name:p.shape for name,p in model.named_parameters()}, shape0)
        same = RecurrentActorCritic(hidden_size=8, seed=20)
        for a,b in zip(model.parameters(), same.parameters()):
            self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))

    def test_policy_local_generator_seed_and_state(self):
        policy = RecurrentPolicy(RecurrentActorCritic(hidden_size=8, seed=11))
        before = torch.random.get_rng_state().clone()
        state_a = policy.initial_state(n=3, seed=123)
        state_b = policy.initial_state(n=3, seed=123 + 2**63)
        inputs = [Observation(i%3,(i+1)%3,float(i),i == 0) for i in range(8)]
        for obs in inputs:
            out_a = policy.act(obs, state_a, n=3)
            out_b = policy.act(obs, state_b, n=3)
            self.assertEqual(out_a.action, out_b.action)
            self.assertIn(out_a.action, range(3))
            state_a, state_b = out_a.state, out_b.state
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_invalid_model_inputs(self):
        model = RecurrentActorCritic(hidden_size=8, seed=1)
        valid = Observation(1,2,0,True)
        for n in (1, True, 3.5):
            with self.subTest(n=n), self.assertRaises((TypeError, ValueError)):
                model.forward_observation(valid, model.initial_hidden(), n=n)
        for obs in (Observation(3,2,0,True), Observation(1,2,float('nan'),True)):
            with self.subTest(obs=obs), self.assertRaises((TypeError, ValueError)):
                model.forward_observation(obs, model.initial_hidden(), n=3)
        for hidden in (torch.zeros(7), torch.tensor([float('nan')]*8)):
            with self.subTest(hidden=hidden), self.assertRaises((TypeError, ValueError)):
                model.forward_observation(valid, hidden, n=3)

    def test_checkpoint_roundtrip_and_existing_path(self):
        model = RecurrentActorCritic(hidden_size=8, seed=21)
        obs = Observation(1,0,-2,False)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.pt'
            save_checkpoint(model, path)
            saved = path.read_bytes()
            loaded = load_checkpoint(path)
            for n in (3,4,5):
                out = model.forward_observation(obs, model.initial_hidden(), n=n)
                new = loaded.forward_observation(obs, loaded.initial_hidden(), n=n)
                for x,y in zip(out,new):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
            with self.assertRaises((FileExistsError, ValueError)):
                save_checkpoint(model, path)
            self.assertEqual(path.read_bytes(), saved)
            broken = Path(tmp) / 'broken.pt'
            torch.save({'format_version': 1, 'architecture': {'hidden_size': 8}, 'state_dict': {}}, broken)
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                load_checkpoint(broken)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            parameter_name = next(iter(payload['state_dict']))
            payload['state_dict'][parameter_name].view(-1)[0] = float('nan')
            nonfinite = Path(tmp) / 'nonfinite.pt'
            torch.save(payload, nonfinite)
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                load_checkpoint(nonfinite)
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float('nan')
            unsaved = Path(tmp) / 'must-not-exist.pt'
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                save_checkpoint(model, unsaved)
            self.assertFalse(unsaved.exists())


if __name__ == '__main__':
    unittest.main()
