"""Contract checks for nominal action-set equivariance and checkpoint formats."""
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import random
import unittest
from copy import deepcopy

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.action_set import ActionSetActorCritic
    from mamrl.environment import MemoryOneEnv, Observation
    from mamrl.policy import RandomPolicy
    from mamrl.recurrent import RecurrentActorCritic, RecurrentPolicy, load_checkpoint, save_checkpoint
    from mamrl.rollout import run_episode
    from mamrl.rules import RuleSpec, RuleTerm, compile_rule
    from mamrl.training import TrainConfig, collect_trial, replay_trial, train


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class ActionSetContract(unittest.TestCase):
    def test_shared_candidate_features_without_ordinal_indices(self):
        model = ActionSetActorCritic(hidden_size=4, seed=7)
        gru = next(m for m in model.modules() if isinstance(m, torch.nn.GRUCell))
        captures = []
        handle = gru.register_forward_pre_hook(lambda _mod, inputs: captures.append(inputs[0].detach().clone()))
        try:
            model.forward_observation(Observation(1,2,-4.,True), model.initial_hidden(n=3), n=3)
        finally:
            handle.remove()
        actual = captures[0]
        expected = torch.tensor([[0,0,math.tanh(-1),1,1/3],
                                 [1,0,math.tanh(-1),1,1/3],
                                 [0,1,math.tanh(-1),1,1/3]], dtype=torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_arbitrary_label_bijections_equivariance_and_shared_gradient(self):
        before = torch.random.get_rng_state().clone()
        model = ActionSetActorCritic(hidden_size=4, seed=13)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        shapes = {name:tuple(p.shape) for name,p in model.named_parameters()}
        for n in (3,4,5):
            candidates = list(permutations(range(n))) if n == 3 else [tuple(reversed(range(n))), tuple((x+2)%n for x in range(n))]
            hidden = torch.arange(n*4, dtype=torch.float32).reshape(n,4)/17
            for p in candidates:
                obs = Observation(0,n-1,-.5,False)
                mapped_obs = Observation(p[0],p[n-1],-.5,False)
                mapped_hidden = torch.empty_like(hidden)
                for a in range(n):
                    mapped_hidden[p[a]] = hidden[a]
                logits, value, next_hidden = model.forward_observation(obs, hidden, n=n)
                mapped_logits, mapped_value, mapped_next = model.forward_observation(mapped_obs, mapped_hidden, n=n)
                torch.testing.assert_close(mapped_logits[list(p)], logits, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(mapped_next[list(p)], next_hidden, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(mapped_value, value, rtol=1e-6, atol=1e-7)
                weights = torch.arange(1,n+1, dtype=torch.float32)/n
                g = torch.autograd.grad((weights*logits).sum()+value, tuple(model.parameters()), retain_graph=True)
                g_mapped = torch.autograd.grad((weights*mapped_logits[list(p)]).sum()+mapped_value, tuple(model.parameters()))
                for a,b in zip(g,g_mapped):
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            self.assertEqual(shapes, {name:tuple(p.shape) for name,p in model.named_parameters()})
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_zero_hidden_memoryless_and_invalid_hidden(self):
        model = ActionSetActorCritic(hidden_size=4, seed=3, memoryless=True)
        obs = Observation(0,1,2.,False)
        zeros = model.initial_hidden(n=3)
        other = torch.full((3,4), 1.25)
        first = model.forward_observation(obs, zeros, n=3)
        second = model.forward_observation(obs, other, n=3)
        for a,b in zip(first,second):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        for invalid in (torch.zeros(4), torch.zeros(4,4), torch.full((3,4),float('nan'))):
            with self.subTest(hidden=invalid), self.assertRaises((TypeError,ValueError)):
                model.forward_observation(obs, invalid, n=3)
        with self.assertRaises((TypeError,ValueError)):
            model.forward_observation(Observation(-1,1,0.,True), zeros, n=3)

    def test_v1_and_v2_checkpoint_roundtrip_and_run_policy(self):
        with TemporaryDirectory() as tmp:
            models = [RecurrentActorCritic(hidden_size=4, seed=2),
                      ActionSetActorCritic(hidden_size=4, seed=2),
                      ActionSetActorCritic(hidden_size=4, seed=2, memoryless=True)]
            for index, model in enumerate(models):
                path = Path(tmp)/f'{index}.pt'
                save_checkpoint(model, path)
                raw = torch.load(path, map_location='cpu', weights_only=True)
                if index == 0:
                    self.assertEqual(raw['format_version'], 1)
                else:
                    self.assertEqual(raw['format_version'], 2)
                    self.assertEqual(raw['architecture']['kind'], 'action_set_v1' if index == 1 else 'action_set_ff_v1')
                loaded = load_checkpoint(path)
                for n in (3,4,5):
                    obs = Observation(1,2,0.,True)
                    a = model.forward_observation(obs, model.initial_hidden(n=n), n=n)
                    b = loaded.forward_observation(obs, loaded.initial_hidden(n=n), n=n)
                    for x,y in zip(a,b):
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
                    tensor = compile_rule(RuleSpec(n,(RuleTerm('action_bonus',1,1),)))
                    episode = run_episode(MemoryOneEnv(tensor,max_steps=2),
                                          (RecurrentPolicy(loaded),RandomPolicy()), seed=4,
                                          policy_seeds=(5,6))
                    self.assertEqual(len(episode.transitions), 2)
                previous = path.read_bytes()
                with self.assertRaises((FileExistsError,ValueError)):
                    save_checkpoint(model,path)
                self.assertEqual(path.read_bytes(), previous)

    def test_v2_checkpoint_rejects_bad_kind_shape_and_nonfinite_weights(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'good.pt'
            save_checkpoint(ActionSetActorCritic(hidden_size=4, seed=2),path)
            original = torch.load(path,map_location='cpu',weights_only=True)
            key = next(iter(original['state_dict']))
            for case in ('kind','shape','nonfinite','extra-key'):
                payload = deepcopy(original)
                if case == 'kind':
                    payload['architecture']['kind'] = 'unknown'
                elif case == 'shape':
                    payload['state_dict'][key] = torch.zeros(1)
                elif case == 'nonfinite':
                    payload['state_dict'][key].view(-1)[0] = float('nan')
                else:
                    payload['state_dict']['unrecognized'] = torch.zeros(1)
                bad = Path(tmp)/f'{case}.pt'
                torch.save(payload,bad)
                with self.subTest(case=case), self.assertRaises((TypeError,ValueError,RuntimeError)):
                    load_checkpoint(bad)

    def test_recurrent_collector_replay_and_training_updates(self):
        for n in (3,4,5):
            model = ActionSetActorCritic(hidden_size=4, seed=11)
            rule = RuleSpec(n,(RuleTerm('action_bonus',1,1),))
            trial = collect_trial(model,rule,episodes_per_trial=2,max_steps=2,seed=7)
            self.assertEqual(trial, collect_trial(model,rule,episodes_per_trial=2,max_steps=2,seed=7,opponent_kind='random'))
            self.assertEqual(trial.action_event_indices, (0,1,3,4))
            logs,values,_ = replay_trial(model,trial)
            torch.testing.assert_close(logs.detach(),torch.tensor(trial.old_log_probs),rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(values.detach(),torch.tensor(trial.old_values),rtol=1e-6,atol=1e-6)
        for architecture in ('action_set_v1','action_set_ff_v1'):
            cfg = TrainConfig(n=3,architecture=architecture,hidden_size=4,updates=1,
                              trials_per_update=1,episodes_per_trial=2,max_steps=2,epochs=1,
                              evaluation_trials=2,evaluation_interval=1,seed=11)
            result = train(cfg)
            self.assertIsInstance(result.model,ActionSetActorCritic)
            self.assertEqual(result.model.memoryless,architecture=='action_set_ff_v1')
            self.assertTrue(any(not torch.equal(result.model.state_dict()[k],result.initial_model.state_dict()[k])
                                for k in result.model.state_dict()))


if __name__ == '__main__':
    unittest.main()
