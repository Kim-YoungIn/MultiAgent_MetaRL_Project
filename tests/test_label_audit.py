"""Independent physical-label, gradient and reward audit contracts."""
import itertools
import math
import random
from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from html.parser import HTMLParser

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.environment import Observation
    from mamrl.recurrent import RecurrentActorCritic
    from mamrl.label_audit import (PermutedActorCritic, reward_audit,
                                   actor_gradient_audit, loss_gradient_audit)
    from mamrl.training import TrainConfig


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class LabelAuditContract(unittest.TestCase):
    def test_all_observation_mappings_logits_in_physical_order_and_grads(self):
        baseline = RecurrentActorCritic(hidden_size=4, seed=11)
        torch_before = torch.random.get_rng_state().clone()
        for shift in (0,1,2):
            p = tuple((a+shift)%3 for a in range(3))
            model = PermutedActorCritic(hidden_size=4, seed=11, permutation=p)
            self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
            for key in baseline.state_dict():
                self.assertTrue(torch.equal(model.state_dict()[key], baseline.state_dict()[key]))
            for own, opp, reward, start in itertools.product(range(3), range(3), (-1.,0.,2.), (False,True)):
                obs = Observation(own,opp,reward,start)
                internal_obs = Observation(p[own],p[opp],reward,start)
                physical, physical_value, physical_hidden = model.forward_observation(obs, model.initial_hidden(), n=3)
                internal, internal_value, internal_hidden = baseline.forward_observation(internal_obs, baseline.initial_hidden(), n=3)
                torch.testing.assert_close(physical, internal[list(p)], atol=0, rtol=0)
                torch.testing.assert_close(physical_value, internal_value, atol=0, rtol=0)
                torch.testing.assert_close(physical_hidden, internal_hidden, atol=0, rtol=0)
                self.assertEqual(obs, Observation(own,opp,reward,start))
            # Different coefficients in physical order exercise gradient reordering.
            obs = Observation(0,1,2.,False)
            mapped = Observation(p[0],p[1],2.,False)
            physical, value, _ = model.forward_observation(obs, model.initial_hidden(), n=3)
            internal, base_value, _ = baseline.forward_observation(mapped, baseline.initial_hidden(), n=3)
            weights = torch.tensor([.25,-1.,2.])
            grad = torch.autograd.grad((physical*weights).sum()+value, tuple(model.parameters()))
            expected = torch.autograd.grad((internal[list(p)]*weights).sum()+base_value, tuple(baseline.parameters()))
            for a,b in zip(grad,expected):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_invalid_permutations_and_observation_rejected(self):
        for bad in ((0,0,1), (0,1,3), (-1,0,1), (0,1,True), (0,1,2.0), [], None):
            with self.subTest(permutation=bad), self.assertRaises((TypeError, ValueError)):
                PermutedActorCritic(hidden_size=4, seed=1, permutation=bad)
        model = PermutedActorCritic(hidden_size=4, seed=1, permutation=(1,2,0))
        for obs,n in ((Observation(-1,0,0.,True),3), (Observation(0,-1,0.,True),3),
                      (Observation(3,0,0.,True),3), (Observation(0,0,0.,True),4)):
            with self.subTest(obs=obs,n=n), self.assertRaises((TypeError, ValueError)):
                model.forward_observation(obs, model.initial_hidden(), n=n)

    def test_reward_exhaustion_counts(self):
        result = reward_audit(n=3)
        self.assertTrue(result['passed'])
        self.assertEqual(result['tensor_entries_checked'], 3*2*3**4)
        self.assertEqual(result['environment_steps_checked'], 3*3**4)

    def test_actor_gradient_finite_differences_fixture_and_source_unchanged(self):
        model = RecurrentActorCritic(hidden_size=4, seed=13)
        for i,p in enumerate(model.parameters()):
            p.grad = torch.full_like(p, i+.5)
        weights = {k:v.clone() for k,v in model.state_dict().items()}
        grads = {k:p.grad.clone() for k,p in model.named_parameters()}
        rng_before = torch.random.get_rng_state().clone()
        python_rng = random.getstate()
        actor = deepcopy(model.actor).double()
        gradients = [[],[],[]]
        probabilities = [[],[],[]]
        for own,other in itertools.product(range(3), repeat=2):
            with torch.no_grad():
                _,_,hidden = model.forward_observation(Observation(own,other,0.,True), model.initial_hidden(), n=3)
            features = torch.tensor([[a/2, float(a==own), float(a==other)] for a in range(3)], dtype=torch.float64)
            actor_inputs = torch.cat((hidden.detach().double().expand(3,-1),features), dim=1)
            probs = torch.softmax(actor(actor_inputs).squeeze(-1), dim=0)
            for target in range(3):
                g = torch.autograd.grad(probs[target], tuple(actor.parameters()), retain_graph=True)
                gradients[target].append(torch.cat([x.flatten() for x in g]))
                probabilities[target].append(probs[target].item())
        report = actor_gradient_audit(model, n=3)
        self.assertEqual(report['n'], 3)
        self.assertEqual(report['observation_count'], 9)
        self.assertEqual(len(report['targets']), 3)
        expected_fixture = (-1/6, 0, 1/6)
        for actual, expected in zip(report['linear_ordinal_fixture']['derivatives'], expected_fixture):
            self.assertAlmostEqual(actual, expected, places=12)
        for target, item in enumerate(report['targets']):
            self.assertEqual(item['target'], target)
            self.assertTrue(0 <= item['mean_probability'] <= 1)
            self.assertAlmostEqual(item['mean_probability'], sum(probabilities[target])/9, places=12)
            expected_grad = torch.stack(gradients[target])
            self.assertAlmostEqual(item['mean_gradient_norm'], torch.linalg.vector_norm(expected_grad.mean(dim=0)).item(), places=12)
            per = item['per_observation_gradient_norms']
            self.assertEqual(len(per), 9)
            for actual, expected in zip(per, torch.linalg.vector_norm(expected_grad,dim=1)):
                self.assertAlmostEqual(actual, expected.item(), places=12)
            self.assertAlmostEqual(item['mean_per_observation_gradient_norm'], sum(per)/9)
            self.assertLessEqual(item['mean_gradient_norm'], item['mean_per_observation_gradient_norm']+1e-12)
            coord = item['largest_gradient_coordinate']
            self.assertEqual([d['epsilon'] for d in coord['finite_differences']], [1e-4,1e-5,1e-6])
            for difference in coord['finite_differences']:
                self.assertAlmostEqual(difference['central_difference'], coord['autograd'], places=7)
                self.assertAlmostEqual(difference['absolute_error'],
                                       abs(difference['central_difference']-coord['autograd']))
        self.assertAlmostEqual(sum(x['mean_probability'] for x in report['targets']), 1, places=7)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(random.getstate(), python_rng)
        for key,p in model.named_parameters():
            self.assertTrue(torch.equal(p, weights[key]), key)
            self.assertTrue(torch.equal(p.grad, grads[key]), key)

    def test_loss_component_gradients_against_scalar_analytic_fixture(self):
        model = RecurrentActorCritic(hidden_size=4, seed=13)
        actor = model.actor[2].bias[0]
        gru = model.gru.bias_ih[0]
        critic = model.critic[2].bias[0]
        z = float((critic+gru).detach())
        old = float((actor+gru).detach())
        trial = SimpleNamespace(rewards=(1.,0.), old_values=(0.,0.),
                                old_log_probs=(old,-old), actions=(0,1))
        config = TrainConfig(hidden_size=4, fixed_action=1, trials_per_update=1,
                             episodes_per_trial=1, max_steps=2, gamma=0., gae_lambda=0.)
        weights = {key:value.clone() for key,value in model.state_dict().items()}
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        def replay(m, _trial):
            a,g,c = m.actor[2].bias[0],m.gru.bias_ih[0],m.critic[2].bias[0]
            return torch.stack((a+g,-a-g)), torch.stack((c+g,c+g)), torch.stack((2*a+g,2*a+g))
        with patch('mamrl.label_audit.collect_trial', return_value=trial), \
             patch('mamrl.label_audit.replay_trial', side_effect=replay):
            result = loss_gradient_audit(model,config)
        components = result['components']
        # GAE=(1,0); normalized advantages=(1,-1). At ratio=1 the
        # policy derivative is -1 for a and g. Value derivative is z-.5.
        self.assertAlmostEqual(components['policy']['raw_group_norms']['actor'],1.,places=6)
        self.assertAlmostEqual(components['policy']['raw_group_norms']['gru'],1.,places=6)
        self.assertEqual(components['policy']['raw_group_norms']['critic'],0.)
        self.assertAlmostEqual(components['value']['raw_group_norms']['critic'],abs(z-.5),places=6)
        self.assertAlmostEqual(components['value']['raw_group_norms']['gru'],abs(z-.5),places=6)
        self.assertEqual(components['value']['raw_group_norms']['actor'],0.)
        self.assertAlmostEqual(components['entropy']['raw_group_norms']['actor'],2.,places=6)
        self.assertAlmostEqual(components['entropy']['raw_group_norms']['gru'],1.,places=6)
        expected_norm = math.sqrt((-1-.02)**2 + (.5*(z-.5))**2 + (-1+.5*(z-.5)-.01)**2)
        self.assertAlmostEqual(result['total_gradient_norm'],expected_norm,places=6)
        self.assertAlmostEqual(result['global_clip_multiplier'],min(1.,.5/(expected_norm+1e-6)),places=6)
        for component in components.values():
            for group, raw in component['raw_group_norms'].items():
                self.assertAlmostEqual(component['weighted_group_norms'][group],abs(component['weight'])*raw,places=7)
        for name,p in model.named_parameters():
            self.assertTrue(torch.equal(p,weights[name]))
            self.assertTrue(torch.equal(p.grad,torch.ones_like(p)))

    def test_tiny_audit_cli_restores_physical_weights_and_retains_all_conditions(self):
        from mamrl.audit_labels import load_physical_model
        from mamrl.diagnostics import balanced_specs, behavior_metrics
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'audit'
            cmd = [sys.executable,'-B','-m','mamrl.audit_labels','--output-dir',str(output),
                   '--seeds','11','--updates','1','--evaluation-repeats','2',
                   '--trials-per-update','1','--max-steps','2','--hidden-size','4']
            process = subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            record = json.loads((output/'audit.json').read_text())
            self.assertEqual(len(record['runs']),9)
            self.assertEqual({(r['shift'],r['physical_target']) for r in record['runs']},
                             set(itertools.product(range(3),repeat=2)))
            for row in record['runs']:
                self.assertEqual(row['internal_target'],row['permutation'][row['physical_target']])
                self.assertEqual(json.loads((output/row['raw_record']).read_text()),row)
            row = next(r for r in record['runs'] if r['shift']==2 and r['physical_target']==1)
            model = load_physical_model(output/row['checkpoint'],row['permutation'])
            metrics = behavior_metrics(model,balanced_specs(repeats=2,fixed_action=1),
                                       episodes_per_trial=2,max_steps=2,seed=918273)
            self.assertEqual(metrics,row['comparisons']['trained'])
            html = (output/'report.html').read_text()
            self.assertNotRegex(html.lower(),r'<\s*script\b|@import\b|https?://')
            self.assertIn('norm(mean gradient)',html)
            self.assertIn('load_physical_model',html)
            class Parse(HTMLParser):
                pass
            Parse().feed(html)
            before = (output/'audit.json').read_bytes()
            second = subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((output/'audit.json').read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
