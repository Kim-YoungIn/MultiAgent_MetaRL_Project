"""Independent diagnostic metrics, paired uncertainty and supervised probe checks."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.environment import Observation
    from mamrl.recurrent import RecurrentActorCritic, load_checkpoint
    from mamrl.training import collect_trial
    from mamrl.diagnostics import balanced_specs, behavior_metrics, paired_difference, supervised_probe
    from mamrl.rules import RuleSpec, RuleTerm


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class DiagnosticsContract(unittest.TestCase):
    def test_diagnostic_cli_small_protocol_and_record_provenance(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'study'
            cmd = [sys.executable, '-m', 'mamrl.diagnose', '--seeds', '11',
                   '--updates', '2', '--short-updates', '1', '--supervised-steps', '1',
                   '--evaluation-repeats', '2', '--trials-per-update', '1', '--max-steps', '2',
                   '--hidden-size', '4', '--output-dir', str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, result.stderr)
            stdout = json.loads(result.stdout)
            self.assertEqual((stdout['fixed_runs'],stdout['variable_snapshots'],stdout['supervised_runs']), (3,2,1))
            record_path = output/'diagnostic.json'
            report_path = output/'report.html'
            record = json.loads(record_path.read_text(encoding='utf-8'))
            self.assertTrue(report_path.is_file())
            self.assertEqual(record['schema_version'], 1)
            self.assertEqual(record['protocol']['seeds'], [11])
            self.assertEqual(record['protocol']['short_updates'], 1)
            self.assertEqual(record['protocol']['long_updates'], 2)
            self.assertEqual(record['protocol']['evaluation_trials'], 6)
            fixed, variable = record['fixed_runs'], record['variable_runs']
            self.assertEqual([(row['target'],row['update']) for row in fixed], [(0,1),(1,1),(2,1)])
            self.assertEqual([(row['target'],row['update']) for row in variable], [(None,1),(None,2)])
            self.assertEqual(record['supervised_runs'][0]['examples'], 18)
            self.assertEqual(variable[0]['history'], variable[1]['history'][:1])
            common_seeds = None
            for row in fixed + variable:
                self.assertEqual(row['seed'], 11)
                self.assertEqual(len(row['history']), row['update'])
                checkpoint = output/row['checkpoint']
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), row['checkpoint_sha256'])
                self.assertEqual(json.loads((output/row['raw_record']).read_text(encoding='utf-8')), row)
                comparison = row['comparisons']
                targets = [t['terms'][0]['action'] for t in row['evaluation_specs']]
                expected_targets = ([row['target']]*6 if row['target'] is not None else [0,1,2,0,1,2])
                self.assertEqual(targets, expected_targets)
                for label in ('initial','random','trained','hidden_reset'):
                    self.assertEqual(comparison[label]['trial_targets'], expected_targets)
                    if common_seeds is None:
                        common_seeds = comparison[label]['trial_seeds']
                    self.assertEqual(comparison[label]['trial_seeds'], common_seeds)
                self.assertAlmostEqual(row['paired_vs_random']['mean'],
                                       comparison['trained']['mean_reward_per_step'] - comparison['random']['mean_reward_per_step'])
                self.assertAlmostEqual(row['paired_memory_benefit']['mean'],
                                       comparison['trained']['mean_reward_per_step'] - comparison['hidden_reset']['mean_reward_per_step'])
            initial_paths = [output/f'fixed_target{target}_seed11'/'initial.pt' for target in range(3)]
            initial_paths.append(output/'variable_seed11'/'initial.pt')
            states = [load_checkpoint(path).state_dict() for path in initial_paths]
            for other in states[1:]:
                for key in states[0]:
                    self.assertTrue(torch.equal(states[0][key], other[key]))
            variable_model = load_checkpoint(output/variable[-1]['checkpoint'])
            replay = behavior_metrics(variable_model, balanced_specs(3,2), episodes_per_trial=2,
                                      max_steps=2, seed=record['protocol']['evaluation_seed'])
            self.assertEqual(replay, variable[-1]['comparisons']['trained'])
            html = report_path.read_text(encoding='utf-8')
            self.assertNotRegex(html.lower(), r'<\s*script\b|@import\b|https?://')
            for row in fixed+variable:
                self.assertIn(str(row['comparisons']['trained']['mean_reward_per_step']), html)
            snapshot = record_path.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(record_path.read_bytes(), snapshot)

    def test_balanced_and_fixed_targets(self):
        balanced = balanced_specs(n=3, repeats=2)
        self.assertEqual([s.terms[0].action for s in balanced], [0,1,2,0,1,2])
        self.assertTrue(all(s.n == 3 and s.terms[0].kind == 'action_bonus'
                            and s.terms[0].coefficient == 1 for s in balanced))
        fixed = balanced_specs(n=3, repeats=2, fixed_action=1)
        self.assertEqual([s.terms[0].action for s in fixed], [1]*6)
        for bad in (True,-1,3):
            with self.subTest(fixed_action=bad), self.assertRaises((TypeError,ValueError)):
                balanced_specs(n=3, repeats=2, fixed_action=bad)

    def test_behavior_metrics_derived_from_action_transitions_and_stratified_se(self):
        specs = balanced_specs(n=3, repeats=2)
        seed = 918273
        before = torch.random.get_rng_state().clone()
        result = behavior_metrics(None, specs, episodes_per_trial=2, max_steps=3, seed=seed)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        rng = random.Random(seed)
        seeds = [rng.getrandbits(63) for _ in specs]
        trials = [collect_trial(None, spec, episodes_per_trial=2, max_steps=3, seed=s)
                  for spec,s in zip(specs,seeds)]
        self.assertEqual(result['trial_seeds'], seeds)
        self.assertEqual(result['trial_targets'], [0,1,2,0,1,2])
        self.assertEqual(result['steps_per_trial'], 6)
        returns = [sum(t.rewards) for t in trials]
        means = [x/6 for x in returns]
        self.assertEqual(result['trial_returns'], returns)
        self.assertAlmostEqual(result['mean_reward_per_step'], statistics.mean(means))
        self.assertAlmostEqual(result['pooled_standard_error'], statistics.stdev(means)/math.sqrt(6))
        group_values = [[means[i] for i in range(6) if i%3 == target] for target in range(3)]
        expected_se = math.sqrt(sum((2/6)**2 * statistics.variance(group)/2 for group in group_values))
        self.assertAlmostEqual(result['standard_error'], expected_se)
        self.assertEqual(result['standard_error_method'], 'stratified_by_target')
        for target, group in enumerate(group_values):
            stats = result['by_target'][str(target)]
            self.assertEqual(stats['trials'], 2)
            self.assertAlmostEqual(stats['mean_reward_per_step'], statistics.mean(group))
            self.assertAlmostEqual(stats['standard_error'], statistics.stdev(group)/math.sqrt(2))
        successes = opportunities = 0
        for trial in trials:
            for ep in trial.episodes:
                for transition in ep.transitions:
                    obs = transition.observations[0]
                    if not obs.episode_start_flag and obs.previous_own_reward > 0:
                        opportunities += 1
                        successes += transition.actions[0] == obs.previous_own_action
        self.assertEqual(result['success_opportunities'], opportunities)
        self.assertEqual(result['repeat_after_success_count'], successes)
        self.assertEqual(result['repeat_after_success_rate'], successes/opportunities if opportunities else None)
        self.assertAlmostEqual(result['second_episode_first_action_mean'],
                               statistics.mean(t.episodes[1].transitions[0].rewards[0] for t in trials))
        self.assertAlmostEqual(result['second_episode_first_three_actions_mean'],
                               statistics.mean(sum(x.rewards[0] for x in t.episodes[1].transitions)/3 for t in trials))

    def test_no_repeat_opportunities_and_singleton_strata(self):
        specs = balanced_specs(n=3, repeats=1)
        metrics = behavior_metrics(None, specs, episodes_per_trial=2, max_steps=1, seed=4)
        self.assertEqual(metrics['success_opportunities'], 0)
        self.assertEqual(metrics['repeat_after_success_count'], 0)
        self.assertIsNone(metrics['repeat_after_success_rate'])
        self.assertIsNone(metrics['standard_error'])
        self.assertEqual(metrics['second_episode_first_action_mean'], metrics['second_episode_first_three_actions_mean'])
        single_episode = behavior_metrics(None, specs, episodes_per_trial=1, max_steps=2, seed=4)
        self.assertIsNone(single_episode['second_episode_first_action_mean'])
        self.assertIsNone(single_episode['second_episode_first_three_actions_mean'])

    def test_paired_target_strata_means_se_and_mismatched_provenance(self):
        # Differences per step: [.5, 1, 1.5, 2]; targets 0,1,0,1.
        common = {'steps_per_trial':2, 'trial_targets':[0,1,0,1],
                  'trial_seeds':[10,11,12,13], 'trials':4, 'n':2,
                  'episodes_per_trial':1, 'max_steps':2,
                  'spec_fingerprints':['a'*64,'b'*64,'a'*64,'b'*64]}
        left = {**common, 'trial_returns':[1,2,3,4]}
        right = {**common, 'trial_returns':[0,0,0,0]}
        paired = paired_difference(left, right)
        self.assertEqual(paired['trial_differences'], [.5,1,1.5,2])
        self.assertAlmostEqual(paired['mean'], 1.25)
        self.assertAlmostEqual(paired['standard_error'], math.sqrt(.125))
        self.assertEqual(paired['by_target']['0']['trial_differences'], [.5,1.5])
        self.assertAlmostEqual(paired['by_target']['0']['mean'], 1)
        self.assertAlmostEqual(paired['by_target']['0']['standard_error'], .5)
        self.assertEqual(paired['by_target']['1']['trial_differences'], [1,2])
        self.assertAlmostEqual(paired['by_target']['1']['mean'], 1.5)
        self.assertAlmostEqual(paired['by_target']['1']['standard_error'], .5)
        for key, replacement in (('steps_per_trial',3), ('trial_targets',[1,0,0,1]),
                                 ('trial_seeds',[10,11,12,14]), ('trial_returns',[0,0]), ('trials',3),
                                 ('n',3), ('episodes_per_trial',2), ('max_steps',1),
                                 ('spec_fingerprints',['a'*64,'b'*64,'a'*64,'c'*64])):
            mismatch = deepcopy(right)
            mismatch[key] = replacement
            with self.subTest(key=key), self.assertRaises((ValueError,TypeError)):
                paired_difference(left, mismatch)
        both_bad_left, both_bad_right = deepcopy(left), deepcopy(right)
        both_bad_left['steps_per_trial'] = both_bad_right['steps_per_trial'] = 3
        with self.assertRaises(ValueError):
            paired_difference(both_bad_left, both_bad_right)
        one_per = {'steps_per_trial':1, 'trial_targets':[0,1], 'trial_seeds':[5,6], 'trials':2,
                   'n':2, 'episodes_per_trial':1, 'max_steps':1,
                   'spec_fingerprints':['a'*64,'b'*64]}
        self.assertIsNone(paired_difference({**one_per,'trial_returns':[1,1]},
                                            {**one_per,'trial_returns':[0,0]})['standard_error'])

    def test_paired_rejects_different_episode_boundaries_or_action_spaces(self):
        specs3 = balanced_specs(3,2)
        specs4 = tuple(RuleSpec(4, (RuleTerm('action_bonus',1,s.terms[0].action),)) for s in specs3)
        left = behavior_metrics(None, specs3, episodes_per_trial=2, max_steps=2, seed=4)
        other_boundaries = behavior_metrics(None, specs3, episodes_per_trial=1, max_steps=4, seed=4)
        other_n = behavior_metrics(None, specs4, episodes_per_trial=2, max_steps=2, seed=4)
        for right in (other_boundaries, other_n):
            self.assertEqual(left['steps_per_trial'], right['steps_per_trial'])
            self.assertEqual(left['trial_targets'], right['trial_targets'])
            self.assertEqual(left['trial_seeds'], right['trial_seeds'])
            with self.assertRaises(ValueError):
                paired_difference(left, right)

    def test_supervised_probe_labels_floor_and_rng_preservation(self):
        seed = 14
        model = RecurrentActorCritic(hidden_size=4, seed=seed)
        expected_losses, success_probs, failure_avoids = [], [], []
        with torch.no_grad():
            for own in range(3):
                for other in range(3):
                    for reward in (0,1):
                        obs = Observation(own,other,float(reward),False)
                        logits,_,_ = model.forward_observation(obs, model.initial_hidden(), n=3)
                        log_probs = torch.log_softmax(logits, dim=0)
                        if reward:
                            expected_losses.append(-log_probs[own].item())
                            success_probs.append(log_probs[own].exp().item())
                        else:
                            expected_losses.append(-sum(log_probs[a].item() for a in range(3) if a != own)/2)
                            failure_avoids.append(1-log_probs[own].exp().item())
        before_torch = torch.random.get_rng_state().clone()
        before_python = random.getstate()
        result = supervised_probe(seed=seed, steps=1, hidden_size=4, n=3)
        self.assertTrue(torch.equal(before_torch, torch.random.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        self.assertEqual(result['examples'], 18)
        self.assertAlmostEqual(result['target_entropy'], .5*math.log(2))
        self.assertAlmostEqual(result['initial']['cross_entropy'], statistics.mean(expected_losses), places=6)
        self.assertAlmostEqual(result['initial']['repeat_probability_after_success'], statistics.mean(success_probs), places=6)
        self.assertAlmostEqual(result['initial']['avoid_probability_after_failure'], statistics.mean(failure_avoids), places=6)
        for snapshot in ('initial','final'):
            values = result[snapshot]
            self.assertGreaterEqual(values['cross_entropy'] + 1e-6, result['target_entropy'])
            self.assertAlmostEqual(values['excess_cross_entropy'], values['cross_entropy']-result['target_entropy'])
            self.assertTrue(0 <= values['repeat_probability_after_success'] <= 1)
            self.assertTrue(0 <= values['avoid_probability_after_failure'] <= 1)


if __name__ == '__main__':
    unittest.main()
