"""Full-trial recurrent PPO collection, replay and frozen evaluation contracts."""
from dataclasses import asdict
import math
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from xml.etree import ElementTree

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from mamrl.recurrent import RecurrentActorCritic, RecurrentPolicy, load_checkpoint
    from mamrl.label_audit import PermutedActorCritic
    from mamrl.environment import MemoryOneEnv
    from mamrl.policy import RandomPolicy
    from mamrl.recording import load_record
    from mamrl.rollout import run_episode
    from mamrl.rules import RuleSpec, RuleTerm
    from mamrl.rules import compile_rule
    from mamrl.training import TrainConfig, collect_trial, replay_trial, evaluate, train
    from mamrl.training import sample_task, evaluation_specs
    from mamrl.rules import generate_rule


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class TrainingContract(unittest.TestCase):
    def test_identity_model_factory_preserves_baseline_training(self):
        config = TrainConfig(n=3, fixed_action=1, hidden_size=4, updates=1, trials_per_update=1,
                             episodes_per_trial=2, max_steps=2, epochs=1,
                             evaluation_trials=2, evaluation_interval=1, seed=19)
        baseline = train(config)
        explicit = train(config, model_factory=lambda hidden_size, seed:
                         RecurrentActorCritic(hidden_size=hidden_size, seed=seed))
        identity = train(config, model_factory=lambda hidden_size, seed:
                         PermutedActorCritic(hidden_size=hidden_size, seed=seed, permutation=(0,1,2)))
        for variant in (explicit, identity):
            self.assertEqual(baseline.history, variant.history)
            self.assertEqual(baseline.evaluations, variant.evaluations)
            for key in baseline.model.state_dict():
                self.assertTrue(torch.equal(baseline.model.state_dict()[key], variant.model.state_dict()[key]), key)
        with self.assertRaises(TypeError):
            train(config, model_factory=object())
        with self.assertRaises(TypeError):
            train(config, model_factory=lambda **kwargs: object())

    def setUp(self):
        self.model = RecurrentActorCritic(hidden_size=8, seed=4)
        self.spec = RuleSpec(3, (RuleTerm('action_bonus', 1, 1),))

    def test_observe_only_event_and_replay_likelihood_without_boundary_reset(self):
        torch_before = torch.random.get_rng_state().clone()
        trial = collect_trial(self.model, self.spec, episodes_per_trial=2, max_steps=3, seed=71)
        self.assertEqual(len(trial.events), 7)  # T + episodes - 1
        self.assertEqual(trial.action_event_indices, (0,1,2,4,5,6))
        self.assertEqual(len(trial.actions), 6)
        self.assertEqual(len(trial.rewards), 6)
        self.assertEqual(len(trial.episodes), 2)
        self.assertEqual(trial.events[3], trial.episodes[0].transitions[-1].next_observations[0])
        self.assertFalse(trial.events[3].episode_start_flag)
        self.assertEqual(trial.events[4], trial.episodes[1].transitions[0].observations[0])
        self.assertTrue(trial.events[4].episode_start_flag)
        for episode_index, episode in enumerate(trial.episodes):
            for step, transition in enumerate(episode.transitions):
                action_idx = episode_index*3 + step
                event_idx = trial.action_event_indices[action_idx]
                self.assertEqual(trial.events[event_idx], transition.observations[0])
                self.assertEqual(trial.actions[action_idx], transition.actions[0])
                self.assertEqual(trial.rewards[action_idx], transition.rewards[0])
        log_probs, values, entropies = replay_trial(self.model, trial)
        torch.testing.assert_close(log_probs.detach(), torch.tensor(trial.old_log_probs), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(values.detach(), torch.tensor(trial.old_values), atol=1e-6, rtol=1e-6)
        self.assertEqual(entropies.shape, (6,))
        # Independent sequential replay explicitly feeds the observe-only event;
        # there is no zeroing or loss at the episode boundary.
        hidden = self.model.initial_hidden()
        manual_logs, manual_values = [], []
        event_to_action = dict(zip(trial.action_event_indices, trial.actions))
        for i, obs in enumerate(trial.events):
            logits, value, hidden = self.model.forward_observation(obs, hidden, n=3)
            if i in event_to_action:
                manual_logs.append(torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(event_to_action[i])))
                manual_values.append(value)
        torch.testing.assert_close(log_probs, torch.stack(manual_logs), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(values, torch.stack(manual_values), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_reset_ablation_replay_matches_its_collected_likelihood(self):
        trial = collect_trial(self.model, self.spec, episodes_per_trial=2, max_steps=3,
                              seed=71, reset_hidden=True)
        self.assertEqual(trial.action_event_indices, (0,1,2,4,5,6))
        self.assertEqual(trial.reset_event_indices, (0,4))
        log_probs, values, _ = replay_trial(self.model, trial)
        torch.testing.assert_close(log_probs.detach(), torch.tensor(trial.old_log_probs), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(values.detach(), torch.tensor(trial.old_values), atol=1e-6, rtol=1e-6)

    def test_late_action_loss_backpropagates_into_gru_recurrence(self):
        trial = collect_trial(self.model, self.spec, episodes_per_trial=2, max_steps=3, seed=43)
        cells = [module for module in self.model.modules() if isinstance(module, torch.nn.GRUCell)]
        self.assertTrue(cells)
        states = []
        def capture_state(_module, _inputs, output):
            output.retain_grad()
            states.append(output)
        handle = cells[0].register_forward_hook(capture_state)
        try:
            log_probs, _, _ = replay_trial(self.model, trial)
        finally:
            handle.remove()
        self.model.zero_grad(set_to_none=True)
        (-log_probs[-1]).backward()
        self.assertIsNotNone(cells[0].weight_hh.grad)
        self.assertGreater(cells[0].weight_hh.grad.abs().sum().item(), 0)
        self.assertEqual(len(states), 7)
        # Last action's gradient must reach both sides of the episode boundary,
        # including the observe-only event (index 3).
        for index in (0, 2, 3, 4):
            self.assertIsNotNone(states[index].grad, index)
            self.assertGreater(states[index].grad.abs().sum().item(), 0, index)

    def test_paired_random_and_recurrent_trials_have_same_exogenous_draws(self):
        recurrent = collect_trial(self.model, self.spec, episodes_per_trial=3, max_steps=3, seed=123)
        random_trial = collect_trial(None, self.spec, episodes_per_trial=3, max_steps=3, seed=123)
        self.assertEqual(recurrent.environment_seeds, random_trial.environment_seeds)
        self.assertEqual(recurrent.opponent_seeds, random_trial.opponent_seeds)
        for a,b in zip(recurrent.episodes,random_trial.episodes):
            self.assertEqual(a.transitions[0].observations[0].previous_own_action,
                             b.transitions[0].observations[0].previous_own_action)
            self.assertEqual(a.transitions[0].observations[1].previous_own_action,
                             b.transitions[0].observations[1].previous_own_action)
            self.assertEqual([t.actions[1] for t in a.transitions], [t.actions[1] for t in b.transitions])

    def test_evaluate_is_reproducible_and_never_changes_weights(self):
        specs = [self.spec, RuleSpec(3,(RuleTerm('action_bonus',1,2),))]
        before = {key: value.detach().clone() for key,value in self.model.state_dict().items()}
        result = evaluate(self.model, specs, episodes_per_trial=2, max_steps=3, seed=9)
        result_again = evaluate(self.model, specs, episodes_per_trial=2, max_steps=3, seed=9)
        self.assertEqual(result, result_again)
        self.assertEqual(result['trials'], 2)
        self.assertEqual(len(result['mean_reward_by_episode']), 2)
        self.assertEqual(len(result['mean_reward_by_turn']), 6)
        self.assertEqual(len(result['trial_returns']), 2)
        self.assertGreaterEqual(result['standard_error'], 0)
        seed_rng = random.Random(9)
        trials = [collect_trial(self.model, spec, episodes_per_trial=2, max_steps=3,
                                seed=seed_rng.getrandbits(63)) for spec in specs]
        expected_returns = [sum(trial.rewards) for trial in trials]
        self.assertEqual(result['trial_returns'], expected_returns)
        self.assertAlmostEqual(result['mean_reward_per_step'], sum(expected_returns)/12)
        for e in range(2):
            expected = sum(trial.episodes[e].total_rewards[0] for trial in trials) / 6
            self.assertAlmostEqual(result['mean_reward_by_episode'][e], expected)
        for step in range(6):
            expected = sum(trial.rewards[step] for trial in trials)/2
            self.assertAlmostEqual(result['mean_reward_by_turn'][step], expected)
        expected_se = statistics.stdev([r/6 for r in expected_returns])/math.sqrt(2)
        self.assertAlmostEqual(result['standard_error'], expected_se)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_two_update_training_changes_parameters_and_is_reproducible(self):
        config = TrainConfig(n=3, hidden_size=8, updates=2, trials_per_update=2,
                             episodes_per_trial=2, max_steps=3, epochs=1,
                             evaluation_trials=2, evaluation_interval=1, seed=123)
        result = train(config)
        self.assertEqual(len(result.history), 2)
        for row in result.history:
            for field in ('update','loss','entropy','value_loss','approx_kl','clip_fraction',
                          'mean_reward_per_step','gradient_norm'):
                self.assertIn(field, row)
        initial, trained = result.initial_model.state_dict(), result.model.state_dict()
        self.assertTrue(any(not torch.equal(initial[k], trained[k]) for k in initial))
        again = train(config)
        for key, value in result.model.state_dict().items():
            torch.testing.assert_close(value, again.model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(result.history, again.history)

    def test_fixed_target_and_default_task_stream(self):
        for seed in (0, 1, 7, 27, 100):
            expected_target = random.Random(seed).randrange(3)
            default = sample_task(3, 'action_bonus', seed)
            self.assertEqual(default, sample_task(3, 'action_bonus', seed, fixed_action=None))
            self.assertEqual(default.terms[0].action, expected_target)
            self.assertEqual(sample_task(3, 'mixed', seed), generate_rule(3, seed, 'mixed'))
            for action in range(3):
                self.assertEqual(sample_task(3, 'action_bonus', seed, fixed_action=action).terms[0].action, action)
        for action in (True, -1, 3):
            with self.subTest(action=action), self.assertRaises((TypeError, ValueError)):
                TrainConfig(fixed_action=action)
        with self.assertRaises((TypeError, ValueError)):
            TrainConfig(family='mixed', fixed_action=1)
        cfg = TrainConfig(fixed_action=2, hidden_size=4, updates=1, trials_per_update=1,
                          episodes_per_trial=2, max_steps=2, epochs=1, evaluation_trials=3)
        self.assertTrue(all(spec.terms[0].action == 2 for spec in evaluation_specs(cfg)))
        trial = collect_trial(RecurrentActorCritic(hidden_size=4, seed=2),
                              sample_task(3, 'action_bonus', 1, fixed_action=2),
                              episodes_per_trial=2, max_steps=2, seed=90)
        from dataclasses import fields
        self.assertEqual({field.name for field in fields(trial.events[0])},
                         {'previous_own_action','previous_opponent_action',
                          'previous_own_reward','episode_start_flag'})

    def test_passive_training_callback_does_not_change_training(self):
        config = TrainConfig(n=3, fixed_action=1, hidden_size=4, updates=2, trials_per_update=1,
                             episodes_per_trial=2, max_steps=2, epochs=1,
                             evaluation_trials=2, evaluation_interval=2, seed=43)
        captured_targets = []
        def track_collect(model, spec, **kwargs):
            captured_targets.append(spec.terms[0].action)
            return collect_trial(model, spec, **kwargs)
        with mock.patch('mamrl.training.collect_trial', side_effect=track_collect):
            baseline = train(config)
        self.assertGreaterEqual(len(captured_targets), config.updates * config.trials_per_update)
        self.assertTrue(all(target == 1 for target in captured_targets))
        snapshots = []
        def callback(update, model):
            snapshots.append((update, {k:v.detach().clone() for k,v in model.state_dict().items()}))
        observed = train(config, checkpoint_callback=callback)
        self.assertEqual([update for update,_ in snapshots], [0,1,2])
        self.assertEqual(baseline.history, observed.history)
        self.assertEqual(baseline.evaluations, observed.evaluations)
        for key in baseline.model.state_dict():
            self.assertTrue(torch.equal(baseline.model.state_dict()[key], observed.model.state_dict()[key]))
            self.assertTrue(torch.equal(snapshots[0][1][key], baseline.initial_model.state_dict()[key]))
            self.assertTrue(torch.equal(snapshots[-1][1][key], baseline.model.state_dict()[key]))

    def test_train_cli_artifacts_checkpoint_replay_and_variable_n(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'training'
            cmd = [sys.executable, '-m', 'mamrl.train', '--output-dir', str(output),
                   '--hidden-size', '4', '--updates', '1', '--trials-per-update', '1',
                   '--episodes-per-trial', '2', '--max-steps', '2', '--epochs', '1',
                   '--evaluation-trials', '2', '--evaluation-interval', '1', '--seed', '11']
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, result.stderr)
            printed = json.loads(result.stdout)
            self.assertEqual(printed['updates'], 1)
            expected_files = ('config.json', 'training.json', 'evaluation_trial.json',
                              'initial.pt', 'policy.pt', 'report.html',
                              'sample_episode/run.json', 'sample_episode/report.html')
            for name in expected_files:
                self.assertTrue((output / name).is_file(), name)
            snapshots = {name:(output/name).read_bytes() for name in expected_files}
            second = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(snapshots, {name:(output/name).read_bytes() for name in expected_files})

            record = json.loads((output/'training.json').read_text(encoding='utf-8'))
            specs = [RuleSpec(x['n'], tuple(RuleTerm(**term) for term in x['terms']))
                     for x in record['evaluation_task_specs']]
            initial = load_checkpoint(output/'initial.pt')
            trained = load_checkpoint(output/'policy.pt')
            options = dict(episodes_per_trial=2, max_steps=2, seed=record['evaluation_seed'])
            for label, model, reset in (('initial',initial,False), ('trained',trained,False),
                                         ('random',None,False), ('hidden_reset',trained,True)):
                replayed = evaluate(model, specs, reset_hidden=reset, **options)
                comparison = record['comparisons'][label]
                self.assertEqual(replayed['trial_returns'], comparison['trial_returns'])
                self.assertAlmostEqual(replayed['mean_reward_per_step'], comparison['mean_reward_per_step'])
                self.assertAlmostEqual(replayed['standard_error'], comparison['standard_error'])

            trace = json.loads((output/'evaluation_trial.json').read_text(encoding='utf-8'))
            self.assertEqual(trace['trial_seed'], random.Random(record['evaluation_seed']).getrandbits(63))
            replayed_trial = collect_trial(trained, specs[0], episodes_per_trial=2, max_steps=2,
                                          seed=trace['trial_seed'])
            self.assertEqual(trace['rollout']['actions'], list(replayed_trial.actions))
            self.assertEqual(trace['rollout']['rewards'], list(replayed_trial.rewards))
            self.assertEqual(trace['rollout']['action_event_indices'], [0,1,3,4])
            self.assertEqual(trace['rollout']['events'], [asdict(obs) for obs in replayed_trial.events])
            for recorded, replayed in zip(trace['rollout']['old_log_probs'], replayed_trial.old_log_probs):
                self.assertAlmostEqual(recorded, replayed, places=6)

            # The separately stored sample episode is reproducible from its metadata.
            sample = load_record(output/'sample_episode/run.json')
            self.assertEqual(sample['metadata']['policies'], ['RecurrentPolicy', 'RandomPolicy'])
            sample_spec = RuleSpec(sample['rule']['n'], tuple(RuleTerm(**term) for term in sample['rule']['terms']))
            replay = run_episode(MemoryOneEnv(compile_rule(sample_spec), max_steps=2),
                                 (RecurrentPolicy(trained), RandomPolicy()),
                                 seed=sample['metadata']['seed'],
                                 policy_seeds=tuple(sample['metadata']['policy_seeds']))
            self.assertEqual([list(t.actions) for t in replay.transitions],
                             [t['actions'] for t in sample['transitions']])
            self.assertEqual(list(replay.total_rewards), sample['summary']['total_rewards'])

            html = (output/'report.html').read_text(encoding='utf-8')
            self.assertNotRegex(html.lower(), r'<\s*script\b|@import\b|https?://|(?<![a-z])(?:nan|inf(?:inity)?)(?![a-z])')
            for label in ('initial','trained','random','hidden_reset'):
                self.assertIn(str(record['comparisons'][label]['mean_reward_per_step']), html)
                self.assertIn(str(record['comparisons'][label]['standard_error']), html)
            svgs = re.findall(r'<svg\b[^>]*>.*?</svg>', html, re.DOTALL | re.IGNORECASE)
            self.assertGreaterEqual(len(svgs), 2)
            for svg in svgs:
                parsed = ElementTree.fromstring(svg)
                self.assertTrue(any(node.tag.endswith('polyline') for node in parsed.iter()))
                self.assertNotRegex(svg.lower(), r'(?<![a-z])(?:nan|[+-]?inf(?:inity)?)(?![a-z])')

            # A checkpoint trained at n=3 can be executed with the same weights at n=5;
            # this verifies the interface, not learned cross-n generalization.
            expanded = Path(tmp) / 'n5'
            expanded_cmd = [sys.executable, '-m', 'mamrl', '--checkpoint', str(output/'policy.pt'),
                            '--n', '5', '--steps', '2', '--seed', '71', '--output-dir', str(expanded)]
            run = subprocess.run(expanded_cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(run.returncode, 0, run.stderr)
            n5 = load_record(expanded/'run.json')
            self.assertEqual(n5['metadata']['n'], 5)
            self.assertEqual(n5['metadata']['policies'], ['RecurrentPolicy','RandomPolicy'])
            self.assertEqual(n5['tensor']['shape'], [2,5,5,5,5])
            self.assertEqual(n5['summary']['steps'], 2)


if __name__ == '__main__':
    unittest.main()
