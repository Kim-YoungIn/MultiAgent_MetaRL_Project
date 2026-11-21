"""Independent, hand-checkable D-031 experiment acceptance contracts."""
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class ExperimentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import experiment
        from mamrl.diverse_catalog import build_diverse_manifest
        manifest = build_diverse_manifest(n=3, seed=104, train_rules=2,
                                          fresh_rules=2, structural_rules=2)
        cls.entries = [row for row in manifest['rules'] if row['split'] == 'train']

    def test_budget_and_pairing(self):
        from experiment.protocol import budget, plan
        from experiment.evaluation import trial_seed
        cells = plan()
        self.assertEqual(len(cells), 175)
        self.assertEqual(sum(len(c['methods']) for c in cells), 375)
        primary = [c for c in cells if c['condition'] == 'primary']
        self.assertEqual(len(primary), 25)
        self.assertEqual({c['seed'] for c in primary},
                         {41000+100*c+j for c in range(1, 6) for j in range(1, 6)})
        self.assertEqual(budget(False)['updates']*budget(False)['trials'] *
                         (budget(False)['k']+2*budget(False)['h']), 1843200)
        self.assertEqual(trial_seed(3, 'fresh_rules', 'r', 'steady', 0),
                         0x5bf51dc3408d5d59)

    def test_nonuniform_trapezoidal_auc(self):
        from experiment.evaluation import adaptation_auc
        ks = [0, 3, 6, 12, 24]
        # R(k) = 2 + k/12. The centered integral is the area of a
        # triangle of base 24 and height 2, divided by 24.
        curve = [{'k': k, 'return': 2+k/12} for k in ks]
        self.assertAlmostEqual(adaptation_auc(curve), 1.0)
        for point in curve:
            point['return'] += 7
        self.assertAlmostEqual(adaptation_auc(curve), 1.0)

    def test_observation_mask_and_initial_observation(self):
        from experiment.model import ContextActor
        from mamrl.environment import Observation
        model = ContextActor(seed=4, masked=True)
        hidden = model.initial_hidden(n=3)
        a = Observation(1, 0, 0.8, False)
        b = Observation(1, 2, -0.9, False)
        la, va, ha = model.forward_observation(a, hidden, n=3)
        lb, vb, hb = model.forward_observation(b, hidden, n=3)
        self.assertTrue(torch.equal(la, lb))
        self.assertTrue(torch.equal(va, vb))
        self.assertTrue(torch.equal(ha, hb))
        # The initial observation includes the opponent's action.
        a = Observation(1, 0, 0, True)
        b = Observation(1, 2, 0, True)
        la, _, _ = model.forward_observation(a, hidden, n=3)
        lb, _, _ = model.forward_observation(b, hidden, n=3)
        self.assertFalse(torch.equal(la, lb))

    def test_empty_trial_is_identical_for_all_teachers(self):
        from experiment.evaluation import trial_row
        from mamrl.meta_exploration import RandomExplorer, BalancedExplorer, decode_rule
        from mamrl.rules import compile_rule
        tensor = compile_rule(decode_rule(self.entries[0]['rule']))
        for profile in ('steady', 'sarsa', 'sarsa_heldout'):
            random_result = trial_row(tensor, RandomExplorer, profile, 7139, 0, 7)
            balanced_result = trial_row(tensor, BalancedExplorer, profile, 7139, 0, 7)
            self.assertEqual(random_result, balanced_result)
            self.assertEqual(random_result['G'], 0)
            self.assertEqual(random_result['steps_with_control'], 14)

    def test_handoff_bootstraps_from_new_episode(self):
        from mamrl.meta_exploration import RandomExplorer, decode_rule
        from mamrl.meta_exploration_rollout import run_meta_trial
        rule = decode_rule(self.entries[0]['rule'])
        for k in (1, 3, 6):
            trial = run_meta_trial(rule, RandomExplorer, seed=319, steps=k,
                                   assessment_steps=5, opponent_profile='sarsa')
            last = trial['exploration'][-1]
            first = trial['assessment'][0]
            self.assertEqual(last['learning_next_observations'], first['observations'])
            self.assertTrue(first['observations'][0]['episode_start_flag'])
            self.assertFalse(last['terminal'])
            self.assertEqual(len(trial['assessment']), 5)
            self.assertEqual(len(trial['erased_assessment']), 5)
            self.assertEqual(trial['environment_steps_with_control'], k+10)

    def test_p_target_and_m_o_target(self):
        from experiment.training import train
        with tempfile.TemporaryDirectory() as tmp:
            for method in ('P', 'M', 'O'):
                train(Path(tmp)/method, self.entries, method=method, seed=44,
                      profiles=('steady', 'sarsa'), updates=1, trials=3, k=3, h=3)
                state = torch.load(Path(tmp)/method/'training_state.pt', weights_only=True)
                record = state['history'][0]
                field = 'exploration_reward' if method == 'P' else 'G'
                self.assertAlmostEqual(record['target'], record[field], places=6)
                self.assertEqual(sum(record['rule_exposures'].values()), 3)
                self.assertEqual(sum(record['opponent_exposures'].values()), 3)

    def test_raw_and_reduced_corruption_is_detected(self):
        from experiment.artifacts import evaluate_to_disk, verify_evaluation
        from experiment.evaluation import evaluate
        from mamrl.meta_exploration import RandomExplorer
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = evaluate(self.entries, RandomExplorer, reps=1, fixed_budget=True,
                            ks=(0, 1, 3), h=3)
            evaluate_to_disk(directory, 'random', rows, 1)
            verify_evaluation(directory, 'random')
            path = directory/'random.json.gz'
            path.write_bytes(path.read_bytes() + b'corruption')
            with self.assertRaises(ValueError):
                verify_evaluation(directory, 'random')


if __name__ == '__main__':
    unittest.main()
