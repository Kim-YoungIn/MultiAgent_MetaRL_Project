"""Separate exploration and assessment lengths preserve learning/control boundaries."""
import statistics
import unittest
from mamrl.meta_exploration_rollout import run_meta_trial
from mamrl.meta_exploration import RepeatExplorer, summarize_meta
from mamrl.relational_rules import social_rule


class HorizonContract(unittest.TestCase):
    def test_distinct_horizons_cost_updates_and_reset_bootstrap(self):
        trial=run_meta_trial(social_rule(),RepeatExplorer,seed=91,steps=2,assessment_steps=5)
        self.assertEqual(trial['phase_steps'],{'exploration':2,'assessment':5,'erased_assessment':5})
        self.assertEqual(trial['environment_steps_with_control'],12)
        self.assertEqual(trial['student_final']['updates'],7)
        self.assertEqual(trial['opponent_final']['updates'],7)
        self.assertEqual(trial['exploration'][-1]['learning_next_observations'],trial['assessment'][0]['observations'])
        self.assertFalse(trial['exploration'][-1]['terminal'])
        self.assertEqual(trial['assessment'][0]['observations'],trial['erased_assessment'][0]['observations'])
        self.assertEqual(trial['opponent_before_assessment'],trial['erased_opponent_before_assessment'])
        self.assertTrue(trial['assessment'][-1]['terminal'])
        expected=statistics.mean(e['rewards'][0] for e in trial['exploration']+trial['assessment'])
        self.assertAlmostEqual(trial['gameplay_mean'],expected)
        result=summarize_meta([dict(trial,task_id='one')])
        self.assertAlmostEqual(result['total_gameplay_mean'],expected)
        self.assertEqual(len(result['assessment_curve']),5)
        self.assertAlmostEqual(result['opponent_nonuniform_action_fraction'],trial['opponent_nonuniform_action_events']/7)

    def test_unspecified_assessment_is_legacy_equal_horizon(self):
        rule=social_rule()
        self.assertEqual(run_meta_trial(rule,RepeatExplorer,seed=7,steps=3),
                         run_meta_trial(rule,RepeatExplorer,seed=7,steps=3,assessment_steps=3))
        with self.assertRaises(ValueError):run_meta_trial(rule,RepeatExplorer,seed=7,assessment_steps=0)
