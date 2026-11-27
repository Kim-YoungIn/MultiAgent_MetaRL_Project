"""Core contract & invariant tests for the refactored v2 MAMRL package."""

from __future__ import annotations

import unittest

import numpy as np

from mamrl.agents.explorers import GreedyExplorer, OracleExplorer, RandomExplorer, UCBExplorer
from mamrl.agents.opponents import QLearningOpponent
from mamrl.envs.active_probe_game import ActiveProbeGame, generate_task_suite
from mamrl.eval.oracle import audit_task_suite
from mamrl.eval.protocol import evaluate_suite
from mamrl.runner import run_experiment


class TestGate0Oracle(unittest.TestCase):
    def test_active_mode_strictly_requires_opening_change(self) -> None:
        for div in ("low", "medium", "high"):
            tasks = generate_task_suite(24, mode="active", diversity=div, seed=7)
            audit = audit_task_suite(tasks)
            self.assertEqual(audit["positive_voi_rate"], 1.0)
            self.assertEqual(audit["must_change_opening_rate"], 1.0)
            self.assertGreater(audit["mean_loss_keeping_masked_opening"], 0.10)

    def test_passive_mode_has_voi_without_changing_opening(self) -> None:
        for div in ("low", "medium", "high"):
            tasks = generate_task_suite(24, mode="passive", diversity=div, seed=7)
            audit = audit_task_suite(tasks)
            self.assertEqual(audit["positive_voi_rate"], 1.0)
            self.assertEqual(audit["must_change_opening_rate"], 0.0)
            self.assertAlmostEqual(audit["mean_loss_keeping_masked_opening"], 0.0, places=9)


class TestEnvironmentAndControls(unittest.TestCase):
    def test_observation_determinism_and_no_leakage(self) -> None:
        task = generate_task_suite(2, mode="active", diversity="high", seed=1)[0]
        env1 = ActiveProbeGame(task, seed=123)
        env2 = ActiveProbeGame(task, seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        self.assertEqual(obs1, obs2)
        self.assertTrue(obs1.is_start)
        self.assertEqual(obs1.prev_self_action, -1)

        r1 = env1.step(0, 1)
        r2 = env2.step(0, 1)
        self.assertEqual(r1, r2)
        self.assertFalse(r1.focal_obs.is_start)

    def test_opponent_snapshot_restore_exactness(self) -> None:
        opp = QLearningOpponent(n_actions=4, seed=99)
        task = generate_task_suite(1, mode="active", diversity="low", seed=0)[0]
        env = ActiveProbeGame(task, seed=0)
        _, obs_o = env.reset(seed=0)

        snap = opp.snapshot()
        actions_before = []
        for t in range(5):
            a_o = opp.act(obs_o)
            actions_before.append(a_o)
            res = env.step(t % 4, a_o)
            opp.update(obs_o, a_o, res.opp_reward, res.opp_obs)
            obs_o = res.opp_obs

        opp.restore(snap)
        _, obs_o = env.reset(seed=0)
        actions_after = []
        for t in range(5):
            a_o = opp.act(obs_o)
            actions_after.append(a_o)
            res = env.step(t % 4, a_o)
            opp.update(obs_o, a_o, res.opp_reward, res.opp_obs)
            obs_o = res.opp_obs

        self.assertEqual(actions_before, actions_after)

    def test_oracle_beats_greedy_in_active_mode(self) -> None:
        tasks = generate_task_suite(16, mode="active", diversity="high", seed=5)
        res = evaluate_suite(
            tasks,
            {
                "greedy": lambda _s: GreedyExplorer(),
                "random": lambda _s: RandomExplorer(n_actions=4),
                "ucb": lambda _s: UCBExplorer(beta=0.5),
                "oracle": lambda s: OracleExplorer(s),
            },
            opponent_kind="q_learning",
            student_kind="bayesian",
            explore_steps=4,
            total_horizon=24,
            restore_opponent=False,
            seed=10,
        )
        oracle_mean = res["methods"]["oracle"]["total_mean"]
        greedy_mean = res["methods"]["greedy"]["total_mean"]
        self.assertGreater(oracle_mean, greedy_mean + 5.0)
        self.assertEqual(res["paired_contrasts"]["oracle_minus_greedy"]["win_rate"], 1.0)


class TestRunnerSmoke(unittest.TestCase):
    def test_end_to_end_runner_smoke(self) -> None:
        cfg = {
            "name": "smoke_active",
            "mode": "active",
            "diversity": "low",
            "n_train_tasks": 8,
            "n_test_tasks": 8,
            "opponent_kind": "reactive",
            "student_kind": "bayesian",
            "explore_steps": 2,
            "total_horizon": 12,
            "updates": 4,
            "batch_size": 4,
            "seed": 3,
        }
        out = run_experiment(cfg)
        self.assertEqual(out["gate0_oracle_audit"]["must_change_opening_rate"], 1.0)
        self.assertIn("meta_full_minus_greedy", out["continuous_deployment"]["paired_contrasts"])
        self.assertIn("meta_full_minus_meta_masked", out["controlled_info_only"]["paired_contrasts"])


if __name__ == "__main__":
    unittest.main()
