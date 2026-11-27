"""Unit tests for SPEC v3.0 Exp 2A: Online Learning Opponents & Restored vs Continuous Pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from mamrl.agents.baselines_v3 import ContextualUCBPolicyV3, GreedyPolicyV3, StructuralModelPolicyV3
from mamrl.agents.drqn_rl2 import DRQNConfig, train_drqn_rl2
from mamrl.agents.opponents_v3 import OnlineOpponentConfig, make_online_opponent
from mamrl.envs.delayed_response import ACT_S, ACT_X00, VectorizedDelayedResponseEnv
from mamrl.study_exp2a import rollout_online_protocol, run_study_exp2a


class TestOnlineOpponentsV3(unittest.TestCase):
    def test_q_learning_and_expected_sarsa_td_math(self) -> None:
        # Exact 1-step Q-learning TD verification
        cfg_ql = OnlineOpponentConfig(
            kind="q_learning", regime="R3", alpha=0.5, gamma=0.5, epsilon=0.0, warmup_steps=0
        )
        opp = make_online_opponent(cfg_ql, batch_size=2, seed=7)
        opp.reset(np.array([0, 3], dtype=np.int64), seed=7)
        opp.q_table[0, 1, :] = np.array([0.0, 2.0, 0.0, 0.0])  # max at q'=1 is 2.0
        opp.update(
            q_prev=np.array([0, 0]),
            y_act=np.array([0, 1]),
            r_opp=np.array([2.0, 0.0]),
            q_next=np.array([1, 0]),
        )
        # Episode 0: 0 + 0.5 * (2.0 + 0.5 * 2.0 - 0) = 1.5
        self.assertAlmostEqual(float(opp.q_table[0, 0, 0]), 1.5, places=10)
        self.assertAlmostEqual(float(opp.q_table[1, 0, 1]), 0.0, places=10)

        # Exact 1-step Expected SARSA TD verification
        cfg_es = OnlineOpponentConfig(
            kind="expected_sarsa", regime="R3", alpha=0.5, gamma=0.5, temperature=1.0, warmup_steps=0
        )
        opp_es = make_online_opponent(cfg_es, batch_size=1, seed=7)
        opp_es.reset(np.array([0], dtype=np.int64), seed=7)
        q_next_vec = np.array([1.0, 2.0, 0.0, 0.0])
        opp_es.q_table[0, 1, :] = q_next_vec
        ex = np.exp(q_next_vec - np.max(q_next_vec))
        probs = ex / np.sum(ex)
        expected_boot = float(np.sum(probs * q_next_vec))
        opp_es.update(
            q_prev=np.array([0]),
            y_act=np.array([0]),
            r_opp=np.array([2.0]),
            q_next=np.array([1]),
        )
        self.assertAlmostEqual(float(opp_es.q_table[0, 0, 0]), 0.5 * (2.0 + 0.5 * expected_boot), places=10)

    def test_snapshot_restore_exactness_and_partial_mask(self) -> None:
        cfg = OnlineOpponentConfig(kind="q_learning", regime="R3", alpha=0.4, gamma=0.5, epsilon=0.1, warmup_steps=32)
        opp = make_online_opponent(cfg, batch_size=8, seed=11)
        thetas = np.tile(np.arange(4, dtype=np.int64), 2)
        opp.reset(thetas, seed=11)
        snap = opp.snapshot()
        q0 = opp.q_table.copy()

        # Advance 4 turns
        for _ in range(4):
            q_prev = np.full(8, ACT_S, dtype=np.int64)
            y = opp.act(q_prev)
            opp.update(q_prev=q_prev, y_act=y, r_opp=np.full(8, 2.0), q_next=np.full(8, ACT_X00))

        self.assertGreater(float(np.mean(opp.q_l1_distance_from_init())), 0.0)

        # Partial mask restore: first 4 episodes restored to q0, last 4 unchanged
        q_before_mask = opp.q_table.copy()
        mask = np.array([True, True, True, True, False, False, False, False])
        opp.restore(snap=None, mask=mask)
        np.testing.assert_allclose(opp.q_table[:4], q0[:4])
        np.testing.assert_allclose(opp.q_table[4:], q_before_mask[4:])

        # Full snapshot restore matches exact subsequent actions
        opp.restore(snap=snap, mask=None)
        np.testing.assert_allclose(opp.q_table, q0)
        y_after_1 = opp.act(np.full(8, ACT_S, dtype=np.int64))
        opp.restore(snap=snap, mask=None)
        y_after_2 = opp.act(np.full(8, ACT_S, dtype=np.int64))
        np.testing.assert_array_equal(y_after_1, y_after_2)

    def test_restore_public_state_preserves_turn_k_feedback_and_sets_flag(self) -> None:
        env = VectorizedDelayedResponseEnv("R3", batch_size=4, horizon=12, seed=5)
        env.reset(theta=np.array([0, 1, 2, 3]))
        obs, rf, _ro, y = env.step(np.array([1, 2, 3, 4]))
        restored = env.restore_public_state(obs, mask=np.array([True, False, True, False]))
        np.testing.assert_array_equal(env.q, np.array([ACT_S, 2, ACT_S, 4]))
        np.testing.assert_array_equal(restored["prev_self_action"], np.array([ACT_S, 2, ACT_S, 4]))
        np.testing.assert_array_equal(restored["reset_flag"], np.array([True, False, True, False]))
        # Turn-K opponent action and reward must be preserved!
        np.testing.assert_array_equal(restored["prev_opp_action"], y)
        np.testing.assert_allclose(restored["prev_self_reward"], rf)

    def test_paired_continuous_and_restored_share_exact_prefix(self) -> None:
        opp_cfg = OnlineOpponentConfig(kind="expected_sarsa", regime="R3", warmup_steps=24)
        pol = StructuralModelPolicyV3(horizon=12).fit_offline("R3", n_episodes=64, seed=9)
        cont = rollout_online_protocol(pol, regime="R3", opp_cfg=opp_cfg, protocol="continuous", n_episodes=16, horizon=12, restore_k=4, seed=77)
        rest = rollout_online_protocol(pol, regime="R3", opp_cfg=opp_cfg, protocol="restored", n_episodes=16, horizon=12, restore_k=4, seed=77)
        # Prefix turns 0..3 must be bit-for-bit identical!
        np.testing.assert_array_equal(cont["actions"][:, :4], rest["actions"][:, :4])
        np.testing.assert_array_equal(cont["opp_actions"][:, :4], rest["opp_actions"][:, :4])
        np.testing.assert_allclose(cont["focal_rewards"][:, :4], rest["focal_rewards"][:, :4])

    def test_exp2a_runner_end_to_end_and_resume_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exp2a_smoke"
            cfg = {
                "study_id": "exp2a_smoke",
                "output_dir": str(out_dir),
                "regime": "R3",
                "methods": [
                    "mt_drqn_aux_yr",
                    "mt_drqn_aux_yr_masked",
                    "mt_structural_model",
                    "zs_structural_model",
                    "contextual_ucb",
                    "greedy",
                ],
                "seeds": [11, 12],
                "horizon": 12,
                "primary_k": 4,
                "n_eval_episodes": 16,
                "eval_seed_base": 3000,
                "train_reset_prob": 0.5,
                "train_opponent": {
                    "kind": "q_learning",
                    "alpha": 0.35,
                    "gamma": 0.50,
                    "epsilon": 0.08,
                    "warmup_steps": 24,
                    "warmup_epsilon": 1.0,
                },
                "eval_opponents": {
                    "q_learning_indist": {
                        "kind": "q_learning",
                        "alpha": 0.35,
                        "gamma": 0.50,
                        "epsilon": 0.08,
                        "warmup_steps": 24,
                        "warmup_epsilon": 1.0,
                    },
                    "expected_sarsa_unseen": {
                        "kind": "expected_sarsa",
                        "alpha": 0.25,
                        "gamma": 0.75,
                        "temperature": 0.45,
                        "warmup_steps": 20,
                        "warmup_temperature": 0.80,
                    },
                },
                "hidden_dim": 16,
                "drqn_batch_episodes": 8,
                "drqn_iters": 4,
                "drqn_warmup_iters": 2,
                "drqn_replay_capacity": 64,
                "drqn_grad_steps": 1,
                "drqn_minibatch": 8,
                "drqn_log_interval": 2,
                "structural_offline_episodes": 64,
                "n_bootstrap": 200,
                "delta_total": 0.30,
                "delta_post": 0.25,
            }
            s1 = run_study_exp2a(cfg, workers=1)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "tables" / "methods.csv").exists())
            self.assertTrue((out_dir / "tables" / "paired_contrasts.csv").exists())
            self.assertTrue((out_dir / "tables" / "restore_vs_continuous.csv").exists())

            # Resume run must match bit-for-bit
            s2 = run_study_exp2a(cfg, workers=1)
            for opp_name in ("q_learning_indist", "expected_sarsa_unseen"):
                for proto in ("continuous", "restored"):
                    for m in cfg["methods"]:
                        self.assertAlmostEqual(
                            s1["opponents"][opp_name]["protocols"][proto]["methods"][m]["total_mean"],
                            s2["opponents"][opp_name]["protocols"][proto]["methods"][m]["total_mean"],
                            places=12,
                        )


if __name__ == "__main__":
    unittest.main()
