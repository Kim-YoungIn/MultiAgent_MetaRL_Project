"""Unit & Contract Tests for SPEC v3.0 Pipeline (PPO RL², Masking, Baselines, Audit, Receipts)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np

from mamrl.agents.baselines_v3 import (
    BayesOptimalPolicyV3,
    ClairvoyantPolicyV3,
    StructuralModelPolicyV3,
)
from mamrl.agents.drqn_rl2 import RecurrentQNet, compute_aux_losses
from mamrl.agents.ppo_rl2 import encode_obs_batch
from mamrl.envs.delayed_response import N_ACTIONS, N_OPP_SYMBOLS
from mamrl.eval.audit_v3 import compute_episode_audit, hierarchical_paired_bootstrap, rollout_policy_v3
from mamrl.eval.exact_dp import run_gate0_audit
from mamrl.study_v3 import run_study_v3


class TestMaskingAndCeilings(unittest.TestCase):
    def test_masking_and_prior_only_strip_feedback_strictly(self) -> None:
        obs = {
            "prev_self_action": np.array([2, 3], dtype=np.int64),
            "prev_opp_action": np.array([1, 3], dtype=np.int64),
            "prev_self_reward": np.array([1.0, -2.0], dtype=np.float64),
            "is_start": np.array([False, False]),
            "reset_flag": np.array([False, False]),
            "turn": np.array([2, 2], dtype=np.int64),
        }
        fb_slice = slice(N_ACTIONS, N_ACTIONS + N_OPP_SYMBOLS + 1)

        # rl2 sees full feedback at t=2
        x_full = encode_obs_batch(obs, horizon=24, step_t=2, variant="rl2", mask_k=4)
        self.assertGreater(float(np.abs(x_full[:, fb_slice]).sum()), 0.0)

        # rl2_masked hides ALL feedback from turns 1..4 (so step_t=0..4 are strictly masked; step_t=5 reveals turn-5 feedback)
        for t_masked in (0, 1, 2, 3, 4):
            x_m_before = encode_obs_batch(obs, horizon=24, step_t=t_masked, variant="rl2_masked", mask_k=4)
            self.assertEqual(float(np.abs(x_m_before[:, fb_slice]).sum()), 0.0, f"Leak at step_t={t_masked}")
        x_m_after = encode_obs_batch(obs, horizon=24, step_t=5, variant="rl2_masked", mask_k=4)
        self.assertGreater(float(np.abs(x_m_after[:, fb_slice]).sum()), 0.0)

        # prior_only hides feedback at ALL turns
        for t in (0, 2, 4, 20):
            x_p = encode_obs_batch(obs, horizon=24, step_t=t, variant="prior_only", mask_k=4)
            self.assertEqual(float(np.abs(x_p[:, fb_slice]).sum()), 0.0)

    def test_drqn_aux_losses_respect_mode_and_masking(self) -> None:
        import torch

        qnet = RecurrentQNet(hidden_dim=16)
        hs = torch.randn(4, 12, 16)
        ax = torch.randint(0, N_ACTIONS, (4, 12))
        yx = torch.randint(0, N_OPP_SYMBOLS, (4, 12))
        rx = torch.randn(4, 12)

        y0, r0 = compute_aux_losses(qnet, hs, ax, yx, rx, variant="rl2", aux_mode="none", mask_k=4)
        self.assertEqual(float(y0.item()), 0.0)
        self.assertEqual(float(r0.item()), 0.0)

        yy, ry = compute_aux_losses(qnet, hs, ax, yx, rx, variant="rl2", aux_mode="y", mask_k=4)
        self.assertGreater(float(yy.item()), 0.0)
        self.assertEqual(float(ry.item()), 0.0)

        yr_y, yr_r = compute_aux_losses(qnet, hs, ax, yx, rx, variant="rl2", aux_mode="r", mask_k=4)
        self.assertEqual(float(yr_y.item()), 0.0)
        self.assertGreater(float(yr_r.item()), 0.0)

        # prior_only disables aux losses even if aux_mode='yr'
        yp, rp = compute_aux_losses(qnet, hs, ax, yx, rx, variant="prior_only", aux_mode="yr", mask_k=4)
        self.assertEqual(float(yp.item()), 0.0)
        self.assertEqual(float(rp.item()), 0.0)

        # rl2_masked ignores targets at t < mask_k completely
        yx_mod = yx.clone()
        yx_mod[:, :4] = (yx_mod[:, :4] + 1) % N_OPP_SYMBOLS
        rx_mod = rx.clone()
        rx_mod[:, :4] += 100.0
        ym1, rm1 = compute_aux_losses(qnet, hs, ax, yx, rx, variant="rl2_masked", aux_mode="yr", mask_k=4)
        ym2, rm2 = compute_aux_losses(qnet, hs, ax, yx_mod, rx_mod, variant="rl2_masked", aux_mode="yr", mask_k=4)
        self.assertAlmostEqual(float(ym1.item()), float(ym2.item()), places=7)
        self.assertAlmostEqual(float(rm1.item()), float(rm2.item()), places=7)

    def test_bayes_optimal_and_clairvoyant_reproduce_exact_gate0_values(self) -> None:
        for reg in ("R0", "R1", "R3"):
            gate0 = run_gate0_audit(reg, horizon=24, audit_ks=(2, 4, 8))
            b_pol = BayesOptimalPolicyV3(regime=reg, horizon=24)
            b_bun = rollout_policy_v3(b_pol, regime=reg, n_episodes=64, horizon=24, seed=7)
            b_aud = compute_episode_audit(b_bun, audit_ks=(2, 4, 8), primary_k=4)
            self.assertAlmostEqual(b_aud["end_to_end"]["total_mean"], float(gate0.bayes_optimal_value), places=9)
            self.assertAlmostEqual(b_aud["layer3"]["live_continuation_gap_k4_mean"], 0.0, places=9)

            c_pol = ClairvoyantPolicyV3(regime=reg, horizon=24)
            c_bun = rollout_policy_v3(c_pol, regime=reg, n_episodes=64, horizon=24, seed=7)
            c_aud = compute_episode_audit(c_bun, audit_ks=(2, 4, 8), primary_k=4)
            self.assertAlmostEqual(c_aud["end_to_end"]["total_mean"], float(gate0.clairvoyant_value), places=9)

    def test_structural_model_learns_from_unlabeled_rollouts(self) -> None:
        pol = StructuralModelPolicyV3(horizon=24).fit_offline("R1", n_episodes=512, seed=42)
        self.assertEqual(pol.offline_train_transitions, 512 * 24)
        bun = rollout_policy_v3(pol, regime="R1", n_episodes=64, horizon=24, seed=9)
        aud = compute_episode_audit(bun)
        # Should strongly beat blind all-S floor (4.8) without ever seeing theta labels
        self.assertGreater(aud["end_to_end"]["total_mean"], 12.0)


class TestBootstrapAndRunnerSmoke(unittest.TestCase):
    def test_hierarchical_bootstrap_verdicts_and_ties(self) -> None:
        a = np.ones((5, 32)) * 2.0
        b = np.ones((5, 32)) * 1.0
        res_pos = hierarchical_paired_bootstrap(a, b, practical_threshold=0.25, n_boot=500, seed=0)
        self.assertEqual(res_pos["verdict"], "SUPPORTED_POS")
        self.assertEqual(res_pos["win_rate"], 1.0)
        self.assertEqual(res_pos["tie_rate"], 0.0)

        res_null = hierarchical_paired_bootstrap(a, a, practical_threshold=0.25, n_boot=500, seed=0)
        self.assertEqual(res_null["verdict"], "RESOLVED_NULL")
        self.assertEqual(res_null["tie_rate"], 1.0)

    def test_study_v3_end_to_end_and_atomic_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "study_id": "smoke_v3",
                "output_dir": tmpdir,
                "regimes": ["R3"],
                "methods": ["rl2", "drqn_aux_yr", "drqn_vanilla", "drqn_aux_yr_masked", "contextual_ucb", "bayes_optimal"],
                "seeds": [1, 2],
                "horizon": 12,
                "primary_k": 4,
                "audit_ks": [2, 4],
                "n_eval_episodes": 16,
                "hidden_dim": 16,
                "batch_episodes": 16,
                "total_transitions": 384,
                "ppo_epochs": 2,
                "minibatches": 2,
                "drqn_batch_episodes": 8,
                "drqn_iters": 4,
                "drqn_warmup_iters": 2,
                "drqn_grad_steps": 2,
                "drqn_minibatch": 8,
                "n_bootstrap": 200,
                "device": "cpu",
            }
            s1 = run_study_v3(cfg)
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "tables" / "behavior_repetition.csv").exists())
            self.assertTrue((Path(tmpdir) / "fits" / "R3_rl2_seed1" / "train_curve.csv").exists())
            self.assertTrue((Path(tmpdir) / "fits" / "R3_rl2_seed1" / "eval_episodes.npz").exists())

            # Second call resumes cleanly from atomic receipts and matches exact numbers
            s2 = run_study_v3(cfg)
            self.assertAlmostEqual(
                s1["regimes"]["R3"]["methods"]["rl2"]["total_mean"],
                s2["regimes"]["R3"]["methods"]["rl2"]["total_mean"],
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
