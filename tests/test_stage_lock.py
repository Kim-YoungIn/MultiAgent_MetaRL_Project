"""Unit tests for Compact 6-Action 3-Layer Sequential Stage-Lock Environment & Gate-0 Exact DP."""

from __future__ import annotations

import unittest
import numpy as np

from mamrl.envs.stage_lock import (
    SL_ACT_P,
    SL_ACT_S,
    SL_FULL_KEY_MASK,
    SL_N_ACTIONS,
    SL_N_KEYS,
    SL_N_STAGES,
    ExactStageLockDP,
    VectorizedStageLockEnv,
    run_sl_gate0_report,
)


class TestStageLockEnv(unittest.TestCase):
    def test_gate0_horizon_dependent_optimal_depth(self) -> None:
        report = run_sl_gate0_report((4, 8, 16, 24, 48))
        # Exact horizon-dependent barrier climbing: Tier 0 -> Tier 1 -> Tier 2 -> Tier 3
        self.assertEqual(report["T=4"]["optimal_barrier_tier"], 0)
        self.assertEqual(report["T=8"]["optimal_barrier_tier"], 1)
        self.assertEqual(report["T=16"]["optimal_barrier_tier"], 2)
        self.assertEqual(report["T=48"]["optimal_barrier_tier"], 3)
        self.assertEqual(report["T=48"]["optimal_opening"], "P")
        self.assertTrue(report["T=48"]["must_change_opening"])
        self.assertGreater(report["T=48"]["opening_margin_over_S"], 0.0)
        self.assertGreater(report["T=48"]["tier3_over_tier2_gain"], 20.0)
        self.assertGreater(report["T=48"]["tier2_over_tier1_gain"], 20.0)

    def test_exact_dp_matches_vectorized_simulator_over_all_64_key_tuples(self) -> None:
        horizon = 48
        dp = ExactStageLockDP(max_tier=3)
        expected_v = float(dp.value(horizon, 0, 0, SL_ACT_S, SL_FULL_KEY_MASK))

        all_keys = np.array(
            [[k0, k1, k2] for k0 in range(SL_N_KEYS) for k1 in range(SL_N_KEYS) for k2 in range(SL_N_KEYS)],
            dtype=np.int64,
        )
        b = all_keys.shape[0]
        self.assertEqual(b, 64)

        env = VectorizedStageLockEnv(batch_size=b, horizon=horizon, seed=0)
        obs = env.reset(keys=all_keys)
        masks = np.full(b, SL_FULL_KEY_MASK, dtype=np.int64)
        stages = np.zeros(b, dtype=np.int64)
        unlocked = np.zeros(b, dtype=np.int64)
        tot_r = np.zeros(b, dtype=np.float64)

        for t in range(horizon):
            q_prev = obs["prev_self_action"].copy()
            acts = np.array(
                [dp.optimal_action(horizon - t, int(stages[i]), int(unlocked[i]), int(q_prev[i]), int(masks[i])) for i in range(b)],
                dtype=np.int64,
            )
            obs, rf, _ro, y = env.step(acts)
            tot_r += rf
            for i in range(b):
                ai, qi, yi = int(acts[i]), int(q_prev[i]), int(y[i])
                # Public belief update
                new_m = 0
                for k in range(SL_N_KEYS):
                    if ((int(masks[i]) >> k) & 1) and dp._f_table[k][qi] == yi:
                        new_m |= 1 << k
                if ai == SL_ACT_P and unlocked[i] == 1 and stages[i] < SL_N_STAGES - 1:
                    stages[i] += 1
                    unlocked[i] = 0
                    masks[i] = SL_FULL_KEY_MASK
                else:
                    if ai >= 2 and qi == ai and yi == 1:
                        unlocked[i] = 1
                    masks[i] = new_m

        self.assertAlmostEqual(float(np.mean(tot_r)), expected_v, places=9)
        self.assertTrue(np.all(env.stage == 2))

    def test_zero_passive_leakage_on_safe_action(self) -> None:
        env = VectorizedStageLockEnv(batch_size=64, horizon=10, seed=1)
        env.reset()
        for _ in range(5):
            obs, rf, _ro, y = env.step(np.full(64, SL_ACT_S, dtype=np.int64))
            self.assertTrue(np.all(y == 0))
            self.assertTrue(np.allclose(rf, 0.2))

    def test_strict_masking_boundary_and_pbvi_exact_d0(self) -> None:
        from mamrl.study_exp4b_compact import encode_public_obs
        from mamrl.study_diagnostics_v3 import CorruptedR3PBVISolver

        obs = {
            "prev_self_action": np.array([1, 2], dtype=np.int64),
            "prev_opp_action": np.array([3, 1], dtype=np.int64),
            "prev_self_reward": np.array([1.5, -0.6], dtype=np.float64),
            "is_start": np.array([False, False], dtype=np.bool_),
        }
        # At step_t == mask_k (8), feedback must STILL be strictly masked (step_t > mask_k)
        x_at_k = encode_public_obs(obs, n_actions=6, n_symbols=5, r_scale=2.5, horizon=48, step_t=8, variant="rl2_masked", mask_k=8)
        self.assertTrue(np.all(x_at_k[:, 6:12] == 0.0))
        x_after_k = encode_public_obs(obs, n_actions=6, n_symbols=5, r_scale=2.5, horizon=48, step_t=9, variant="rl2_masked", mask_k=8)
        self.assertGreater(float(np.sum(np.abs(x_after_k[:, 6:12]))), 0.0)

        # CorruptedR3PBVISolver at d=0.0 must match exact rational 17.575
        solver = CorruptedR3PBVISolver(noise_d=0.0, horizon=24, n_beliefs=64, seed=0)
        v0 = solver.value(24, 0, np.full(4, 0.25, dtype=np.float64))
        self.assertAlmostEqual(v0, 17.575, places=6)

    def test_recurrent_double_dqn_targets_preserve_turn0_history(self) -> None:
        import torch
        from mamrl.study_exp4b_compact import AuxDiscoveryQNet, compute_recurrent_double_dqn_targets

        torch.manual_seed(0)
        qnet = AuxDiscoveryQNet(obs_dim=14, hidden_dim=16, n_actions=6, n_symbols=5, r_scale=2.5)
        tnet = AuxDiscoveryQNet(obs_dim=14, hidden_dim=16, n_actions=6, n_symbols=5, r_scale=2.5)
        tx = torch.randn(4, 8, 14)
        tr = torch.randn(4, 8)

        q_all, _ = qnet(tx)
        targets = compute_recurrent_double_dqn_targets(q_all, tnet, tx, tr, gamma=0.99)
        # Verify exact match with step-by-step full-history unroll of tnet
        with torch.no_grad():
            h = torch.zeros(1, 4, 16)
            step_qs = []
            for t in range(8):
                qt, h = tnet.step(tx[:, t : t + 1, :], h)
                step_qs.append(qt)
            q_target_step = torch.stack(step_qs, dim=1)
            next_a = torch.argmax(q_all[:, 1:, :], dim=-1, keepdim=True)
            expected = tr.clone()
            expected[:, :-1] = tr[:, :-1] + 0.99 * q_target_step[:, 1:, :].gather(-1, next_a).squeeze(-1)
        self.assertTrue(torch.allclose(targets, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
