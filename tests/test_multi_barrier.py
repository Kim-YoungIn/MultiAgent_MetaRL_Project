"""Unit & Exact Gate-0 Rational DP Tests for 3-Layer Multi-Barrier Game (Exp 4)."""

from __future__ import annotations

import unittest
import numpy as np

from mamrl.envs.multi_barrier import (
    MB_ACT_P1,
    MB_ACT_S,
    MB_N_ACTIONS,
    MB_N_TYPES,
    ExactMultiBarrierDP,
    VectorizedMultiBarrierEnv,
    build_mb_opp_reward_table,
    build_mb_response_table,
    run_mb_gate0_report,
)


class TestMultiBarrierGate0(unittest.TestCase):
    def test_opponent_honest_response_strictly_dominant(self) -> None:
        f_tbl = build_mb_response_table()
        r_opp = build_mb_opp_reward_table()
        for th in range(MB_N_TYPES):
            for q in range(MB_N_ACTIONS):
                hy = int(f_tbl[th, q])
                for a in range(MB_N_ACTIONS):
                    for y in range(3):
                        if y != hy:
                            self.assertGreater(r_opp[th, q, a, hy], r_opp[th, q, a, y])

    def test_exact_gate0_proves_horizon_dependent_barrier_progression(self) -> None:
        rep = run_mb_gate0_report(horizons=(6, 10, 24, 64))
        # Ultra-short horizon T=6: optimal stops at Tier 0 (all-S)
        self.assertEqual(rep["T=6"]["optimal_barrier_tier"], 0)
        # Short horizon T=10: optimal climbs to Tier 2, NOT Tier 3 (-10.0 barrier too costly!)
        self.assertEqual(rep["T=10"]["optimal_barrier_tier"], 2)
        # Long horizon T=64: optimal climbs ALL 3 nested barriers with massive headroom over Tier 2 & Tier 1!
        self.assertEqual(rep["T=64"]["optimal_barrier_tier"], 3)
        self.assertGreater(rep["T=64"]["tier3_over_tier2_gain"], 40.0)
        self.assertGreater(rep["T=64"]["tier2_over_tier1_gain"], 30.0)
        self.assertGreater(rep["T=64"]["opening_margin_over_S"], 1.5)

    def test_vectorized_env_reproduces_exact_bayes_optimal_at_t64(self) -> None:
        solver = ExactMultiBarrierDP(max_tier=3)
        env = VectorizedMultiBarrierEnv(batch_size=8, horizon=64, seed=0)
        thetas = np.arange(MB_N_TYPES, dtype=np.int64)
        obs = env.reset(theta=thetas)
        masks = np.full(8, (1 << MB_N_TYPES) - 1, dtype=np.int64)
        f_tbl = build_mb_response_table()
        totals = np.zeros(8, dtype=np.float64)
        prev_q = np.full(8, MB_ACT_S, dtype=np.int64)

        for t in range(64):
            rem = 64 - t
            acts = np.array([solver.optimal_action(rem, int(obs["prev_self_action"][i]), int(masks[i])) for i in range(8)])
            prev_q = obs["prev_self_action"].copy()
            obs, rf, _ro, y = env.step(acts)
            totals += rf
            for i in range(8):
                new_m = 0
                for th in range(MB_N_TYPES):
                    if ((int(masks[i]) >> th) & 1) and int(f_tbl[th, int(prev_q[i])]) == int(y[i]):
                        new_m |= 1 << th
                masks[i] = new_m

        self.assertAlmostEqual(float(np.mean(totals)), float(solver.value(64, MB_ACT_S, 255)), places=9)

    def test_exp4_masking_and_aux_discovery_smoke(self) -> None:
        import tempfile
        from pathlib import Path
        from mamrl.study_exp4_multibarrier import encode_mb_obs_batch, run_study_exp4

        obs = {
            "prev_self_action": np.array([1, 2], dtype=np.int64),
            "prev_opp_action": np.array([1, 2], dtype=np.int64),
            "prev_self_reward": np.array([-1.0, 2.2], dtype=np.float64),
            "is_start": np.array([False, False]),
            "turn": np.array([4, 4], dtype=np.int64),
        }
        fb = slice(MB_N_ACTIONS, MB_N_ACTIONS + 4)
        # Strict step_t > mask_k: t=0..8 masked, t=9 open
        for tm in (0, 4, 8):
            xm = encode_mb_obs_batch(obs, horizon=64, step_t=tm, variant="rl2_masked", mask_k=8)
            self.assertEqual(float(np.abs(xm[:, fb]).sum()), 0.0)
        xo = encode_mb_obs_batch(obs, horizon=64, step_t=9, variant="rl2_masked", mask_k=8)
        self.assertGreater(float(np.abs(xo[:, fb]).sum()), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exp4_smoke"
            cfg = {
                "study_id": "exp4_smoke",
                "output_dir": str(out_dir),
                "horizon": 16,
                "mask_k": 4,
                "seeds": [11],
                "n_eval_episodes": 16,
                "eval_seed_base": 6000,
                "hidden_dim": 16,
                "batch_episodes": 8,
                "n_iters": 4,
                "warmup_iters": 2,
                "replay_capacity": 64,
                "grad_steps_per_iter": 1,
                "minibatch_episodes": 8,
                "log_interval": 2,
                "n_bootstrap": 100,
                "delta_return": 0.30,
                "methods": ["drqn_aux_discovery", "drqn_aux_uniform", "drqn_vanilla", "arm_ucb"],
            }
            s = run_study_exp4(cfg, workers=1)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "tables" / "methods.csv").exists())
            gates = s["methods"]["drqn_aux_discovery"]["final_gates_mean"]
            self.assertAlmostEqual(sum(gates.values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
