"""Unit tests for SPEC v3.0 Exp 2B: Symmetric 2-Player 4-Action Competition & Cooperation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from mamrl.envs.symmetric_ma import (
    SYM_N_ACTIONS,
    VectorizedSymmetricGameEnv,
    build_symmetric_rule_bank,
)
from mamrl.study_exp2b import (
    SymQLearningPolicy,
    SymUCBPolicy,
    encode_sym_obs_batch,
    rollout_symmetric_matchup,
    run_study_exp2b,
)


class TestSymmetricGamesExp2B(unittest.TestCase):
    def test_strict_symmetry_zero_overlap_and_role_balance(self) -> None:
        for game in ("comp_rps", "coop_trap"):
            bank = build_symmetric_rule_bank(game)
            tr_r1, te_r1 = bank["train"]["r1"], bank["test"]["r1"]
            tr_r2, te_r2 = bank["train"]["r2"], bank["test"]["r2"]

            # Strict symmetry: R2[k, a1, a2] == R1[k, a2, a1]
            np.testing.assert_allclose(tr_r2, np.transpose(tr_r1, (0, 2, 1)))
            np.testing.assert_allclose(te_r2, np.transpose(te_r1, (0, 2, 1)))

            # Zero-sum check for comp_rps
            if game == "comp_rps":
                np.testing.assert_allclose(tr_r1 + tr_r2, 0.0)
                np.testing.assert_allclose(te_r1 + te_r2, 0.0)
                self.assertEqual(tr_r1.shape[0], 4)
                self.assertEqual(te_r1.shape[0], 4)
                # Balanced Anchor counts across actions 0..3
                np.testing.assert_array_equal(np.bincount(bank["train"]["anchor"], minlength=4), [1, 1, 1, 1])
                np.testing.assert_array_equal(np.bincount(bank["test"]["anchor"], minlength=4), [1, 1, 1, 1])
            else:
                self.assertEqual(tr_r1.shape[0], 16)
                self.assertEqual(te_r1.shape[0], 8)
                for role in ("safe", "coop", "defect", "trap"):
                    np.testing.assert_array_equal(np.bincount(bank["train"][role], minlength=4), [4, 4, 4, 4])
                    np.testing.assert_array_equal(np.bincount(bank["test"][role], minlength=4), [2, 2, 2, 2])

            # Strictly disjoint payoff matrices (zero overlap between train and unseen test!)
            for i in range(tr_r1.shape[0]):
                for j in range(te_r1.shape[0]):
                    self.assertFalse(np.allclose(tr_r1[i], te_r1[j]), f"Overlap found in {game}: train[{i}] == test[{j}]")

    def test_masking_strips_early_feedback_strictly(self) -> None:
        env = VectorizedSymmetricGameEnv("coop_trap", split="train", batch_size=4, horizon=12, seed=1)
        obs1, _obs2 = env.reset()
        obs1, _obs2, _r1, _r2 = env.step(np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0]))
        x_masked = encode_sym_obs_batch(obs1, horizon=12, step_t=1, variant="rl2_masked", mask_k=4)
        # Opp action one-hot (4..7) and reward (8) must be strictly zero for t < 4
        np.testing.assert_allclose(x_masked[:, SYM_N_ACTIONS : 2 * SYM_N_ACTIONS + 1], 0.0)
        x_open = encode_sym_obs_batch(obs1, horizon=12, step_t=4, variant="rl2_masked", mask_k=4)
        self.assertGreater(float(np.sum(np.abs(x_open[:, SYM_N_ACTIONS : 2 * SYM_N_ACTIONS]))), 0.0)

    def test_exp2b_runner_end_to_end_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exp2b_smoke"
            cfg = {
                "study_id": "exp2b_smoke",
                "output_dir": str(out_dir),
                "games": ["comp_rps", "coop_trap"],
                "meta_methods": ["meta_drqn_aux_yr", "meta_drqn_vanilla", "meta_drqn_masked"],
                "seeds": [11, 12],
                "horizon": 12,
                "primary_k": 4,
                "n_eval_episodes": 16,
                "eval_seed_base": 4000,
                "hidden_dim": 16,
                "drqn_batch_episodes": 8,
                "drqn_iters": 4,
                "drqn_warmup_iters": 2,
                "drqn_replay_capacity": 64,
                "drqn_grad_steps": 1,
                "drqn_minibatch": 8,
                "drqn_log_interval": 2,
                "n_bootstrap": 200,
                "delta_return": 0.30,
            }
            s1 = run_study_exp2b(cfg, workers=1)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "tables" / "matchups.csv").exists())
            self.assertTrue((out_dir / "tables" / "paired_contrasts.csv").exists())
            # Zero-sum invariant check on comp_rps: welfare == 0.0 identically
            for sp in ("in_dist", "unseen_rules"):
                for md in s1["games"]["comp_rps"]["splits"][sp]["matchups"].values():
                    self.assertAlmostEqual(md["welfare_mean"], 0.0, places=9)
            # Resume determinism check
            s2 = run_study_exp2b(cfg, workers=1)
            self.assertAlmostEqual(
                s1["games"]["coop_trap"]["splits"]["unseen_rules"]["matchups"]["meta_aux_yr_vs_meta_aux_yr_xseed"]["welfare_mean"],
                s2["games"]["coop_trap"]["splits"]["unseen_rules"]["matchups"]["meta_aux_yr_vs_meta_aux_yr_xseed"]["welfare_mean"],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
