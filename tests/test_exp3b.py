"""Unit tests for Exp 3B: Exact S_4 Permutation Equivariance, Semantic Game Suites & Runner Smoke."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from mamrl.agents.equivariant_drqn import (
    EquivariantRecurrentQNet,
    encode_equivariant_tokens_batch,
)
from mamrl.envs.semantic_games import SEM_N_ACTIONS, build_semantic_bank
from mamrl.study_exp3b import run_study_exp3b


class TestEquivariantSemanticExp3B(unittest.TestCase):
    def test_exact_s4_permutation_equivariance_across_all_24_permutations(self) -> None:
        """Verify Q-values, opp-action logits, and reward prediction are strictly S_4-equivariant/invariant."""
        torch.manual_seed(42)
        net = EquivariantRecurrentQNet(hidden_dim=16, n_actions=SEM_N_ACTIONS)
        net.eval()

        rng = np.random.default_rng(7)
        b, t_len = 4, 8
        a_self = rng.integers(0, SEM_N_ACTIONS, size=(b, t_len))
        a_opp = rng.integers(0, SEM_N_ACTIONS, size=(b, t_len))
        rews = rng.uniform(-1.0, 1.5, size=(b, t_len))
        act_query = rng.integers(0, SEM_N_ACTIONS, size=(b, t_len))

        def _build_seq(a_s: np.ndarray, a_o: np.ndarray) -> torch.Tensor:
            seq = np.zeros((b, t_len, SEM_N_ACTIONS, 6), dtype=np.float32)
            for t in range(t_len):
                obs = {
                    "prev_self_action": np.full(b, -1, dtype=np.int64) if t == 0 else a_s[:, t - 1],
                    "prev_opp_action": np.full(b, -1, dtype=np.int64) if t == 0 else a_o[:, t - 1],
                    "prev_self_reward": np.zeros(b, dtype=np.float64) if t == 0 else rews[:, t - 1],
                    "is_start": np.full(b, t == 0, dtype=np.bool_),
                }
                seq[:, t] = encode_equivariant_tokens_batch(obs, n_actions=SEM_N_ACTIONS, horizon=t_len, step_t=t)
            return torch.from_numpy(seq)

        with torch.no_grad():
            q_orig, h_orig = net(_build_seq(a_self, a_opp))
            opp_orig, r_orig = net.predict_aux(h_orig, torch.from_numpy(act_query))

            for perm in permutations(range(SEM_N_ACTIONS)):
                # perm maps old action index i -> new action index perm[i]
                p_arr = np.array(perm, dtype=np.int64)
                inv_p = np.argsort(p_arr)
                a_s_p = p_arr[a_self]
                a_o_p = p_arr[a_opp]
                aq_p = p_arr[act_query]

                q_perm, h_perm = net(_build_seq(a_s_p, a_o_p))
                opp_perm, r_perm = net.predict_aux(h_perm, torch.from_numpy(aq_p))

                # q_perm[:, :, new_idx] must equal q_orig[:, :, old_idx] where old_idx = inv_p[new_idx]
                np.testing.assert_allclose(q_perm.numpy(), q_orig.numpy()[:, :, inv_p], atol=1e-5)
                np.testing.assert_allclose(opp_perm.numpy(), opp_orig.numpy()[:, :, inv_p], atol=1e-5)
                np.testing.assert_allclose(r_perm.numpy(), r_orig.numpy(), atol=1e-5)

    def test_semantic_suites_strict_symmetry_and_zero_overlap(self) -> None:
        b_in = build_semantic_bank(n_sem_train=4, eval_suite="in_dist")
        b_up = build_semantic_bank(n_sem_train=4, eval_suite="unseen_perm")
        b_pa = build_semantic_bank(n_sem_train=4, eval_suite="unseen_params")
        b_fa = build_semantic_bank(n_sem_train=4, eval_suite="unseen_family")

        for bk in (b_in, b_up, b_pa, b_fa):
            np.testing.assert_allclose(bk["r2"], np.transpose(bk["r1"], (0, 2, 1)))

        # Zero matrix overlap between in_dist and every unseen suite!
        for suite_name, bk in (("unseen_perm", b_up), ("unseen_params", b_pa), ("unseen_family", b_fa)):
            for i in range(b_in["r1"].shape[0]):
                for j in range(bk["r1"].shape[0]):
                    self.assertFalse(
                        np.allclose(b_in["r1"][i], bk["r1"][j]),
                        f"Overlap between in_dist[{i}] and {suite_name}[{j}]",
                    )

    def test_exp3b_runner_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exp3b_smoke"
            cfg = {
                "study_id": "exp3b_smoke",
                "output_dir": str(out_dir),
                "seeds": [11, 12],
                "horizon": 12,
                "n_eval_episodes": 64,
                "eval_seed_base": 5000,
                "drqn_batch_episodes": 8,
                "drqn_replay_capacity": 64,
                "drqn_grad_steps": 1,
                "drqn_minibatch": 8,
                "drqn_log_interval": 2,
                "n_bootstrap": 100,
                "delta_return": 0.30,
                "conditions": {
                    "equiv_aux_yr_n4_b1": {"arch": "equivariant", "aux_mode": "yr", "n_sem_train": 4, "n_iters": 3, "warmup_iters": 1, "hidden_dim": 12},
                    "onehot_aux_yr_n4_b1": {"arch": "onehot", "aux_mode": "yr", "n_sem_train": 4, "n_iters": 3, "warmup_iters": 1, "hidden_dim": 16},
                },
            }
            s = run_study_exp3b(cfg, workers=1)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "tables" / "matchups.csv").exists())
            self.assertTrue((out_dir / "tables" / "paired_contrasts.csv").exists())
            self.assertEqual(set(s["suites"].keys()), {"in_dist", "unseen_perm", "unseen_params", "unseen_family"})


if __name__ == "__main__":
    unittest.main()
