"""Unit tests for Exp 3A Reliability & Dynamics Stress-Test Sweep."""

from __future__ import annotations

import unittest
import numpy as np

from mamrl.agents.baselines_v3 import GreedyPolicyV3
from mamrl.study_exp2b import SymUCBPolicy
from mamrl.study_reliability_sweep import (
    ShiftedChannelBayesRefV3,
    rollout_coop_dynamics_shift,
    rollout_r3_obs_corruption,
)


class TestReliabilitySweepExp3A(unittest.TestCase):
    def test_r3_obs_corruption_rate_and_shifted_bayes_ref(self) -> None:
        ref0 = ShiftedChannelBayesRefV3(regime="R3", horizon=24, noise_d=0.0)
        m0, _ = rollout_r3_obs_corruption(ref0, noise_d=0.0, n_episodes=64, horizon=24, seed=7)
        self.assertEqual(m0["realized_corrupt_rate"], 0.0)
        self.assertGreater(m0["total_mean"], 17.0)

        ref3 = ShiftedChannelBayesRefV3(regime="R3", horizon=24, noise_d=0.3)
        m3, _ = rollout_r3_obs_corruption(ref3, noise_d=0.3, n_episodes=128, horizon=24, seed=7)
        self.assertGreater(m3["realized_corrupt_rate"], 0.22)
        self.assertLess(m3["realized_corrupt_rate"], 0.38)
        # Even at d=0.3 observation corruption, true payoff on X_theta is unchanged so shifted Bayes beats blind 4.8
        self.assertGreater(m3["total_mean"], 10.0)

        g_m, _ = rollout_r3_obs_corruption(GreedyPolicyV3(), noise_d=0.6, n_episodes=64, horizon=24, seed=7)
        self.assertGreater(g_m["realized_corrupt_rate"], 0.5)

    def test_coop_trap_dynamics_slip_reduces_welfare_monotonically(self) -> None:
        m0, _ = rollout_coop_dynamics_shift(SymUCBPolicy(), SymUCBPolicy(), slip_d=0.0, n_episodes=128, horizon=24, seed=19)
        m6, _ = rollout_coop_dynamics_shift(SymUCBPolicy(), SymUCBPolicy(), slip_d=0.6, n_episodes=128, horizon=24, seed=19)
        self.assertGreater(m0["welfare_mean"], m6["welfare_mean"])


if __name__ == "__main__":
    unittest.main()
