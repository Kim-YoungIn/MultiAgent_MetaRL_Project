"""Contract & Exact Rational Gate-0 Verification Tests for R0 / R1 / R3 Delayed-Response Game."""

from __future__ import annotations

from fractions import Fraction
import unittest
import numpy as np

from mamrl.envs.delayed_response import (
    ACT_P,
    ACT_P0,
    ACT_P1,
    ACT_S,
    ACT_X00,
    DelayedResponseEnv,
    N_ACTIONS,
    N_OPP_SYMBOLS,
    N_TYPES,
    VectorizedDelayedResponseEnv,
    build_opp_reward_table,
    build_response_table,
)
from mamrl.eval.exact_dp import ExactDelayedDP, run_gate0_audit


class TestGate0ExactRationalDP(unittest.TestCase):
    def test_r3_matches_independent_rational_dp_benchmarks(self) -> None:
        # Independently verify all exact numbers from the v2.2 independent review (§3)
        rep = run_gate0_audit("R3", horizon=24, audit_ks=(2, 4, 8))
        self.assertEqual(rep.myopic_openings, (ACT_S,))
        self.assertEqual(rep.optimal_openings, (ACT_P0, ACT_P1))
        self.assertEqual(rep.opening_q_values["S"], Fraction(16775, 1000))
        self.assertEqual(rep.opening_q_values["P"], Fraction(17450, 1000))
        self.assertEqual(rep.opening_q_values["P0"], Fraction(17575, 1000))
        self.assertEqual(rep.opening_q_values["P1"], Fraction(17575, 1000))
        for x_name in ("X00", "X01", "X10", "X11"):
            self.assertEqual(rep.opening_q_values[x_name], Fraction(15675, 1000))

        self.assertEqual(rep.opening_margin, Fraction(4, 5))  # 0.8
        self.assertTrue(rep.must_change_opening)
        self.assertEqual(rep.blind_value, Fraction(24, 5))    # 4.8
        self.assertEqual(rep.feedback_value, Fraction(12775, 1000))
        self.assertEqual(rep.open_loop_bounds[2], Fraction(17575, 1000))
        self.assertEqual(rep.open_loop_bounds[4], Fraction(169, 10))
        self.assertEqual(rep.open_loop_bounds[8], Fraction(137, 10))
        # K=2 has zero feedback headroom due to 1-turn delay; K=4 and K=8 have strictly positive headroom
        self.assertEqual(rep.feedback_conditioning_headroom[2], Fraction(0))
        self.assertGreater(rep.feedback_conditioning_headroom[4], Fraction(0))
        self.assertGreater(rep.feedback_conditioning_headroom[8], Fraction(0))

    def test_r3_horizon_sensitivity_exact_values(self) -> None:
        expected = {
            8: (Fraction(175, 100), Fraction(16, 10)),
            16: (Fraction(9575, 1000), Fraction(32, 10)),
            32: (Fraction(25575, 1000), Fraction(64, 10)),
        }
        for h, (v_opt, v_blind) in expected.items():
            rep = run_gate0_audit("R3", horizon=h, audit_ks=())
            self.assertEqual(rep.bayes_optimal_value, v_opt)
            self.assertEqual(rep.blind_value, v_blind)

    def test_r0_passive_requires_no_opening_change(self) -> None:
        rep = run_gate0_audit("R0", horizon=24, audit_ks=(1, 2, 4, 8))
        self.assertEqual(rep.myopic_openings, (ACT_S,))
        self.assertEqual(rep.optimal_openings, (ACT_S,))
        self.assertFalse(rep.must_change_opening)
        self.assertEqual(rep.opening_margin, Fraction(0))
        # Feedback still matters greatly (learning theta at turn 1 beats staying blind in S)
        self.assertGreater(rep.feedback_value, Fraction(15, 1))
        # Turn 1 S reveals theta immediately -> K=1 open-loop matches adaptive; delaying switch at K=2 costs 0.8
        self.assertEqual(rep.feedback_conditioning_headroom[1], Fraction(0))
        self.assertEqual(rep.feedback_conditioning_headroom[2], Fraction(4, 5))

    def test_r1_active_1step_requires_opening_change_without_k4_branching(self) -> None:
        rep = run_gate0_audit("R1", horizon=24, audit_ks=(2, 4, 8))
        self.assertEqual(rep.myopic_openings, (ACT_S,))
        self.assertEqual(rep.optimal_openings, (ACT_P,))
        self.assertTrue(rep.must_change_opening)
        self.assertGreater(rep.opening_margin, Fraction(0))
        # Single open-loop probe P at t=1 reveals all 4 types on t=2 -> K=2 open-loop equals adaptive!
        self.assertEqual(rep.feedback_conditioning_headroom[2], Fraction(0))
        # At K=4, R1 still beats R3's information bottleneck (check Bayes-optimal ordering: R0 > R1 > R3)
        r0 = run_gate0_audit("R0", horizon=24, audit_ks=())
        r3 = run_gate0_audit("R3", horizon=24, audit_ks=())
        self.assertGreater(r0.bayes_optimal_value, rep.bayes_optimal_value)
        self.assertGreater(rep.bayes_optimal_value, r3.bayes_optimal_value)


class TestEnvironmentInvariants(unittest.TestCase):
    def test_honest_response_strictly_dominant_for_opponent(self) -> None:
        for regime in ("R0", "R1", "R3"):
            f_table = build_response_table(regime)
            r_opp = build_opp_reward_table(regime)
            for th in range(N_TYPES):
                for q in range(N_ACTIONS):
                    honest_y = int(f_table[th, q])
                    for a in range(N_ACTIONS):
                        honest_val = r_opp[th, q, a, honest_y]
                        for y in range(N_OPP_SYMBOLS):
                            if y != honest_y:
                                self.assertGreater(honest_val, r_opp[th, q, a, y])

    def test_vectorized_matches_scalar_exactness(self) -> None:
        rng = np.random.default_rng(123)
        for regime in ("R0", "R1", "R3"):
            batch = 16
            thetas = np.arange(batch, dtype=np.int64) % N_TYPES
            vec = VectorizedDelayedResponseEnv(regime, batch_size=batch, horizon=24, seed=0)
            v_obs = vec.reset(theta=thetas)
            scalars = [DelayedResponseEnv(regime, int(th), horizon=24) for th in thetas]
            s_obs = [env.reset() for env in scalars]

            for t in range(24):
                acts = rng.integers(0, N_ACTIONS, size=batch, dtype=np.int64)
                v_obs, v_rf, v_ro, v_y = vec.step(acts)
                for i, env in enumerate(scalars):
                    obs_i, rf_i, ro_i, y_i = env.step(int(acts[i]))
                    self.assertEqual(int(v_y[i]), y_i)
                    self.assertAlmostEqual(float(v_rf[i]), rf_i, places=12)
                    self.assertAlmostEqual(float(v_ro[i]), ro_i, places=12)
                    self.assertEqual(int(v_obs["prev_self_action"][i]), obs_i.prev_self_action)
                    self.assertEqual(int(v_obs["prev_opp_action"][i]), obs_i.prev_opp_action)
                    self.assertEqual(bool(v_obs["is_start"][i]), obs_i.is_start)

    def test_layer1_bayes_audit_identifies_probe_vs_safe_prefix(self) -> None:
        dp = ExactDelayedDP("R3", horizon=24)
        # Theta = 3 = (b=1, c=1)
        # Safe prefix (S, S, S, S) observes y=0 four times -> still 4 types uncertain
        env_safe = DelayedResponseEnv("R3", theta=3, horizon=24)
        env_safe.reset()
        traj_safe = []
        r_safe = 0.0
        for _ in range(4):
            q_prev = env_safe.q
            _, rf, _, y = env_safe.step(ACT_S)
            traj_safe.append((q_prev, y))
            r_safe += rf
        audit_safe = dp.audit_prefix_layer1(traj_safe, r_safe, remaining_turns=20, q_current=env_safe.q)
        self.assertEqual(audit_safe["posterior_support_size"], 4)
        self.assertEqual(audit_safe["rule_identified"], 0.0)

        # Adaptive probe prefix: P -> obs(S)=0; S -> obs(P)=1 (b=1); P1 -> obs(S)=0; S -> obs(P1)=1 (c=1)
        env_probe = DelayedResponseEnv("R3", theta=3, horizon=24)
        env_probe.reset()
        traj_probe = []
        r_probe = 0.0
        for a in (ACT_P, ACT_S, ACT_P1, ACT_S):
            q_prev = env_probe.q
            _, rf, _, y = env_probe.step(a)
            traj_probe.append((q_prev, y))
            r_probe += rf
        audit_probe = dp.audit_prefix_layer1(traj_probe, r_probe, remaining_turns=20, q_current=env_probe.q)
        self.assertEqual(audit_probe["posterior_support_size"], 1)
        self.assertEqual(audit_probe["rule_identified"], 1.0)
        self.assertGreater(audit_probe["restored_commitment_value"], audit_safe["restored_commitment_value"] + 10.0)
        self.assertGreater(audit_probe["net_restored_adaptive_return"], audit_safe["net_restored_adaptive_return"])


if __name__ == "__main__":
    unittest.main()
