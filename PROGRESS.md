# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Dev Pilot Complete & Major Mechanistic Breakthrough Verified (18/18 tests OK)**. Ready to integrate `Aux-DRQN` (Full / Masked / Prior-Only) into `study_v3.py`, run 5-seed Sealed Core (Exp 1), and proceed to Exp 2A/2B.

## Completed Work (Verified)
- Finalized `docs/SPEC.md` (v3.0 §7 logging/statistics + Exp 2B $d$-bit held-out compositional rules) and implemented `ppo_rl2.py`, `baselines_v3.py`, `audit_v3.py`, `study_v3.py`, and `tests/test_v3_pipeline.py` (18/18 unit tests pass in `6.2 s`; exact float64 resume invariance verified).
- **Dev Pilot (`runs/dev_pilot_v3`) + Representation Diagnostic Results (`T=24`):**
  - **Online non-meta baselines** (`Random`, `Contextual UCB`, `Arm-UCB`, `Greedy`): `-23.4` to `+0.06` across `R0/R1/R3` (catastrophic `-2.0` penalties under unguided exploration).
  - **Vanilla reward-only Meta-RL** (`RL²-PPO` at `2^18` & `2^20` w/ entropy annealing, and vanilla `DRQN`): collapsed to safe local optimum **`+4.800`** (`S-S-S-S`, `rep = 1.00/1.00`) across `R0/R1/R3` because rare `+1.0` exploitation (`0.39%` of random steps) cannot train hidden belief representations before policy collapse.
  - **Self-Supervised Predictive Meta-RL (`Aux-DRQN`: unprivileged GRU-64 + next `(y, r)` prediction head):** broke the barrier completely — **R0 `+20.200`** (100% Bayes-Opt), **R1 `+18.800`** (98.4% Bayes-Opt, `probe=1.00`), **R3 `+17.575`** (**100.0% exact Bayes-Optimal!**, `med=+17.300`, `L1_K4=+17.000`, `L3_gap=0.000`, `probe=1.00`, `tv=1.00`, `rep(E/X)=0.08/0.97`, autonomously discovering the exact asymmetric `P1 → P0 → {X11 | P→X01 | P→S}` tree in `46 s` CPU).
  - **Model-Based Structural Meta-RL (`structural_model`: unlabeled EM + belief lookahead):** **R0 `+20.200`**, **R1 `+19.100`**, **R3 `+17.513`** (99.6% Bayes-Opt) using 10–25× fewer offline transitions (`24,576`).

## Next Actionable Step
1. Add clean `Aux-DRQN` (`aux_drqn`, `aux_drqn_masked`, `aux_drqn_prior_only`) to `src/mamrl/` + unit tests, lock 5-seed Sealed Core config (`configs/core_v3_sealed.json`), and run Exp 1.
2. Extend to Exp 2A (`R3` × online Q-learning / unseen Expected SARSA opponents) and Exp 2B ($d$-bit held-out compositional rules & Hamming distance).
