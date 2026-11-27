# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Exp 1, Exp 2A, Exp 2B, Exp 3A (Reliability/Dynamics Sweep), and Exp 3B (S4-Equivariant Anti-Memorization + Semantic Diversity / D-031) ALL SEALED (5 seeds `1001..1005`, 35/35 tests pass).**

## Verified Results Summary (5 Sealed Seeds, `n_eval=512`, `10k` Hierarchical Bootstrap)
- **Exp 1 (`R0/R1/R3`, Fixed Opponent):** `drqn_aux_yr` `+20.20 / +19.10 / +17.52` (99.7% Bayes-Opt; beats strict `masked +2.345`, `vanilla +4.590`, `aux_r +5.305`, online baselines; ties `structural_model +17.50` at 28× fewer steps).
- **Exp 2A (`R3` × Online `Q-learning` / Unseen `Expected SARSA`):** Zero-shot `zs_drqn_aux_yr` collapsed (`+4.41 / -7.02`), losing to `zs_structural_model` (`+8.74 / +1.74`). Matched `mt_drqn_aux_yr` led `Q-learning` (`+9.84`, `+1.75` vs strict `masked`; beat `mt_structural_model` on Restored `+1.86`, DiD `-0.92`), but lost to `prior_only` (`+4.80`) on noisy unseen `Expected SARSA` (`+3.29`, `-1.51` `SUPPORTED_NEG`).
- **Exp 2B (Symmetric Single-Game 4-Action):** Unseen `coop_trap` cross-seed `meta_aux_yr` Welfare `+31.70` beat `UCB×UCB` (`+12.19`) and `meta_vanilla` (`+11.00`, `SUPPORTED_POS`); `comp_rps` exposed competitive exploitability (`-0.98` vs Q) and unseen-orientation collapse.
- **Exp 3A (Legacy `dynamics_transfer` Bridge, $d \in \{0, 0.3, 0.6\}$, Eval-Only):** On `R3` obs corruption, online-trained `mt_drqn_aux_yr` beat fixed-trained by `+9.95 / +15.73` (`SUPPORTED_POS`) and flipped over `structural_model` at $d=0.6$ (`+4.66`), though `prior_only` won at $d=0.6$. On `coop_trap` slip $d$, `meta_aux_yr − ucb×ucb` reversed monotonically: **`+12.32` ($d=0$) → `+2.47` ($d=0.3$) → `-3.85` ($d=0.6$, `SUPPORTED_NEG`)**.
- **Exp 3B (Anti-Memorization `EquivariantDRQN` + 4 Semantic Families + Unseen 5th Family + D-031):**
  - **Equivariance Breakthrough:** On 4 semantic families, One-Hot `DRQN` lost to `UCB×UCB` on EVERY suite (`+10.12 / +9.46 / +8.10 / +17.10` vs `+11.42 / +11.45 / +11.87 / +22.69`), even at 2× budget. Exact $S_4$ **`EquivariantDRQN`** surged **`+12.63 / +13.61 / +12.68` (`SUPPORTED_POS`)** over One-Hot (`+22.75 / +23.08 / +20.78`; `b2: +27.17`) with **zero permutation gap**, beating `UCB×UCB` by `+11.63 / +8.91` (`SUPPORTED_POS`) and hitting **`+30.71` (`+8.02` over `UCB×UCB`, `+15.04` over One-Hot, `SUPPORTED_POS`)** on the brand-new unseen 5th family (`vs_ucb`).
  - **Equivariance vs Aux:** Under exact $S_4$ equivariance, pure-reward `equiv_vanilla` matched `equiv_aux_yr` (`INCONCLUSIVE` across all 4 suites).
  - **D-031 Decomposed:** Broad `n4` beat narrow `n2` on diverse `unseen_params` (`+3.83`, `SUPPORTED_POS`); matching exposure `n4_b2 − n4_b1` recovered `+4.09..+6.30` (`SUPPORTED_POS`); on coordination-aligned 5th family, narrow `n2` still beat `n4_b2` (`-5.37` `SUPPORTED_NEG`, prior-alignment effect).

## Next Actionable Step
1. Optional final mechanism study: **Auxiliary Task Discovery (`보조 목표 탐색`)** (with `TASK-001` falsification gates) or **Online Surprise-Calibrated Exploration** (for Exp 3A's $d=0.6$ OOD boundary), then Codex audit & publication figures/report.
