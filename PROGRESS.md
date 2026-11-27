# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Exp 1 (Sealed 5-Seed Core, `R0/R1/R3`, 210/210 fits) COMPLETE & VERIFIED (`results/core_v3_sealed/`, repo `2.4 MB`)**. Ready for Exp 2A (Online Q-Learning / SARSA Opponents) and Exp 2B (Symmetric Competition & Cooperation Matchups).

## Verified Exp 1 Results (`results/core_v3_sealed/`, 5 sealed seeds `1001..1005`, `n_eval=512`, `10k` bootstrap)
- **R0 (`Opt=20.20`) / R1 (`Opt=19.10`) / R3 (`Opt=17.575`):**
  - `drqn_aux_yr` (unprivileged GRU-64 + self-supervised `y,r`): **`+20.200 ± 0.00` / `+19.100 ± 0.00` / `+17.520 ± 0.12`** (99.7% Bayes-Opt; R3 `L1_K4 commit=+17.000`, `L3_gap=0.000`, `probe=1.00`, `tv=1.00`, `rep(E/X)=0.07/0.97`).
  - **Feedback Conditioning (`R3: drqn_aux_yr − masked`):** Total `+1.570 [+1.443, +1.678]` (`SUPPORTED_POS`, 100% win), Layer-1 commitment `+10.400 [+9.598, +11.725]` (`SUPPORTED_POS`), net cost-inclusive `+0.840 [+0.717, +0.955]` (`SUPPORTED_POS`).
  - **Auxiliary Signal Ablation (`R3`):** `drqn_vanilla` (pure reward) reached `+12.930 ± 4.32` (med `+16.900`, `probe=1.00`) — breaking the `all-S` trap where on-policy `rl2_ppo` collapsed (`+4.800 ± 0.00`), but with high seed variance. `drqn_aux_r` (`+12.215 ± 3.71`) did not help; **`drqn_aux_y` (`+16.795 ± 1.47`, `L1_K4=+17.000`)** unlocked 100% exploration info; combining `y+r` (`+17.520 ± 0.12`) cut seed SD 36× (`SUPPORTED_POS` vs `vanilla`: `+4.590 [+1.298, +7.863]`).
  - **Model-Based & Online Baselines (`R3`):** `structural_model` matched `drqn_aux_yr` (`+17.500 ± 0.07`, `RESOLVED_NULL`); `contextual_ucb` (`-20.73`), `arm_ucb` (`-7.30`), `greedy` (`-0.13`), `random` (`-23.22`) all lost decisively (`SUPPORTED_POS` across all 4 metrics).

## Next Actionable Step
1. Implement & run **Exp 2A** (`R3` × online Q-learning / unseen Expected SARSA opponents, Restored vs. Continuous) and **Exp 2B** (Symmetric 4-action Competition & Cooperation games: `Non-Meta vs. Non-Meta`, `Meta vs. Non-Meta`, `Meta vs. Meta`).
2. Keep **Auxiliary Task Discovery** (`보조 목표 탐색`) as the top-priority extension after Exp 2A/2B.
