# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Exp 1 (`results/core_v3_sealed/`, 210/210 fits) & Exp 2A (`results/exp2a_online_opp/`, 90/90 units) COMPLETE & VERIFIED (24/24 unit tests pass).** Next: Exp 2B (Symmetric 4-Action Competition & Cooperation Matchups).

## Verified Exp 1 Results (`R0/R1/R3`, Fixed Reactive Opponent, 5 sealed seeds `1001..1005`)
- **`drqn_aux_yr`:** `+20.200 ± 0.00` / `+19.100 ± 0.00` / `+17.520 ± 0.12` (99.7% Bayes-Opt; `L1_K4 commit=+17.000`, `L3_gap=0.000`). Beats masked (`+1.570`, `SUPPORTED_POS`), `drqn_vanilla` (`+4.590`, `SUPPORTED_POS`), `drqn_aux_r` (`+5.305`), `rl2_ppo` (`+4.800`), and online UCB/Greedy/Random; ties `structural_model` (`+17.500`, `RESOLVED_NULL`).
- **Aux Signal Finding:** `aux_y` (opponent-action prediction) unlocks 100% exploration info (`+16.795`); `aux_r` alone does not (`+12.215`).

## Verified Exp 2A Results (`R3` × Online `Q-learning` / Unseen `Expected SARSA`, 5 sealed seeds, Continuous vs. Restored)
- **Zero-Shot Brittleness:** Fixed-trained `zs_drqn_aux_yr` collapsed under online opponents (`+4.407` on Q-learning, `-7.018` on unseen Expected SARSA; `zs_bayes_optimal = +4.443 / -9.644`), losing to smoothed `zs_structural_model` (`+8.737 / +1.739`, `SUPPORTED_NEG`).
- **Matched Online Meta-Training (`mt_drqn_aux_yr`):** Rescued performance on `q_learning_indist` (`+9.837 ± 0.83`, `+5.430` over zero-shot, `SUPPORTED_POS`) by learning risk-aware fallback to `S` (`18.4%` vs `4.8%`, cutting `X_mismatch` 6.5×). Beat `masked` (`+1.497`, `SUPPORTED_POS`), `prior_only` (`+5.037`), `vanilla` (`+2.818`), and online UCB/Greedy/Random, with mutualistic opponent return (`50.32` vs `43.21`, `both_up=75.8%`).
- **Restored vs. Continuous (`K=4`):** `mt_drqn_aux_yr` retained value after reset (`+9.837 → +9.345`, `-0.492` `INCONCLUSIVE`) whereas `mt_structural_model` dropped significantly (`+8.895 → +7.486`, `-1.409` `SUPPORTED_POS`), yielding `mt_drqn_aux_yr > mt_structural_model` on Restored (`+1.859`, `SUPPORTED_POS`; DiD `-0.917`, `SUPPORTED_NEG`).
- **Honest OOD Noise Boundary (`expected_sarsa_unseen`, early honesty 59%):** Matched training prevented collapse (`-7.018 → +3.294`, beating UCB/Greedy and Restored `mt_structural_model`), **but ALL active probers lost to blind safe play `prior_only` (`+4.800`; `-1.506`, `SUPPORTED_NEG`)** when unseen early opponent noise exceeded the meta-trained risk threshold.

## Next Actionable Step
1. Implement & run **Exp 2B** (Symmetric 4-Action Competition & Cooperation games: `Non-Meta vs. Non-Meta`, `Meta vs. Non-Meta`, `Meta vs. Meta` on strictly held-out unseen rules).
2. Keep **Auxiliary Task Discovery (`보조 목표 탐색`)** as the top-priority extension after Exp 2B.
