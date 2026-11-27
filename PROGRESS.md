# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Exp 1 (`core_v3_sealed`), Exp 2A (`exp2a_online_opp`), and Exp 2B (`exp2b_symmetric_ma`) ALL COMPLETE & VERIFIED (27/27 unit tests pass, repo ~3.1 MB < 5 MB).** Ready for Priority Extension (**Auxiliary Task Discovery / 보조 목표 탐색**) and publication figures/report.

## Verified Results Summary (All 5 Sealed Seeds `1001..1005`, `n_eval=512`, `10k` Hierarchical Bootstrap)
- **Exp 1 (`R0/R1/R3`, Fixed Opponent):** `drqn_aux_yr` reached `+20.20 / +19.10 / +17.52` (99.7% Bayes-Opt; `L1_K4 commit=+17.00`), beating `masked` (`+1.57`, `SUPPORTED_POS`), `vanilla` (`+4.59`), `aux_r` (`+5.31`), `rl2_ppo` (`+4.80`), and online baselines, while tying `structural_model` (`+17.50`, `RESOLVED_NULL`). `aux_y` is the critical signal (`+16.80` vs `aux_r +12.22`).
- **Exp 2A (`R3` × Online `Q-learning` / Unseen `Expected SARSA`, Continuous vs. Restored):**
  - Zero-shot fixed-trained `zs_drqn_aux_yr` collapsed (`+4.41 / -7.02`), losing to smoothed `zs_structural_model` (`+8.74 / +1.74`, `SUPPORTED_NEG`).
  - Matched online training `mt_drqn_aux_yr` recovered to #1 on `Q-learning` (`+9.84`, `+5.43` over zero-shot, `SUPPORTED_POS`) via risk-aware fallback to `S` (`X_mismatch` 6.5× lower), and beat `mt_structural_model` under Restored (`+1.86`, `SUPPORTED_POS`; DiD `-0.92`, `SUPPORTED_NEG`).
  - Honest OOD boundary: on noisy unseen `Expected SARSA` (`59%` early honesty), all active probers lost to blind `prior_only` (`+4.80`; `-1.51`, `SUPPORTED_NEG`).
- **Exp 2B (Symmetric 2-Player 4-Action Games, Zero `PROBE` Buttons, Disjoint Train/Unseen Test Splits):**
  - **`coop_trap` (16 Train / 8 Unseen Rules):** On **unseen rules**, cross-seed **`Meta vs. Meta` (`meta_aux_yr × meta_aux_yr`)** reached **Welfare `+31.70` (`coopL=61.6%`)**, beating `Q×Q` (`+4.87`, `+26.83` `SUPPORTED_POS`), `UCB×UCB` (`+19.51`, `+12.19` `SUPPORTED_POS`), and pure-reward `meta_vanilla × meta_vanilla` (`+20.70`, `+11.00` `SUPPORTED_POS`). In `Meta vs. Non-Meta`, Meta bore unilateral probing cost (`margin < 0`) while lifting joint welfare (`Q-learner` `+6.22`, `Contextual-UCB` `+15.91`, `SUPPORTED_POS`).
  - **`comp_rps` (4 Train / 4 Flipped-Orientation Unseen Rules):** In-dist Meta exploited stateless `UCB` (`+3.41` margin, 76.5% win) but greedy `meta_aux_yr` lost to stateful `Q-learner` (`-0.98`, `SUPPORTED_NEG`); on **unseen flipped orientations**, all Meta agents collapsed (`-0.04` vs UCB, `-0.99` vs Q-learner) due to spurious `Anchor→orientation` shortcut memorization in the narrow 4-rule training set, while tabula-rasa Non-Meta learners had zero generalization gap.

## Next Actionable Step
1. **Priority Extension (`보조 목표 탐색` — Auxiliary Task Discovery):** Meta-learn auxiliary prediction weights/gates over candidate public-observation targets instead of hand-picking `aux_y/aux_r`, then generate publication figures & draft the 5-page NeurIPS report + 5-min video/quiz (deadlines Dec 17 / Dec 20).
