# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **All 6 Sealed 5-Seed Studies COMPLETE (Exp 1, 2A, 2B, 3A, 3B, 4; 39/39 unit tests pass; repo ~3.9 MB / git zip < 1 MB << 5 MB).** Ready for final Codex cross-audit & publication figures/deliverables (or targeted intrinsic-bonus follow-up on Exp 4).

## Verified 5-Seed Sealed Evidence Map
- **Exp 1 (`R0/R1/R3`, Fixed Opponent):** `drqn_aux_yr` reached `99.7%` Bayes-Opt (`+17.52 / 17.575`), beating strict `masked` (`+2.345`), `vanilla` (`+4.590`), `aux_r` (`+5.305`), and `rl2_ppo` (`+4.80`), while tying 28×-cheaper `structural_model` (`+17.50`, `RESOLVED_NULL`).
- **Exp 2A & 3A (Online Opponents + $d \in \{0,0.3,0.6\}$ Stress Sweep):** Fixed-trained `zs_drqn_aux_yr` collapsed under non-stationarity (`+4.41/-7.02`); online-matched `mt_drqn_aux_yr` led `Q-learning` (`+9.84`, Restored `>+1.86` over `structural_model`, `d=0.6` `>+4.66` over `structural_model`), yet lost to blind `prior_only (+4.80)` on noisy unseen `Expected SARSA` and reversed vs `UCB×UCB` on `coop_trap` slip (`+12.32 → +2.47 → -3.85`).
- **Exp 2B & 3B (Symmetric Games + Anti-Memorization `EquivariantDRQN` + Semantic Diversity / D-031):** On 4 semantic families, One-Hot `DRQN` lost to `UCB×UCB` everywhere even at 2× budget; exact $S_4$ **`EquivariantDRQN`** jumped **`+12.63/+13.61/+12.68` (`SUPPORTED_POS`)** over One-Hot with zero permutation gap, reached `+30.71` (`+8.02` over `UCB×UCB`) on an unseen 5th family (`vs_ucb`), needed no `aux_yr` (`INCONCLUSIVE` vs `equiv_vanilla`), and decomposed D-031 into exposure dilution (`+4.09..+6.30`) + prior alignment.
- **Exp 4 (3-Layer Multi-Barrier `T=64`, 22 actions, Exact Gate-0 `T0=12.8 / T1=41.8 / T2=75.6 / T3=120.6` + Auxiliary Task Discovery):** Honest scaling wall + bootstrap catch-22! ALL 8 neural methods (`drqn_aux_yr`, `drqn_aux_discovery`, `uniform`, `vanilla`, etc.) collapsed 100% to Tier 0 (`all-S = +12.80`, `RESOLVED_NULL` vs `vanilla`), because 22-action random warmup rarely repeats gated exploits (`<0.1%`) and once TD settles on `S`, gradient-aligned `drqn_aux_discovery` receives no extrinsic signal favoring `cand_y` (`w_y = 0.08`).

## Next Actionable Step
1. Decide with user: (A) test whether **curiosity/count-based intrinsic bonus or curriculum** breaks Exp 4's Tier-0 wall (and test Auxiliary Task Discovery on a regime where TD *is* active, e.g. lower-budget `R3` with distractors), OR (B) run final Codex `research_critic`/`validator` audit across Exp 1–4 and begin publication figures + report/video prep.
