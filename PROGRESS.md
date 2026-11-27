# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Exp 1, Exp 2A, Exp 2B sealed & re-verified after `TASK-001/002/003` Codex audits (strict `step_t > mask_k` masking fix + complete CSV columns; 30/30 tests pass; git archive `~0.65 MB < 5 MB`).**

## Verified Results Summary (5 Sealed Seeds `1001..1005`, `n_eval=512`, `10k` Hierarchical Bootstrap)
- **Exp 1 (`R0/R1/R3`, Fixed Opponent):** `drqn_aux_yr` reached `+20.20 / +19.10 / +17.52` (99.7% Bayes-Opt; `L1_K4 commit=+17.00`), beating strict `masked` (`+2.345`, `L1_commit +13.00`, `SUPPORTED_POS`), `vanilla` (`+4.590`, variance-reduction on hard `θ=2`), `aux_r` (`+5.305`), `rl2_ppo` (`+4.80`, note `264k` vs `691k` steps), and online baselines, while tying `structural_model` (`+17.50`, `RESOLVED_NULL`, at 28× fewer offline transitions `24.6k`).
- **Exp 2A (`R3` × Online `Q-learning` / Unseen `Expected SARSA`, Continuous vs. Restored):**
  - Zero-shot fixed-trained `zs_drqn_aux_yr` collapsed (`+4.41 / -7.02`), losing to smoothed `zs_structural_model` (`+8.74 / +1.74`, `SUPPORTED_NEG`).
  - Matched online training `mt_drqn_aux_yr` recovered to #1 mean on `Q-learning` (`+9.84`, `+5.43` over zero-shot, `+1.75` over strict `masked`, `SUPPORTED_POS`) and beat `mt_structural_model` under Restored (`+1.86`, `SUPPORTED_POS`; DiD `-0.92`, `SUPPORTED_NEG`; Continuous `+0.94`, `INCONCLUSIVE`).
  - Honest OOD boundary: on unseen `Expected SARSA`, `mt_drqn_aux_yr` (`+3.29`) lost to blind `prior_only` (`+4.80`, `-1.51` `SUPPORTED_NEG`) and conservative `mt_drqn_vanilla` (`+4.24`, `-0.94` `SUPPORTED_NEG`).
- **Exp 2B (Symmetric 4-Action Games, Disjoint Train/Unseen Splits):**
  - **`coop_trap` (16/8):** Unseen cross-seed `meta_aux_yr × meta_aux_yr` reached Welfare `+31.70` (`coopL=61.6%`), beating `Q×Q` (`+26.83`), `UCB×UCB` (`+12.19`), and `meta_vanilla × meta_vanilla` (`+11.00`, all `SUPPORTED_POS`); vs strict `masked` (`+2.44` vs Q, `+7.68` vs UCB, `SUPPORTED_POS`).
  - **`comp_rps` (4/4):** In-dist Meta exploited `UCB` (`+3.41` margin) but `meta_aux_yr` lost to stateful `Q-learner` (`-0.98`, `SUPPORTED_NEG`); on unseen flipped orientations, margins vs `Q`/`Contextual-UCB` were `-0.99 / -0.95` (`SUPPORTED_NEG`) and vs `UCB` `-0.08` (`INCONCLUSIVE`).

## Next Actionable Step (Aligned With User + Codex Audits)
1. **Legacy Bridge A (Evaluation-only, ~1 min):** Reliability / dynamics sweep $d \in \{0, 0.3, 0.6\}$ on existing `R3` & `coop_trap` checkpoints.
2. **Anti-Memorization + Semantic Rule Diversity (D-031 integrated):** Permutation-equivariant `EquivariantDRQN` vs one-hot `DRQN` across genuinely distinct semantic game families (Coordination / Stag-Hunt / Chicken / Dilemma / Cyclic) with exposure-controlled diversity scaling.
3. **Priority Extension:** Auxiliary Task Discovery (`보조 목표 탐색`) under `TASK-001` falsification gates.
