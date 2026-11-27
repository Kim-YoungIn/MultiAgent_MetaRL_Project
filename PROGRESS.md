# Progress Snapshot — 2026-11-27 (Interim Checkpoint)

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Interim Checkpoint COMPLETE (not final).** 6 sealed 5-seed studies (Exp 1, 2A, 2B, 3A, 3B, 4) + 5 independent Codex CLI audits (`TASK-001..005`) adopted into `handoff/reports/INTERIM_SYNTHESIS_EXP1_TO_4.md`. **Antigravity-Claude** collaboration configured (`CLAUDE.md`, `GEMINI.md`, `AGENTS.md`, `docs/COLLABORATION.md`, `TASK-006`). Ready to push to GitHub `master`.

## Verified 5-Seed Sealed Evidence (Audit-Qualified)
- **Exp 1 (`R0/R1/R3`, Fixed Opponent):** `drqn_aux_yr` hits `99.7%` Bayes-Opt (`+17.52 / 17.575`), beating strict `masked` (`+2.345`; `R1` K4 commit/adapt/net revised to `+13.0/+1.9/+0.5` after `ac0766a`), `vanilla` (`+4.590`), `aux_r` (`+5.305`), and `rl2_ppo` (`+4.80`, `264k` budget caveat), while tying 28×-cheaper `structural_model` (`+17.50`, `RESOLVED_NULL`).
- **Exp 2A & 3A (Online Opponents + $d \in \{0,0.3,0.6\}$ Stress Sweep):** Fixed-trained `zs_drqn_aux_yr` collapses (`+4.41/-7.02`); Exp-2A online-matched `mt_drqn_aux_yr` leads `Q-learning` (`+9.84`, Restored `>+1.86` over `structural_model`, `d=0.6` `>+4.66` over `structural_model`), yet loses to blind `prior_only (+4.80)` & `mt_vanilla` on unseen `Expected SARSA` and $d=0.6$, and `coop_trap` slip reverses vs `UCB×UCB` (`+12.32 → +2.47 → -3.85`).
- **Exp 2B & 3B (Symmetric Games + $S_4$ `EquivariantDRQN` + Semantic Diversity / D-031):** On 4 semantic families, One-Hot `DRQN` loses to `UCB×UCB` everywhere even at 2× budget; the exact $S_4$ **`EquivariantDRQN` package** (`14.3k` params, `4×32` slot states — capacity control pending) jumps **`+12.63/+13.61/+12.68` (`SUPPORTED_POS`)** over One-Hot, reaches `+30.71` (`+8.02` over `UCB×UCB`) vs UCB on unseen 5th family (`xseed` `-3.89 INCONCLUSIVE`), and shows `INCONCLUSIVE` diff vs `equiv_vanilla`. Doubling `n4` budget helps 3/4 suites (`+4.09..+6.30`), while `n2` beats `n4_b2` on unseen family (`+5.37`).
- **Exp 4 (3-Layer Multi-Barrier `T=64`, 22 actions, Gate-0 `12.8 → 41.8 → 75.55 → 120.625` + Cosine Gradient-Alignment Aux Gating):** All 8 neural methods fail to exploit higher tiers (`tier0_only = 1.0`, `discovery +12.74`, `aux_y +10.01`, `vanilla +12.80`, `RESOLVED_NULL`), while `arm_ucb/greedy` crash (`-94.8/-36.9`). Sparse training coverage and local TD-cosine gating are candidate causes pending intervention.

## Next Actionable Step
1. Dispatch **Antigravity-Claude** on `TASK-006` (`handoff/tasks/TASK-006_claude_interim_design_review.md`) and choose next follow-up experiment: (1) Exp 4B coverage × auxiliary-gating intervention, (2) Exp 3B capacity-matched & masked/D-031 controls, or (3) online surprise-calibrated fallback for Exp 2A/3A.
