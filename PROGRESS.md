# Progress Snapshot — 2026-11-28

## Status & Deadlines
- **Goal:** Unbiased empirical study of how meta-learning affects exploration & fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Active Exploration (Not Frozen).** 8 5-seed studies complete (`Exp 1, 2A, 2B, 3A, 3B, 4, 5A, 4B`); 3-agent collaboration (`Gemini Lead` + `Codex CLI` + `Antigravity-Claude` via `agy`) fully operational.

## Verified Evidence Summary (Compact)
- **Exp 1 (`R0/R1/R3`, `T=24`, 8a):** `drqn_aux_yr` hits `99.7%` Bayes-Opt (`+17.52 / 17.575`), tying structure-informed `structural_model` (`+17.50`).
- **Exp 2A / 3A / 5A-1A,1B (Online Opponents + Noise $d \in \{0, 0.3, 0.6\}$):** Online-matched DRQN beats misspecified `structural_model` (Restored `+1.86`, $d=0.6$ `+4.66`), yet collapses OOD via overconfident exploitation despite PBVI proving `VoI >= +7.20` (`12.00` vs `4.80`). Eval-only surprise switch fails escalation vs `prior_only` → **GRU-surprise retraining STOPPED**.
- **Exp 2B / 3B / 5A-1C (Symmetric & 5 Semantic Families, `T=24`, 4a):** Gate-0 confirms `100%` positive VoI rules on `coop_trap` & Exp 3B (`0` on `comp_rps` → **STOPPED**). `EquivariantDRQN` jumps `+12.6..+13.6` over One-Hot and hits `+30.71` (`+8.02` vs UCB) on unseen 5th family.
- **Exp 4 / 5A-1D / 4B (Multi-Barrier `T=64` 22a → Compact Stage-Lock `T=48` 6a):** Diag 1D proved 22-action warmup data already sufficed (`structural_model_mb` mean `84.84`, median `120.55`, 5/5 unlock) → **22-action interventions STOPPED**. Compact 6-action Stage-Lock unlocks Tier 3 (`vanilla +67.27`, `aux_yr` median `+76.25` vs `arm_ucb +0.40`); `drqn_aux_discovery` assigns `95.3%/87.2%` to `(y,r)` (`R3 +7.50` vs vanilla, Stage-Lock `+23.52` vs uniform). Recurrent target bug in `study_exp4b_compact.py` fixed & unit-tested (re-seal pending).

## Environment Design Principle & Active Roadmap (`docs/SPEC.md` §8)
- **Design Rule:** Drop 1-to-1 action-per-attribute enumeration; build mathematically structured repeated-game families with pre-verified exact Oracles and clean horizon scaling (`T=48~64`).
- **7 Active Items:** `U1` Multi-family × Aux Discovery (global vs dynamic gate) · `U2` Multi-barrier × online learning opponents · `U3` Bilevel meta-gradient & richer aux candidates · `U4` Noise-matched training ($d \in \{0, 0.3\} \to 0.6$) · `P1` Exp 3B state-matched/augmented/masked controls · `P2` Bug-fixed Exp 4B re-seal + probe attribution · `P3` `rl2_ppo` `691k` budget match & SARSA reconciliation.
