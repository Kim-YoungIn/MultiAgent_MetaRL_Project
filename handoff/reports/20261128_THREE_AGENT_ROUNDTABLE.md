# 3-Agent Project-Wide Roundtable Synthesis (2026-11-28)

- **Participants:** Antigravity-Gemini (Lead), Codex CLI `research_critic` (`TASK-007`, `gpt-6.1-sol/high`), Antigravity-Claude `claude_analyst` (`TASK-008`, `claude-opus-5-5-high`).
- **Source Reports:** [20261128_codex_project_discussion.md](../reviews/20261128_codex_project_discussion.md), [TASK-008_claude_project_discussion.md](TASK-008_claude_project_discussion.md).

## 1. Unanimous 3-Way Agreements
1. **Conditional Amortization Thesis:** Meta-learning does *not* provide universal exploration superiority or require auxiliary losses everywhere. It accelerates adaptation specifically when (a) information has positive decision value, (b) probe→payoff credit paths are tractable, (c) inductive bias matches task symmetry, and (d) test lies inside training support.
2. **Exp 5A + Exp 4B Key Verdicts:**
   - **Diag 1D (22-action Exp 4):** Equal-budget random warmup data already contain 508–602 positive Tier-3 events (`8/8` cells), and structure-informed `structural_model_mb` unlocks Tier 3 on 5/5 seeds (cross-seed mean **84.84**, median **120.55**, 3/5 hit exact `120.625`). Pure "missing data" Catch-22 is falsified.
   - **Diag 1A ($d=0.6$ PBVI):** Achievable PBVI lower bound is **12.00** vs blind `4.80` (`VoI >= +7.20`); OOD collapse (`0.44`) is miscalibrated overconfident exploitation, not VoI collapse.
   - **Diag 1B (Surprise Switch):** Improves $d=0.6$ (`0.44→2.00`, median `-0.50→4.20`, 5/5 seeds) but still trails `prior_only` (`-2.80`) and hurts $d=0.3$ (`-0.99`). **Stop GRU-surprise retraining.**
   - **Exp 4B (`R3+Distractor` & Compact 6-Action Stage-Lock):** Local cosine aux gating passes distractor suppression (`95.3%` / `87.2%` weight on `(y,r)`, `+7.50` over vanilla on R3, `+23.52` over uniform on Stage-Lock) — frame accurately as **harm-avoidance**, not universal gain over vanilla (`INCONCLUSIVE` on Stage-Lock). Compact sequential Stage-Lock unlocks Tier 3 (`vanilla +67.27`, `aux_yr` median `+76.25` vs `arm_ucb +0.40`), but changed 4 factors at once so do not isolate action count as sole cause.
3. **Unanimous #1 Final Experiment Block (Pre-Freeze ~Dec 8):**
   - **Exp 3B Controls (`25` fits):** One-Hot `state-matched (128 scalars) × permutation-augmented` controls (3 new cells × 5 seeds) + Equivariant `masked-feedback` & `prior_only` (10 fits), primary contrast = Equivariant vs Augmented State-Matched One-Hot focal return vs UCB on `unseen_family`.
4. **Unanimous STOP List:** No GRU-surprise retraining, no new barrier/environment variants, no 22-action interventions, no full D-031 expansion, no new `comp_rps` work.

## 2. Productive Disagreements & Lead Resolution
- **Coverage vs Multi-Factor Thesis:** Claude centered *training-support coverage*; Codex argued coverage alone cannot explain 22-action neural failure given Diag 1D and urged *information exists, but representation + planning + optimization govern conversion*. **Lead Resolution:** Adopt Codex's broader multi-factor framing while retaining Claude's sharp *bounded-amortization / OOD commitment* operationalization.
- **Fairness Labeling of `structural_model*`:** Both flagged that `structural_model_sl` (and `structural_model`) is data-unprivileged (zero hidden keys/labels) yet **model-class-informed** (known transition/cluster structure). **Lead Resolution:** Label strictly as **"structure-informed learned-model planner (model class known)"** so the DRQN gap cleanly measures *the amortization cost of unknown model class*.

## 3. Mandatory Engineering & Integrity Fixes Before Freeze
1. **Fix `study_exp4b_compact.py:344` recurrent Bellman target history mismatch** caught by Codex (`tx[:, 1:, :]` omitted turn 0 hidden state; switch to full-sequence unroll `tnet(tx)[:, 1:, :]` and reuse `q_all.detach()[:, 1:, :]`, then re-seal Exp 4B with exact Exp 1 hyperparams on `R3+distractor` too!).
2. **Eval-only Stage-Lock probe attribution** (Claude §3B): report `P`-before-exploit rate, turns to key ID, and repeat-trial discovery rate on checkpoints.
3. **Budget-match `rl2_ppo` on R3 (`691k` transitions, 5 fits)** OR explicitly scope out RL² claims.
4. **Reconcile Diag 1B `Expected SARSA` (`11.60`) vs Exp 2A (`3.29`)** opponent config mismatch, report Diag 1D cross-seed mean `84.84`, rename Exp 4 floor ties to `NOT_TESTABLE_AT_FLOOR`, and print `k/5` seed counts + median/IQR beside every bootstrap CI.
