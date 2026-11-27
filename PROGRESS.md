# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Design Verification Gate (v2.1)**. Repository cleanly refactored on `refactor/v2-clean` (1.7 MB tracked, 6/6 core tests passing). Finalizing unbiased experimental design (`docs/SPEC.md`) before updating `src/mamrl/` and running final multi-seed experiments.

## Completed Work (Verified)
- Archived full legacy repo (`290dabf`, 15 verified studies) at git tag `archive/v0.9-legacy-studies`; distilled failures into `docs/PRIOR_FINDINGS.md` and preserved key figures/tables (`results/legacy_summary/`, 340 KB).
- Audited proposal vs. legacy studies: confirmed that prior work varied superficial size (`n=3->6`, `depth=1->5`) while *reducing* active-probing necessity (`12.5% -> 6.3%`), never cleanly testing structured partial observability or arm-independent UCB.
- Diagnosed pilot flaw in initial v2 `students.py`: passing privileged `spec` into `BayesianRewardStudent` gave `UCB` secret model knowledge. Redesigned `docs/SPEC.md` (v2.1) with strict unprivileged baselines, R0–R3 observability/complexity regimes, `RL²` comparison, and pre-registered falsification criteria.

## Active Design Verification Questions (`docs/SPEC.md` §5)
1. How to pair cross-action probing (R3) with an unprivileged Student (e.g., meta-prior Student + end-to-end `RL²`) without biasing UCB.
2. Simplest genuinely multi-agent R3 game (not a disguised single-agent bandit) with tractable exact Oracle.
3. REINFORCE vs. PPO-GAE stability under noisy/multi-step regimes (R2/R3).
4. Scope sizing for the 2026-12-17 hard deadline.

## Next Actionable Step
1. Run `scripts/pack_context.py` and review `handoff/context_pack.md` with GPT / `research_critic` to resolve §5 open design questions and lock `docs/SPEC.md`.
2. Implement locked v2.1 design in `src/mamrl/`, pass Gate-0 Oracle & unit tests, and run Exp 1 (R0–R3).
