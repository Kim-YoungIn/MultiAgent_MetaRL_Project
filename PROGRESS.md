# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Unbiased empirical inquiry into how meta-learning affects exploration and fast adaptation in multi-agent repeated games.
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** **Design Verification Gate (v2.2)**. Adopted unprivileged End-to-End `RL²` (Recurrent PPO+GAE) + **3-Layer Exploration/Exploitation Decomposition** (Neutral Bayes Info/Value Audit + Mixed-Prefix Neutral Exploiter + Belief/Exploitation Audit) to eliminate Student bottleneck and OOD prefix-swap bias.

## Completed Work (Verified)
- Archived full legacy repo (`290dabf`, 15 verified studies) at git tag `archive/v0.9-legacy-studies`; distilled failures into `docs/PRIOR_FINDINGS.md` and preserved compact evidence (`results/legacy_summary/`, 340 KB).
- Rebuilt clean v2 repo on `refactor/v2-clean` (1.7 MB tracked footprint, 6/6 baseline core tests passing).
- Finalized unbiased `docs/SPEC.md` (v2.2) & `AGENTS.md`: R0–R3 observability/complexity regimes, strictly unprivileged baselines, 3-layer Exploration/Exploitation decomposition without OOD bias, Opponent-Restored vs. Continuous shaping split, and pre-registered interpretation rules.

## Open Verification Points (`docs/SPEC.md` §5)
1. Exact R3 multi-agent game rules (genuinely multi-agent + tractable Gate-0 Oracle & Layer-1 Bayes audit).
2. Standard GRU vs. Action-Equivariant Set/Attention architecture under action relabeling.
3. Exact $(T, K_{\text{audit}})$ & PPO-GAE training budget.
4. Matrix sizing for the 2026-12-17 hard deadline.

## Next Actionable Step
1. Share `handoff/context_pack.md` with GPT / `research_critic` to audit `docs/SPEC.md` v2.2 and close §5 design questions.
2. Lock `docs/SPEC.md`, upgrade `src/mamrl/` to `RL²`-PPO + 3-Layer Decomposition + R0–R3 regimes, verify tests & Gate-0 Oracle, and run Exp 1.
