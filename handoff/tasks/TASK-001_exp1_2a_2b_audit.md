# TASK-001: Independent Scientific Audit of Sealed Exp 1 + Exp 2A + Exp 2B

- **Task ID:** `TASK-001`
- **Assigned Role:** `research_critic` (`.codex/agents/research_critic.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
Perform an unbiased, publication-grade scientific audit of the sealed 5-seed evidence across **Exp 1** (`results/core_v3_sealed/`), **Exp 2A** (`results/exp2a_online_opp/`), and **Exp 2B** (`results/exp2b_symmetric_ma/`) before finalizing the Priority Extension (**Auxiliary Task Discovery**) and drafting the 5-page NeurIPS report.

## Required Reading
- `AGENTS.md`, `PROGRESS.md`, `docs/SPEC.md`, `docs/PRIOR_FINDINGS.md`
- `results/gate0_exact_dp.json`
- `results/core_v3_sealed/tables/{methods,paired_contrasts,layer1_k248,behavior_repetition}.csv`
- `results/exp2a_online_opp/tables/{methods,paired_contrasts,restore_vs_continuous}.csv`
- `results/exp2b_symmetric_ma/tables/{matchups,paired_contrasts}.csv`

## Audit Checklist
1. **Claim Boundaries & Overclaiming Check:** Which claims are strictly supported (`SUPPORTED_POS/NEG`), resolved null (`RESOLVED_NULL`), or inconclusive (`INCONCLUSIVE`) across Exp 1, 2A, and 2B?
2. **Negative/Conditional Findings Integration:** Verify that PPO collapse, zero-shot brittleness vs `structural_model`, `expected_sarsa_unseen` OOD failure vs `prior_only`, and `comp_rps` shortcut memorization are framed rigorously rather than hidden.
3. **Exp 2A & 2B Methodological Critique:** Identify any remaining confound (e.g., `comp_rps` 4-rule split design, `coop_trap` cross-seed variance, Restored intervention semantics) and state whether it requires a caveat or a small control run.
4. **Priority Extension Gate:** Give concrete minimal-viable falsification criteria for **Auxiliary Task Discovery (`보조 목표 탐색`)** so it stays compact and unbiased before 2026-12-17.

## Allowed Write Paths
- `handoff/reviews/20261127_exp1_2a_2b_audit.md`

## Forbidden Paths
- All other files (`src/`, `configs/`, `tests/`, `results/`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`).

## Expected Report Output
Write your structured audit directly to `handoff/reviews/20261127_exp1_2a_2b_audit.md`. Do not run `git commit`.
