# TASK-005: Engineering & Contract Validation Audit for Exp 3A, 3B, 4 and Masking Fix

- **Task ID:** `TASK-005`
- **Assigned Role:** `validator` (`.codex/agents/validator.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
Run a strict, independent contract and integrity audit on all code and sealed outputs added since `TASK-002` (`ac0766a`, `5799b7d`, `b9a0abe`, `7ad91cf`):
1. **Strict Masking Consistency (`step_t > mask_k`):** Verify across `src/mamrl/agents/ppo_rl2.py`, `src/mamrl/study_exp2b.py`, `src/mamrl/study_exp3b.py`, and `src/mamrl/study_exp4_multibarrier.py` that NO off-by-one leakage remains for masked controls or blind `prior_only` controls.
2. **Non-Leakage & Unprivileged Fairness:** Verify no privileged `theta`, payoff tables, or opponent internals leak into evaluated neural/online policies in `src/mamrl/study_reliability_sweep.py`, `src/mamrl/study_exp3b.py`, `src/mamrl/agents/equivariant_drqn.py`, and `src/mamrl/study_exp4_multibarrier.py`.
3. **Mathematical Invariants:** Verify `EquivariantDRQN` $S_4$ equivariance, Exp 3B disjoint train/test matrices, Exp 4 exact rational Gate-0 DP (`T=6..64`), and Exp 4 bilevel gate update math (`src/mamrl/study_exp4_multibarrier.py`).
4. **Artifact & Table Integrity:** Programmatically verify CSV ↔ `summary.json` numerical consistency and non-NaN completeness across `results/exp3a_reliability_sweep/`, `results/exp3b_equivariant_semantic/`, `results/exp4_multibarrier_aux_discovery/`, and updated `results/core_v3_sealed/`, `results/exp2a_online_opp/`, `results/exp2b_symmetric_ma/`.
5. **Repo Health & Submission Budget:** Verify `git archive --format=zip HEAD` size (`<< 5 MB`) and confirm no stray large files are tracked.

## Required Reading
- `AGENTS.md`, `handoff/BOARD.md`
- `src/mamrl/agents/ppo_rl2.py`, `src/mamrl/study_exp2b.py`, `src/mamrl/study_reliability_sweep.py`, `src/mamrl/envs/semantic_games.py`, `src/mamrl/agents/equivariant_drqn.py`, `src/mamrl/study_exp3b.py`, `src/mamrl/envs/multi_barrier.py`, `src/mamrl/study_exp4_multibarrier.py`
- `results/*/tables/*.csv` and `results/*/summary.json` (programmatic read-only check; skip `runs/`).

## Allowed Write Paths
- `handoff/reports/TASK-005_exp3_4_contract_audit.md`

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `src/`, `configs/`, `tests/`, sealed `results/`).

## Verification & Expected Report Output
1. Run read-only Python verification checks (do NOT edit any source/test files).
2. Write a concise PASS/FAIL matrix (`<= 100` lines) to `handoff/reports/TASK-005_exp3_4_contract_audit.md` listing exact checks, any bugs/edge cases found, CSV/JSON reconciliation counts, and submission archive size.
3. Do NOT run `git commit`.
