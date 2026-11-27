# TASK-002: Independent Contract, Reproducibility & Submission Size Verification

- **Task ID:** `TASK-002`
- **Assigned Role:** `validator` (`.codex/agents/validator.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
Independently verify code contracts, non-leakage invariants, sealed CSV/JSON consistency across Exp 1 / 2A / 2B, and compliance with the `<= 5 MB` course code-zip limit.

## Verification Scope
1. Run the full test suite:
   `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`
2. Independently inspect `src/mamrl/agents/{baselines_v3,drqn_rl2,ppo_rl2,opponents_v3}.py`, `src/mamrl/envs/{delayed_response,symmetric_ma}.py`, and `src/mamrl/study_{v3,exp2a,exp2b}.py` for:
   - Zero hidden-type or payoff-matrix leakage into unprivileged agents.
   - Exact masking boundaries (`rl2_masked`, `prior_only`) and no retrospective revelation.
   - Exact `train` vs `test` rule disjointness in `symmetric_ma.py`.
   - Consistency between `results/*/summary.json` and `results/*/tables/*.csv`.
3. Measure exact git-tracked submission archive size (`git archive --format=zip HEAD` to `/tmp/mamrl_size_check.zip` and remove `/tmp/mamrl_size_check.zip` afterward) against the `5 MB` limit.

## Allowed Write Paths
- `handoff/reports/TASK-002_contract_audit_report.md`

## Forbidden Paths
- All other files (`src/`, `configs/`, `tests/`, `results/`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`).

## Expected Report Output
Write a concise PASS/FAIL verification report with exact commands and observed outputs to `handoff/reports/TASK-002_contract_audit_report.md`. Do not run `git commit`.
