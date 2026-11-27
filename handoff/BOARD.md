# Lead ↔ Codex Collaboration Board (Lead-Owned)

- **Lead Agent:** `Antigravity` (sole writer of `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/`, git commits).
- **Codex Collaborators:** `.codex/agents/{research_critic, validator, experiment_worker, paper_reviewer}.toml`.
- **Rule:** Codex agents MUST read `AGENTS.md` + their assigned brief in `handoff/tasks/` and write ONLY inside `Allowed Write Paths`.

## Active Task Queue

| Task ID | Role | Status | Brief | Allowed Write Paths | Expected Report |
|---|---|---|---|---|---|
| `TASK-001` | `research_critic` | `OPEN` | `handoff/tasks/TASK-001_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` |
| `TASK-002` | `validator` | `OPEN` | `handoff/tasks/TASK-002_contract_audit.md` | `handoff/reports/TASK-002_contract_audit_report.md` | `handoff/reports/TASK-002_contract_audit_report.md` |

## Lead-Owned Protected Paths (Never Modified by Codex Workers)
- `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`
- Sealed evidence: `results/core_v3_sealed/`, `results/exp2a_online_opp/`, `results/exp2b_symmetric_ma/`, `results/gate0_exact_dp.json`

## Quick Dispatch Commands
```bash
.venv/bin/python scripts/codex_handoff.py status
.venv/bin/python scripts/codex_handoff.py prompt --id TASK-001
.venv/bin/python scripts/codex_handoff.py check-scope --id TASK-001
```
