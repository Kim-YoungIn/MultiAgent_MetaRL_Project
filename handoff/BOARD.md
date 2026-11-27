# Lead ↔ Codex Collaboration Board (Lead-Owned)

- **Lead Agent:** `Antigravity` (sole writer of `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/`, git commits).
- **Codex Collaborators:** `.codex/agents/{research_critic, results_analyst, validator, experiment_worker}.toml`.
- **Token Rule:** Read compact `results/*/tables/*.csv` first (skip raw `runs/` & full `summary.json` unless needed); keep reports `<= 120` lines.

## Active Task Queue

| Task ID | Role | Status | Brief | Allowed Write Paths | Expected Report |
|---|---|---|---|---|---|
| `TASK-001` | `research_critic` | `REVIEWED` | `handoff/tasks/TASK-001_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` |
| `TASK-002` | `validator` | `REVIEWED` | `handoff/tasks/TASK-002_contract_audit.md` | `handoff/reports/TASK-002_contract_audit_report.md` | `handoff/reports/TASK-002_contract_audit_report.md` |
| `TASK-003` | `results_analyst` | `REVIEWED` | `handoff/tasks/TASK-003_legacy_bridge_and_insights.md` | `handoff/reports/TASK-003_legacy_bridge_and_insights.md` | `handoff/reports/TASK-003_legacy_bridge_and_insights.md` |

## Lead-Owned Protected Paths (Never Modified by Codex Workers)
- `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`
- Sealed evidence: `results/core_v3_sealed/`, `results/exp2a_online_opp/`, `results/exp2b_symmetric_ma/`, `results/gate0_exact_dp.json`

## Quick Dispatch Commands
```bash
.venv/bin/python scripts/codex_handoff.py status
.venv/bin/python scripts/codex_handoff.py run --id TASK-001     # invokes OpenAI Codex CLI directly if installed
.venv/bin/python scripts/codex_handoff.py prompt --id TASK-001  # prints copy-paste prompt
.venv/bin/python scripts/codex_handoff.py check-scope --id TASK-001
```
