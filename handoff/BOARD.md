# Multi-Agent Collaboration Board (Lead-Owned)

- **Default Lead Agent:** `Antigravity-Gemini` (`GEMINI.md`) — sole writer of `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/`, git commits/merges/pushes.
- **Antigravity-Claude Collaborator:** `CLAUDE.md` — roles `claude_architect`, `claude_analyst`, `claude_writer` (parallel collaborator via scoped `TASK-*` brief, or in-place Lead handoff when explicitly switched by the user).
- **Codex CLI Collaborators:** `.codex/agents/{research_critic, results_analyst, validator, experiment_worker}.toml`.
- **Token Rule:** Read compact `results/*/tables/*.csv` and `handoff/context_pack.md` first (skip raw `runs/` & full `summary.json` unless needed); keep reports `<= 120` lines.

## Active Task Queue

| Task ID | Agent | Role | Status | Brief | Allowed Write Paths | Expected Report |
|---|---|---|---|---|---|---|
| `TASK-001` | Codex CLI | `research_critic` | `REVIEWED` | `handoff/tasks/TASK-001_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` | `handoff/reviews/20261127_exp1_2a_2b_audit.md` |
| `TASK-002` | Codex CLI | `validator` | `REVIEWED` | `handoff/tasks/TASK-002_contract_audit.md` | `handoff/reports/TASK-002_contract_audit_report.md` | `handoff/reports/TASK-002_contract_audit_report.md` |
| `TASK-003` | Codex CLI | `results_analyst` | `REVIEWED` | `handoff/tasks/TASK-003_legacy_bridge_and_insights.md` | `handoff/reports/TASK-003_legacy_bridge_and_insights.md` | `handoff/reports/TASK-003_legacy_bridge_and_insights.md` |
| `TASK-004` | Codex CLI | `research_critic` | `REVIEWED` | `handoff/tasks/TASK-004_exp3_4_interim_audit.md` | `handoff/reviews/20261127_exp3_4_interim_audit.md` | `handoff/reviews/20261127_exp3_4_interim_audit.md` |
| `TASK-005` | Codex CLI | `validator` | `REVIEWED` | `handoff/tasks/TASK-005_exp3_4_contract_audit.md` | `handoff/reports/TASK-005_exp3_4_contract_audit.md` | `handoff/reports/TASK-005_exp3_4_contract_audit.md` |
| `TASK-006` | Antigravity-Claude | `claude_analyst` | `OPEN` | `handoff/tasks/TASK-006_claude_interim_design_review.md` | `handoff/reports/TASK-006_claude_interim_design_review.md` | `handoff/reports/TASK-006_claude_interim_design_review.md` |

## Lead-Owned Protected Paths (Never Modified by Collaborators)
- `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`
- Sealed evidence: `results/core_v3_sealed/`, `results/exp2a_online_opp/`, `results/exp2b_symmetric_ma/`, `results/exp3a_reliability_sweep/`, `results/exp3b_equivariant_semantic/`, `results/exp4_multibarrier_aux_discovery/`, `results/gate0_exact_dp.json`

## Quick Dispatch Commands
```bash
.venv/bin/python scripts/codex_handoff.py status
.venv/bin/python scripts/codex_handoff.py run --id TASK-004     # invokes OpenAI Codex CLI directly
.venv/bin/python scripts/codex_handoff.py prompt --id TASK-006  # prints ready prompt for Antigravity-Claude
.venv/bin/python scripts/codex_handoff.py check-scope --id TASK-006
```
