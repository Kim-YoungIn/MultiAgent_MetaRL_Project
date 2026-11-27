# Multi-Agent Collaboration Guide (Lead: Antigravity ↔ Codex / GPT)

This repository uses a **Lead-Orchestrated Handoff Architecture** so **Antigravity (Lead Agent)** can delegate work cleanly to **Codex agents** (Codex CLI / Codex IDE / `.codex/agents/` subagents) and **GPT reviewers** without file conflicts, spec drift, or directory sprawl.

---

## 1. Authority & Write-Ownership Matrix

| Agent / Role | Definition | Exclusive / Allowed Write Scope | Never Modifies |
|---|---|---|---|
| **Antigravity (Lead Agent)** | Primary coordinator | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `handoff/tasks/`, `src/`, `configs/`, `tests/`, `results/`, git commits | — |
| **Codex `research_critic`** | `.codex/agents/research_critic.toml` | Assigned `handoff/reviews/YYYYMMDD_<topic>.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md` |
| **Codex `validator`** | `.codex/agents/validator.toml` | Assigned `tests/` path + `handoff/reports/TASK-*.md` ONLY | `src/`, `configs/`, `results/`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md` |
| **Codex `experiment_worker`** | `.codex/agents/experiment_worker.toml` | Explicit `Allowed Write Paths` in `handoff/tasks/TASK-*.md` + `handoff/reports/TASK-*.md` | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/` |
| **Codex `paper_reviewer`** | `.codex/agents/paper_reviewer.toml` | Assigned `handoff/reports/` or assigned paper/slide draft path | `src/`, `configs/`, sealed `results/`, `docs/SPEC.md`, `PROGRESS.md` |
| **GPT (ChatGPT Web)** | External reviewer | User pastes output into `handoff/reviews/YYYYMMDD_<topic>.md` | Direct repo edits |

---

## 2. Standard 4-Step Lead ↔ Codex Workflow

1. **Lead prepares context & dispatches a scoped task:**
   ```bash
   .venv/bin/python scripts/pack_context.py
   .venv/bin/python scripts/codex_handoff.py new-task \
     --id TASK-003 --role experiment_worker \
     --title "Implement auxiliary task discovery prototype" \
     --allow "src/mamrl/agents/aux_discovery.py,tests/test_aux_discovery.py,handoff/reports/TASK-003_report.md"
   ```
   Lead updates `handoff/BOARD.md` and generates the ready-to-run Codex prompt:
   ```bash
   .venv/bin/python scripts/codex_handoff.py prompt --id TASK-003
   ```

2. **Codex executes strictly within scope:**
   - Paste the generated prompt into a Codex session (or run via Codex CLI in another terminal):
     ```bash
     # Optional isolated branch/worktree when running concurrently with Lead edits:
     git worktree add ../mamrl-codex-TASK-003 -b codex/TASK-003
     ```
   - Codex reads `AGENTS.md` → `handoff/BOARD.md` → `handoff/tasks/TASK-003_*.md`, invokes `.codex/agents/<role>.toml` if using Codex subagents, stays strictly inside `Allowed Write Paths`, runs `unittest`, and writes its report to `handoff/reports/`.

3. **Lead verifies scope & contracts automatically:**
   ```bash
   .venv/bin/python scripts/codex_handoff.py check-scope --id TASK-003
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
   ```

4. **Lead reviews, adopts, updates `PROGRESS.md`, and commits:**
   - Lead inspects the Codex report in `handoff/reports/` or `handoff/reviews/`.
   - Approved changes are merged/integrated, `docs/SPEC.md` (if needed) and `PROGRESS.md` are updated, `handoff/BOARD.md` status is marked `ADOPTED`, and Lead creates a clean commit.

---

## 3. Useful Helper Commands

- Validate `.codex/config.toml` and all `.codex/agents/*.toml`:
  ```bash
  .venv/bin/python scripts/codex_handoff.py validate-codex
  ```
- Inspect active board & pending reports:
  ```bash
  .venv/bin/python scripts/codex_handoff.py status
  ```
- Refresh single-file context pack (`handoff/context_pack.md`):
  ```bash
  .venv/bin/python scripts/pack_context.py
  ```
