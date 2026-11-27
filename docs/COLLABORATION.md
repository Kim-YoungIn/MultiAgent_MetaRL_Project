# Multi-Agent Collaboration Guide (Lead: Antigravity ↔ Codex / GPT)

This repository uses a **Lead-Orchestrated Handoff Architecture** so **Antigravity (Lead Agent)** can delegate work cleanly to **Codex agents** (OpenAI Codex CLI `@openai/codex` / Codex IDE / `.codex/agents/` subagents) without file conflicts, spec drift, directory sprawl, or unnecessary token burn.

---

## 1. Authority & Write-Ownership Matrix (Token-Tiered)

| Agent / Role | Model & Effort | Primary Purpose | Allowed Write Scope | Never Modifies |
|---|---|---|---|---|
| **Antigravity (Lead Agent)** | Lead | Architecture, core experiments, adoption, git commits | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `handoff/tasks/`, `src/`, `configs/`, `tests/`, `results/` | — |
| **Codex `research_critic`** | `gpt-6.1-sol` (`high`) | **Validity Defense:** audit statistical verdicts, confounds, overclaiming, OOD bounds | Assigned `handoff/reviews/YYYYMMDD_<topic>.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Codex `results_analyst`** | `gpt-6.1-sol` (`high`) | **Discovery & Interpretation:** mine CSVs for mechanisms, anomalies, legacy-to-v3 links, next experiments | Assigned `handoff/reports/TASK-*.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Codex `validator`** | `gpt-6.1-terra` (`medium`) | **Contract Verification:** non-leakage, exact DP, disjoint splits, CSV/JSON match, `<5 MB` zip | Assigned `tests/` + `handoff/reports/TASK-*.md` | `src/`, `configs/`, `results/`, governance docs |
| **Codex `experiment_worker`** | `gpt-6.1-terra` (`medium`) | **Scoped Implementation:** implement Lead-assigned extensions/controls inside allowed paths | Explicit `Allowed Write Paths` + `handoff/reports/TASK-*.md` | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/` |

---

## 2. Token-Efficiency Rules
1. **Targeted CSVs over raw dumps:** Codex roles read compact `results/*/tables/*.csv` (`2–15 KB`) instead of full `summary.json` or gitignored `runs/`.
2. **Tiered compute:** Use `gpt-6.1-terra` (`medium`) for mechanical checks/coding (`validator`, `experiment_worker`) and reserve `gpt-6.1-sol` (`high`) for deep scientific critique and insight discovery (`research_critic`, `results_analyst`).
3. **Strict report caps:** Codex writes concise structured outputs (`<= 80–120` lines) to `handoff/reports/` or `handoff/reviews/`, so the Lead Agent absorbs insights in one small read instead of duplicating exploration tokens.

---

## 3. Standard 4-Step Lead ↔ Codex Workflow

1. **Dispatch:**
   ```bash
   .venv/bin/python scripts/codex_handoff.py status
   .venv/bin/python scripts/codex_handoff.py run --id TASK-001      # runs OpenAI Codex CLI directly if installed
   # or print a copy-paste prompt for another Codex terminal/IDE session:
   .venv/bin/python scripts/codex_handoff.py prompt --id TASK-001
   ```
   *(For concurrent coding tasks alongside Lead edits, isolate via `git worktree add ../mamrl-codex-TASK-XXX -b codex/TASK-XXX`.)*

2. **Verify Scope & Tests:**
   ```bash
   .venv/bin/python scripts/codex_handoff.py check-scope --id TASK-001
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
   ```

3. **Adopt:** Lead reviews `handoff/reviews/` or `handoff/reports/`, integrates approved findings/code, updates `PROGRESS.md` & `handoff/BOARD.md`, and commits.

> **Installation Note for OpenAI Codex CLI:** OpenAI's Codex CLI is the Node package `@openai/codex` (`npm install -g @openai/codex`), **not** the unrelated PyPI `codex` comic-book server (`pip install codex`). `scripts/codex_handoff.py run` automatically checks that the detected `codex` binary is OpenAI Codex CLI before executing.
