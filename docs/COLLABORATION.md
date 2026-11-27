# Multi-Agent Collaboration Guide (Antigravity-Gemini ↔ Antigravity-Claude ↔ Codex CLI)

This repository uses a **Lead-Orchestrated Handoff Architecture** so **Antigravity-Gemini (default Lead)**, **Antigravity-Claude (Co-Lead / Collaborator)**, and **Codex CLI (`@openai/codex`)** collaborate cleanly without file conflicts, spec drift, directory sprawl, or unnecessary token burn.

Entry files (all point to `AGENTS.md` as the single source of truth):
- `AGENTS.md` — Shared project rules & scientific integrity contract.
- `GEMINI.md` — Entry file for Antigravity-Gemini sessions.
- `CLAUDE.md` — Entry file for Antigravity-Claude (and Claude Code) sessions.
- `.codex/config.toml` + `.codex/agents/*.toml` — OpenAI Codex CLI role configs.

---

## 1. Authority & Write-Ownership Matrix (Token-Tiered)

| Agent / Role | Platform / Model | Primary Purpose | Allowed Write Scope | Never Modifies |
|---|---|---|---|---|
| **Antigravity-Gemini (Lead)** | Antigravity (`GEMINI.md`) | Architecture, core experiments, audit adoption, git commits/pushes | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `handoff/tasks/`, `src/`, `configs/`, `tests/`, `results/` | — |
| **Antigravity-Claude `claude_architect`** | Antigravity Claude (`CLAUDE.md`) | Experimental design co-architecture, algorithm prototyping, deep code review | Explicit `Allowed Write Paths` + `handoff/reports/TASK-*.md` | Governance docs & sealed `results/` (unless promoted to Lead) |
| **Antigravity-Claude `claude_analyst`** | Antigravity Claude (`CLAUDE.md`) | Cross-study synthesis, mechanism interpretation, hypothesis/falsification critique | Assigned `handoff/reports/TASK-*.md` or `handoff/reviews/*.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Antigravity-Claude `claude_writer`** | Antigravity Claude (`CLAUDE.md`) | NeurIPS 5-page report, presentation script, slides & quiz drafting (when ready) | Assigned draft paths + `handoff/reports/TASK-*.md` | `src/`, `configs/`, `tests/`, sealed `results/`, governance docs |
| **Codex `research_critic`** | Codex CLI (`gpt-6.1-sol`, `high`) | **Validity Defense:** audit statistical verdicts, confounds, overclaiming, OOD bounds | Assigned `handoff/reviews/YYYYMMDD_<topic>.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Codex `results_analyst`** | Codex CLI (`gpt-6.1-sol`, `high`) | **Discovery & Interpretation:** mine CSVs for mechanisms, anomalies, legacy-to-v3 links | Assigned `handoff/reports/TASK-*.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Codex `validator`** | Codex CLI (`gpt-6.1-terra`, `medium`) | **Contract Verification:** non-leakage, exact DP, disjoint splits, CSV/JSON match, `<5 MB` zip | Assigned `tests/` + `handoff/reports/TASK-*.md` | `src/`, `configs/`, `results/`, governance docs |
| **Codex `experiment_worker`** | Codex CLI (`gpt-6.1-terra`, `medium`) | **Scoped Implementation:** implement Lead-assigned extensions/controls inside allowed paths | Explicit `Allowed Write Paths` + `handoff/reports/TASK-*.md` | Governance docs, sealed `results/` |

---

## 2. Token-Efficiency Rules
1. **Targeted CSVs over raw dumps:** Collaborators read compact `results/*/tables/*.csv` (`2–15 KB`) and `handoff/context_pack.md` instead of full `summary.json` or gitignored `runs/`.
2. **Tiered compute:** Use `Codex validator / experiment_worker` (`gpt-6.1-terra`, `medium`) for mechanical checks/coding, `Codex research_critic / results_analyst` (`gpt-6.1-sol`, `high`) for independent adversarial audits, and **Antigravity-Claude** for nuanced synthesis, experimental co-design, and clear scientific writing.
3. **Strict report caps:** Collaborators write concise structured outputs (`<= 80–120` lines) to `handoff/reports/` or `handoff/reviews/`, so the Lead integrates insights in one small read.

---

## 3. Standard Lead ↔ Codex CLI Workflow

1. **Dispatch:**
   ```bash
   .venv/bin/python scripts/codex_handoff.py status
   .venv/bin/python scripts/codex_handoff.py run --id TASK-004      # runs OpenAI Codex CLI directly
   .venv/bin/python scripts/codex_handoff.py prompt --id TASK-004   # prints copy-paste prompt
   ```
   *(For concurrent coding tasks alongside Lead edits, isolate via `git worktree add ../mamrl-codex-TASK-XXX -b codex/TASK-XXX`.)*

2. **Verify Scope & Tests:**
   ```bash
   .venv/bin/python scripts/codex_handoff.py check-scope --id TASK-004
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
   ```

3. **Adopt:** Lead reviews `handoff/reviews/` or `handoff/reports/`, integrates approved findings/code, updates `PROGRESS.md` & `handoff/BOARD.md`, and commits.

---

## 4. Antigravity-Claude Collaboration Workflow

Antigravity-Claude can participate in two clean modes:

### Mode A — Parallel Collaborator (Second Antigravity Chat/Tab Running Claude)
1. Refresh the shared context pack: `.venv/bin/python scripts/pack_context.py`
2. Create or pick a scoped Claude task brief (`role` in `{claude_architect, claude_analyst, claude_writer}`):
   ```bash
   .venv/bin/python scripts/codex_handoff.py new-task --id TASK-006 --role claude_analyst \
       --title "Interim synthesis and Exp 4 follow-up design" \
       --allow "handoff/reports/TASK-006_claude_interim_design_review.md"
   .venv/bin/python scripts/codex_handoff.py prompt --id TASK-006
   ```
3. Paste the generated prompt into the **Antigravity-Claude** conversation. Claude writes ONLY to its `Allowed Write Paths` (or an isolated `git worktree add ../mamrl-claude-TASK-XXX -b claude/TASK-XXX` if editing code).
4. Lead verifies via `check-scope --id TASK-006`, reviews the report, and adopts approved changes.

### Mode B — In-Place Lead Model Handoff (Switching Gemini ↔ Claude in Antigravity)
1. Before switching models, run `.venv/bin/python scripts/pack_context.py`, ensure `PROGRESS.md` and `handoff/BOARD.md` reflect the latest state, and leave a clean `git status`.
2. In the new Claude session, Antigravity automatically loads `AGENTS.md` and `CLAUDE.md`; start with *"Read `PROGRESS.md` and `handoff/BOARD.md` and continue from the current checkpoint."*

> **Installation Note for OpenAI Codex CLI:** OpenAI's Codex CLI is the Node package `@openai/codex` (`npm install -g @openai/codex`), **not** the unrelated PyPI `codex` comic-book server (`pip install codex`). `scripts/codex_handoff.py run` automatically guards against the PyPI binary.
