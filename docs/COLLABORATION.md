# Multi-Agent Collaboration Guide (GPT / Codex / Antigravity)

This repository is structured so multiple AI agents can collaborate cleanly without creating conflicting docs or duplicate codebases.

## Role Division

| Agent | Primary Role | Writes To |
|---|---|---|
| **GPT (ChatGPT Web / Reviewer)** | Experimental critique, theoretical checks, control design, paper/presentation narrative | Outputs pasted into `handoff/reviews/YYYYMMDD_<topic>.md` |
| **Coding Agent (Antigravity / Codex CLI)** | `src/mamrl/` implementation, unit tests, experiment execution, statistical verification, `PROGRESS.md` updates | `src/`, `configs/`, `tests/`, `results/`, `docs/SPEC.md`, `PROGRESS.md` |
| **Codex Subagents (`.codex/agents/`)** | `research_critic` (validity audit) and `validator` (independent contract tests) | Read-only (or explicitly assigned test files) |

## Standard Sync Loop (3 Steps)

1. **Pack current state for GPT:**
   ```bash
   .venv/bin/python scripts/pack_context.py
   ```
   This produces `handoff/context_pack.md` (~10–20 KB) combining `AGENTS.md`, `PROGRESS.md`, `docs/SPEC.md`, `docs/PRIOR_FINDINGS.md`, the current file tree, and the latest `results/` summary JSONs. Upload or paste that single file into GPT.

2. **Record GPT feedback without polluting active specs:**
   Save GPT's critique or suggestions to `handoff/reviews/YYYYMMDD_<topic>.md`.

3. **Adopt selectively:**
   Review with the coding agent, explicitly decide which items to adopt, update `docs/SPEC.md` (single source of truth), implement in `src/mamrl/` + `configs/`, verify with tests, and update `PROGRESS.md`.
