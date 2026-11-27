# Claude Agent Entry (Antigravity-Claude & `agy` CLI)

Read and follow [AGENTS.md](AGENTS.md) as the single source of project rules.

## Collaboration Protocol (Token-Optimized)
- **Active Roles (writing deferred until experiments finalize):**
  - `claude_analyst` — Cross-study mechanism synthesis, hypothesis/falsification critique, next-experiment co-design (read-only analysis → `handoff/reports/` or `handoff/reviews/`).
  - `claude_architect` — Scoped experimental architecture & algorithm prototyping (disjoint paths or `git worktree`).
- **Orchestration:** Check [handoff/BOARD.md](handoff/BOARD.md), [PROGRESS.md](PROGRESS.md), and [docs/COLLABORATION.md](docs/COLLABORATION.md) before editing files.
- **Parallel Collaborator Mode (default alongside Antigravity-Gemini Lead):**
  - Work strictly on your assigned `handoff/tasks/TASK-XXX_*.md` brief and modify ONLY its `Allowed Write Paths`.
  - Automated dispatch via `agy` CLI (`scripts/codex_handoff.py run --id TASK-XXX --model <claude-model>`) or Antigravity `/swarm` starts with a fresh zero-bloat context; do not edit governance files or sealed `results/`, and do not commit unless promoted to Lead.
  - For concurrent coding alongside another agent, use an isolated git worktree (`git worktree add ../mamrl-claude-<task> -b claude/<task>`).
- **Lead Handoff Mode (when user explicitly switches Lead seat to Claude in Antigravity):**
  - Verify `git status --short` and `handoff/BOARD.md` first, then follow all Lead rules in `AGENTS.md`.
- **Token Efficiency:** Read `handoff/context_pack_slim.md` (~12 KB) and `handoff/reports/INTERIM_SYNTHESIS_EXP1_TO_4.md` first; inspect specific `results/*/tables/*.csv` only as needed; skip `runs/` and full `summary.json`; keep reports `<= 120` lines.
