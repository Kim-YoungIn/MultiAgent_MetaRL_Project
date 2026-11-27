# Claude Agent Entry (Antigravity-Claude & Claude Code)

Read and follow [AGENTS.md](AGENTS.md) as the single source of project rules.

## Collaboration Protocol
- **Orchestration:** Check [handoff/BOARD.md](handoff/BOARD.md), [PROGRESS.md](PROGRESS.md), and [docs/COLLABORATION.md](docs/COLLABORATION.md) before editing files.
- **Parallel Collaborator Mode (default alongside Antigravity-Gemini Lead):**
  - Work strictly on your assigned `handoff/tasks/TASK-XXX_*.md` brief (`claude_architect`, `claude_analyst`, `claude_writer`) and modify ONLY its `Allowed Write Paths`.
  - Deliver analyses/proposals to `handoff/reports/` or `handoff/reviews/` (do not edit `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, or sealed `results/`, and do not commit unless promoted to Lead).
  - For concurrent coding alongside another agent, use an isolated git worktree (`git worktree add ../mamrl-claude-<task> -b claude/<task>`).
- **Lead Handoff Mode (when user explicitly switches Lead seat to Claude in Antigravity):**
  - Verify `git status --short` and `handoff/BOARD.md` first to ensure no active worker conflicts, then follow all Lead rules in `AGENTS.md`.
- **Token Efficiency:** Prefer compact `results/*/tables/*.csv` and `handoff/context_pack.md` over raw `runs/` or full `summary.json`.
