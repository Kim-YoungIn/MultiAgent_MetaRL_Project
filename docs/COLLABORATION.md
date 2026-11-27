# Multi-Agent Collaboration Guide (Antigravity-Gemini ↔ Antigravity-Claude ↔ Codex CLI)

This repository uses a **Token-Tiered, Lead-Orchestrated Handoff Architecture** so **Antigravity-Gemini (default Lead)**, **Antigravity-Claude (Co-Design & Analysis)**, and **Codex CLI (`@openai/codex`)** collaborate automatically without file conflicts, spec drift, directory sprawl, or unnecessary token burn. *(Paper/slide writing roles remain deferred until experiments finalize.)*

Entry files (all point to `AGENTS.md` as the single source of truth):
- `AGENTS.md` — Shared project rules & scientific integrity contract.
- `GEMINI.md` — Entry file for Antigravity-Gemini sessions.
- `CLAUDE.md` — Entry file for Antigravity-Claude / `agy` CLI sessions.
- `.codex/config.toml` + `.codex/agents/*.toml` — OpenAI Codex CLI role configs.

---

## 1. 3-Tier Authority & Token-Optimization Matrix

| Tier & Agent / Role | Platform / Compute | Primary Purpose | Allowed Write Scope | Never Modifies |
|---|---|---|---|---|
| **Lead: Antigravity-Gemini** | Antigravity (`GEMINI.md`) | Core execution, orchestration, audit adoption, git commits/pushes | `AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `handoff/tasks/`, `src/`, `configs/`, `tests/`, `results/` | — |
| **Tier 1 (Low-Cost): Codex `validator`** | Codex CLI (`gpt-6.1-terra`, `medium`, `--ephemeral`) | **Contract Verification:** non-leakage, masking, exact DP, CSV/JSON match, `<5 MB` zip | Assigned `tests/` + `handoff/reports/TASK-*.md` | `src/`, `configs/`, `results/`, governance docs |
| **Tier 1 (Low-Cost): Codex `experiment_worker`** | Codex CLI (`gpt-6.1-terra`, `medium`) | **Scoped Implementation:** implement Lead-assigned extensions/controls inside allowed paths | Explicit `Allowed Write Paths` + `handoff/reports/TASK-*.md` | Governance docs, sealed `results/` |
| **Tier 2 (Adversarial): Codex `research_critic`** | Codex CLI (`gpt-6.1-sol`, `high`, `read-only --ephemeral`) | **Validity Defense:** audit statistical verdicts, confounds, overclaiming, OOD bounds | Assigned `handoff/reviews/YYYYMMDD_<topic>.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Tier 2 (Adversarial): Codex `results_analyst`** | Codex CLI (`gpt-6.1-sol`, `high`, `read-only --ephemeral`) | **Discovery & Interpretation:** mine CSVs for mechanisms, anomalies, legacy-to-v3 links | Assigned `handoff/reports/TASK-*.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Tier 3 (Co-Design): Antigravity-Claude `claude_analyst`** | Antigravity `/swarm` or `agy` CLI (`--mode plan --sandbox --effort high`) | **Synthesis & Co-Design:** cross-study mechanism synthesis, hypothesis critique, next-experiment design | Assigned `handoff/reports/TASK-*.md` or `handoff/reviews/*.md` ONLY | `src/`, `configs/`, `tests/`, `results/`, governance docs |
| **Tier 3 (Co-Design): Antigravity-Claude `claude_architect`** | Antigravity `/swarm` (`workspace:"share"`) or `agy` CLI (`--mode accept-edits`) | **Architecture & Prototyping:** scoped method/environment prototyping & deep code review | Explicit `Allowed Write Paths` + `handoff/reports/TASK-*.md` | Governance docs & sealed `results/` (unless promoted to Lead) |

---

## 2. Token-Optimization Rules (Mandatory)
1. **Slim context first (`~5×` smaller):** Generate `handoff/context_pack_slim.md` (`~12 KB`) via `.venv/bin/python scripts/pack_context.py --slim` for routine dispatches (contains `PROGRESS.md`, `handoff/BOARD.md`, and `INTERIM_SYNTHESIS_EXP1_TO_4.md`). Use full `handoff/context_pack.md` (`~76 KB`) only when deep `SPEC.md` / `PRIOR_FINDINGS.md` lookup is required.
2. **Ephemeral fresh-context dispatch:** Always dispatch one-shot audits/reviews in fresh sessions (`codex exec --ephemeral`, `agy -p --disable-slash-commands`, or Antigravity `/swarm` workers) so multi-turn chat history is never re-sent.
3. **Compact CSVs & hard report cap:** Collaborators read `results/*/tables/*.csv` (`2–15 KB`), skip `runs/` and full `summary.json`, and keep reports `<= 120` lines.
4. **Main-session hygiene:** Use `/compact` in the Lead conversation after major milestones, or start a clean session from `PROGRESS.md` + `INTERIM_SYNTHESIS_EXP1_TO_4.md`.

---

## 3. Automated Dispatch (Codex CLI + Antigravity CLI `agy`)

`scripts/codex_handoff.py` dispatches both **Codex CLI** and **Antigravity-Claude (`agy`)** tasks from the same board:

```bash
# 1. Refresh slim context pack (~12 KB) and inspect board
.venv/bin/python scripts/pack_context.py --slim
.venv/bin/python scripts/codex_handoff.py status

# 2. Automated Codex CLI dispatch (read-only tasks auto-save via -o)
.venv/bin/python scripts/codex_handoff.py run --id TASK-004

# 3. Automated Antigravity-Claude dispatch via agy CLI (after 1-time `agy` sign-in)
#    Check available Claude model IDs via: ~/.local/bin/agy models
.venv/bin/python scripts/codex_handoff.py run --id TASK-006 --model <claude-model-id>
.venv/bin/python scripts/codex_handoff.py run --id TASK-006 --model <claude-model-id> --dry-run

# 4. Alternative: Native Antigravity /swarm (zero extra CLI sign-in, isolated workers, zero idle tokens)
#    Invoke /swarm in Antigravity chat and pin a Claude worker to TASK-006.

# 5. Verify scope & unit tests before Lead adoption
.venv/bin/python scripts/codex_handoff.py check-scope --id TASK-006
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

> **One-Time CLI Notes:**
> - **Antigravity CLI (`agy`):** Detected automatically at `~/.local/bin/agy`. Run `~/.local/bin/agy` once interactively in your terminal to sign in, then check model IDs with `~/.local/bin/agy models` (you can also set `AGY_CLAUDE_MODEL=<model-id>`).
> - **OpenAI Codex CLI (`@openai/codex`):** Detected automatically from `PATH` or `~/.nvm/versions/node/*/bin/codex`, with an automatic guard against the unrelated PyPI `codex` package.
