# Multi-Agent Meta-RL: Meta-Learned Exploration for Fast Adaptation

> **Research Question:** 멀티에이전트 반복 게임 환경에서 메타 러닝이 탐색(exploration)을 개선하여 새로운 게임 규칙과 학습 상대에 대한 빠른 적응(fast adaptation)에 어떤 영향을 미치는가?

## Research Overview (Path A)

This project combines two tightly linked parts toward the final course deliverables (**Dec 17** 5-min presentation + quiz; **Dec 20** 5-page NeurIPS report + `<5 MB` code):

1. **Part 1 — Why Standard Multi-Agent Meta-Exploration Fails ([docs/PRIOR_FINDINGS.md](docs/PRIOR_FINDINGS.md)):**
   Across 15 completed historical studies (preserved at git tag `archive/v0.9-legacy-studies`), meta-learned exploration increased *mean* 96-turn return (`+0.446 / +0.428`) while *decreasing* the median (`−0.144 / −0.142`, win-rate 40%) and losing to UCB. Exact 2-turn Oracle analysis revealed why: only **6/96 (6.3%)** game pairs strictly required changing the opening action to obtain useful information, while long-horizon gains were confounded with opponent shaping (`+0.365` counterpart return).
2. **Part 2 — Oracle-Verified Active Exploration Study ([docs/SPEC.md](docs/SPEC.md)):**
   Using a clean, single-package benchmark (`src/mamrl/`), we directly contrast **Passive** vs. **Active** probing necessity (verified by exact Oracle before training), isolate **feedback conditioning** (`Full` vs. `Masked`), separate **information gain** from **opponent shaping** (restored vs. continuous evaluation), and test **H1 task diversity** under both fixed total budget and fixed per-task exposure.

## Repository Layout

```text
AGENTS.md               # Single entry rules for all AI agents (Antigravity, Codex, GPT)
PROGRESS.md             # Concise live status snapshot (<= 30 lines)
docs/
  SPEC.md               # Active Path-A experiment contract
  PRIOR_FINDINGS.md     # Concise synthesis of legacy v0.1-v0.9 (tag: archive/v0.9-legacy-studies)
  COLLABORATION.md      # Workflow for GPT / Codex / Antigravity handoffs
  REFERENCES.md         # Bibliography for the final report
  course/               # Proposal, guideline, evaluation plan
src/mamrl/              # Clean core library (envs, agents, eval, runner)
configs/                # Declarative JSON experiment configs
scripts/pack_context.py # One-command context bundle generator for GPT/external agents
results/                # Compact git-tracked figures & JSON/CSV summaries (< 5 MB)
handoff/                # Context pack & external agent reviews
tests/                  # Fast invariant & contract tests
runs/                   # Gitignored local raw trajectories & checkpoints
```

## Quickstart

```bash
# 1. Run core invariant & Oracle verification tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

# 2. Run the Path-A pilot experiment (Passive vs. Active Oracle check + baseline/meta comparison)
PYTHONPATH=src .venv/bin/python -m mamrl.runner --config configs/pilot_active_explore.json

# 3. Generate a single-file context pack for GPT / external review
.venv/bin/python scripts/pack_context.py
```

## Accessing Legacy Studies

All pre-refactor code and 15 full reports remain immutable in git tag `archive/v0.9-legacy-studies`:

```bash
git show archive/v0.9-legacy-studies:research/retrospective/REPORT.md
```
