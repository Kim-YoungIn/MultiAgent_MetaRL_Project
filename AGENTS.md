# Multi-Agent Meta-RL Research — Agent Rules (Single Entry)

All coding and review agents (Antigravity, Codex, GPT, Claude) MUST follow this file.

## 1. Research Goal & Deliverables

- **Core Question:** *멀티에이전트 반복 게임에서 메타 러닝이 탐색(exploration)을 개선하여 새로운 게임·학습 상대에 대한 빠른 적응(fast adaptation)에 어떤 영향을 미치는가?*
- **Active Strategy (Path A):**
  1. **Part 1 (Diagnosed):** Prior studies (`archive/v0.9-legacy-studies`, summarized in `docs/PRIOR_FINDINGS.md`) showed that meta-learned exploration raised *mean* return while lowering the *median* and losing to UCB—because most tasks did not strictly require changing the opening action to gain information, and long-horizon gains were confounded with opponent shaping.
  2. **Part 2 (Active Focus):** Verify when meta-learned exploration genuinely accelerates adaptation using a controlled multi-agent benchmark (`docs/SPEC.md`) where **active costly probing is provably required** (verified by exact Oracle), separating information gain from opponent shaping and task diversity from per-task exposure.
- **Hard Course Deadlines (`docs/course/Evaluation_Plan.pdf`):**
  - **2026-12-17 23:59:** 5-minute presentation video + 3 quiz questions (75% of project grade).
  - **2026-12-20 23:59:** 5-page NeurIPS report + code zip (**≤ 5 MB**) + slides (20% of project grade).

## 2. Single Source of Truth & Anti-Sprawl Rules

Only three living documents govern active work:
1. `AGENTS.md` — These rules (stable).
2. `docs/SPEC.md` — Active experiment contract (environment, agents, controls, metrics).
3. `PROGRESS.md` — Current status, verified results, blockers, next 1 action (≤ 30 lines; rewrite, never append a diary).

**Strict structural rules:**
- **Never create new top-level directories or per-study code folders** (no `research/study_x/`). Extend `src/mamrl/` cleanly and add declarative JSON configs under `configs/`.
- **Legacy code lives ONLY in git tag `archive/v0.9-legacy-studies`.** Do not restore legacy directories into the working tree; consult `docs/PRIOR_FINDINGS.md` first.
- Keep git-tracked files compact so the submission zip stays well under **5 MB**. Heavy outputs and checkpoints go strictly under `runs/` (gitignored).

## 3. Scientific & Engineering Integrity

- **Unbiased Inquiry First:** Stay strictly anchored to the core research question, never to confirming a specific user hypothesis, prior hunch, or agent proposal. Never tune environments solely to force Meta-RL to beat baselines. Pre-register falsification criteria so positive, conditional, and negative/null results are treated with equal scientific value.
- **Gate 0 — Oracle Verification First:** Never run meta-training on a task distribution without first verifying via `mamrl.eval.oracle` what fraction of tasks have positive Value of Information (VoI > 0) and what fraction strictly require changing the opening action away from the greedy/myopic choice.
- **Fair, Unprivileged Baselines:** Never leak privileged task specs (`spec`/payoff tables) into standard baselines (`UCB`, `Greedy`, `Random`, `Memory-Reset`) or `RL²`. Use privileged Bayesian/Oracle models strictly as separate diagnostic ceilings or policy-agnostic trajectory auditors.
- **No Mean-Only Reporting:** Always report **Mean, Median, Win-Rate vs baselines, Early Adaptation AUC, Total Return (including exploration cost), and Counterpart Return** (to detect opponent shaping vs pure information gain).
- **Mandatory Controls & Decomposition:** Compare End-to-End `RL²` (Recurrent PPO+GAE) against **Greedy, Random, Arm-Independent UCB, Memory-Reset (zero-shot prior), Masked-Feedback `RL²`**, and **Oracle** (diagnostic ceiling). Decompose Exploration vs. Exploitation without OOD bias using (1) **Neutral Bayes Info/Value Audit** of collected trajectories, (2) **Prefix-Swap into a Mixed-Prefix Neutral Exploiter**, and (3) **Opponent-Restored vs. Continuous Deployment** evaluation.
- **No Leakage or Silent Overwrites:** Separate development and final test seeds; record config + git commit + seed in every run output; preserve negative/null findings honestly.

## 4. Multi-Agent Collaboration (GPT / Codex / Antigravity)

- Before consulting an external agent (e.g., ChatGPT web), run `.venv/bin/python scripts/pack_context.py` to generate `handoff/context_pack.md`.
- Store external critiques/proposals in `handoff/reviews/YYYYMMDD_<topic>.md`. External suggestions are **proposals until adopted into `docs/SPEC.md`**.
- One writer at a time per file; validate every consequential code change with `pytest`/`unittest` before updating `PROGRESS.md`.
