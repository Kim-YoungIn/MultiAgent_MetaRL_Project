# TASK-006: Antigravity-Claude Second-Opinion Interim Review & Next-Experiment Co-Design

- **Task ID:** `TASK-006`
- **Assigned Role:** `claude_analyst` (`CLAUDE.md`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity-Gemini`)

## Objective
Provide an independent **Antigravity-Claude** second-opinion synthesis of our interim checkpoint (Exp 1, 2A, 2B, 3A, 3B, 4 + Codex audits `TASK-001..005`) and co-design the highest-leverage next experiment before our final experimental freeze. Note: this is an **interim checkpoint**, not a final paper write-up.

Specifically address:
1. **Unified Mechanism Synthesis:** Across Exp 1–4, what is the sharpest, most falsifiable unifying thesis answering the core research question (*how meta-learning affects exploration & fast adaptation in multi-agent repeated games*) without overclaiming?
2. **Breaking Exp 4's Tier-0 Scaling Wall & Separating the Catch-22:** Compare concrete unprivileged interventions (e.g., episodic `(q,a)` count bonus, prediction-error curiosity, barrier curriculum, or testing bilevel `aux_discovery` on `R3` + noise distractors) — which single controlled design best separates *"exploration trap"* from *"auxiliary task discovery failure"*?
3. **Addressing Exp 2A/3A's OOD Noise Overconfidence:** Evaluate whether online self-supervised surprise (`-log p(y_t | h_t)`) fed into the GRU is a clean, unprivileged test for adaptive fallback under unseen `Expected SARSA` and $d=0.6$.
4. **Blind Spots:** Flag any missing baseline, confound, or alternative explanation not yet caught by Codex `TASK-004`/`TASK-005`.

## Required Reading (Token-Efficient)
- `AGENTS.md`, `CLAUDE.md`, `PROGRESS.md`, `handoff/BOARD.md`
- `handoff/reports/INTERIM_SYNTHESIS_EXP1_TO_4.md` (Lead interim synthesis)
- `handoff/reviews/20261127_exp3_4_interim_audit.md` (`TASK-004`)
- `handoff/reports/TASK-005_exp3_4_contract_audit.md` (`TASK-005`)
- Compact `results/*/tables/*.csv` only as needed (skip `runs/`).

## Allowed Write Paths
- `handoff/reports/TASK-006_claude_interim_design_review.md`

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `src/`, `configs/`, `tests/`, sealed `results/`).

## Verification & Expected Report Output
1. Write a concise, unbiased review (`<= 120` lines) to `handoff/reports/TASK-006_claude_interim_design_review.md`.
2. Do NOT edit code or run `git commit`; leave adoption to the Lead Agent.
