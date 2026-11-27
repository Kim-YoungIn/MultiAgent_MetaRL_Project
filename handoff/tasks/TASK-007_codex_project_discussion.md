# TASK-007: Codex CLI Project-Wide Scientific Synthesis & Pre-Freeze Roundtable

- **Task ID:** `TASK-007`
- **Assigned Role:** `research_critic` (`.codex/agents/research_critic.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity-Gemini`)

## Objective
Participate in a **3-agent project-wide discussion** (alongside Antigravity-Gemini Lead and Antigravity-Claude `TASK-008`) now that **Block 1 (`Exp 5A` diagnostics)** and **Block 2 (`Exp 4B` Option A: `R3+Distractor` gate validity + Compact 6-Action 3-Layer Stage-Lock `T=48`)** are complete alongside `Exp 1–4`.

Specifically address:
1. **Unifying Scientific Answer (Exp 1 → Exp 4B/5A):** How do the new `Exp 5A` (especially Diag 1D `structural_model_mb` hitting `120.625` on 22-action warmup data, Diag 1A PBVI `VoI=+7.20` at $d=0.6$, Diag 1B surprise-switch falsification, Diag 1C Exp 2B/3B Gate-0 VoI) and `Exp 4B` (`drqn_aux_discovery` allocating 95%/87% to `(y,r)`, `+23.52` over uniform aux, and 6-action Stage-Lock unlocking Tier 3 at `67.27`/`64.20` while `structural_model_sl` hits exact `90.10`) refine or falsify our core thesis?
2. **Minimum Decisive Final Experiment Block (Before ~Dec 8 Freeze):** With hard deadlines Dec 17 (video+quiz) and Dec 20 (5-page NeurIPS report), what is the single highest-value remaining control/experiment block (e.g., Exp 3B permutation-augmented & capacity-matched One-Hot controls, or anything else) — and what should we explicitly **STOP** doing?
3. **Hardest Reviewer Attack Vectors:** Identify the top 3 sharpest critiques a skeptical NeurIPS reviewer would make against our complete Exp 1–5A story, plus exact pre-emptive defenses or framing adjustments.
4. **Constructive Disagreement / Audit of New Results:** Inspect `results/exp5a_gate0_and_surprise/tables/*.csv`, `results/exp4b_compact_multibarrier/tables/*.csv`, and `src/mamrl/envs/stage_lock.py` (briefly) for any subtle flaw, fairness caveat (e.g., `structural_model_sl`'s public stage-tracking heuristic vs DRQN), or disagreement with Claude's `TASK-006` thesis.

## Required Reading (Token-Efficient)
- `AGENTS.md`, `PROGRESS.md`, `handoff/BOARD.md`
- `handoff/reports/TASK-006_claude_interim_design_review.md` (Claude's prior framing)
- `results/exp5a_gate0_and_surprise/tables/*.csv` (all 5 tiny CSVs)
- `results/exp4b_compact_multibarrier/tables/*.csv` (all 4 tiny CSVs)
- Targeted inspection of `src/mamrl/study_exp4b_compact.py` / `src/mamrl/study_diagnostics_v3.py` as needed (skip `runs/` & full `summary.json`).

## Allowed Write Paths
- `handoff/reviews/20261128_codex_project_discussion.md`

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `src/`, `configs/`, `tests/`, `results/`).

## Verification & Expected Report Output
1. Write a razor-sharp, unbiased position paper (`<= 120` lines) to `handoff/reviews/20261128_codex_project_discussion.md`.
2. Do NOT edit code or run `git commit`; leave synthesis & adoption to the Lead Agent.
