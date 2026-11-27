# TASK-008: Antigravity-Claude Project-Wide Scientific Synthesis & Pre-Freeze Roundtable

- **Task ID:** `TASK-008`
- **Assigned Role:** `claude_analyst` (`CLAUDE.md`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity-Gemini`)

## Objective
Participate in a **3-agent project-wide discussion** (alongside Antigravity-Gemini Lead and Codex CLI `TASK-007`) now that **Block 1 (`Exp 5A` diagnostics)** and **Block 2 (`Exp 4B` Option A: `R3+Distractor` gate validity + Compact 6-Action 3-Layer Stage-Lock `T=48`)** are complete alongside `Exp 1–4`.

Specifically address:
1. **Updating Your `TASK-006` Thesis Against New Evidence (`Exp 5A` + `Exp 4B`):** Which predictions from your `TASK-006` review were confirmed (e.g., Diag 1D `structural_model_mb` hitting `120.625` on 22-action warmup data; Diag 1B surprise-switch failing to beat `prior_only`), and which must be revised in light of `Exp 4B` (6-action Stage-Lock unlocking Tier 3 with `drqn_vanilla` `+67.27`, `drqn_aux_yr` median `+76.25`, `drqn_aux_discovery` allocating `95.3%/87.2%` weight to `(y,r)` and beating `uniform` by `+23.52` `5/5` seeds, and `structural_model_sl` hitting exact `90.10`)?
2. **Minimum Decisive Final Experiment Block (Before ~Dec 8 Freeze):** Given hard deadlines Dec 17 (video+quiz) and Dec 20 (5-page NeurIPS report), what is the exact minimum final control/experiment block (if any beyond Exp 3B permutation-augmented & capacity-matched One-Hot) needed to make our 5-page paper airtight — and what should we explicitly **cut/freeze**?
3. **Top 3 Reviewer Attack Vectors & Structural Fairness Audit:** Check `results/exp5a_gate0_and_surprise/tables/*.csv`, `results/exp4b_compact_multibarrier/tables/*.csv`, and briefly `src/mamrl/study_exp4b_compact.py` (`structural_model_sl` public-stage tracking, `drqn_aux_yr` on `r3_distractor` scoring `10.75` vs Exp 1's `17.52` due to `400` vs `450` iters or aux scale `2.0` vs `3.0`). Flag any hidden flaw or fairness asymmetry!
4. **Paper Narrative Spine (High-Level Only — No Drafting Yet):** In 5 bullet points, outline the cleanest, non-overclaiming scientific arc connecting Part 1 (legacy diagnosis) + Exp 1, 2A/3A/5A, 2B/3B, 4/4B.

## Required Reading (Token-Efficient)
- `AGENTS.md`, `CLAUDE.md`, `PROGRESS.md`, `handoff/BOARD.md`
- `handoff/reports/TASK-006_claude_interim_design_review.md`
- `results/exp5a_gate0_and_surprise/tables/*.csv` (all 5 tiny CSVs)
- `results/exp4b_compact_multibarrier/tables/*.csv` (all 4 tiny CSVs)
- Targeted read of `src/mamrl/study_exp4b_compact.py` as needed (skip `runs/` & full `summary.json`).

## Allowed Write Paths
- `handoff/reports/TASK-008_claude_project_discussion.md`

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `src/`, `configs/`, `tests/`, `results/`).

## Verification & Expected Report Output
1. Write a razor-sharp, unbiased position paper (`<= 120` lines) to `handoff/reports/TASK-008_claude_project_discussion.md`.
2. Do NOT call write_file or edit_file; return ONLY the complete final markdown report directly as your final message.
