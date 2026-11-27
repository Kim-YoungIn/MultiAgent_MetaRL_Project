# Progress Snapshot — 2026-11-27

## Objective & Stage
- **Goal:** Study how meta-learning affects exploration and fast adaptation in multi-agent games (Path A).
- **Deadlines:** 2026-12-17 (5-min video + 3 quiz questions) · 2026-12-20 (5-page NeurIPS report + <5 MB code zip + slides).
- **Stage:** Clean v2 refactor verified on branch `refactor/v2-clean` (1.7 MB tracked footprint; 6/6 tests pass). Ready for GPT review of `docs/SPEC.md` and Exp 1 calibration.

## Completed Work (Verified)
- Tagged full legacy repo (`290dabf`, 15 verified studies) at `archive/v0.9-legacy-studies`; distilled failures into `docs/PRIOR_FINDINGS.md` and preserved key figures/tables in `results/legacy_summary/` (340 KB).
- Rebuilt single-entry rules (`AGENTS.md`), Path-A spec (`docs/SPEC.md`), collaboration guide (`docs/COLLABORATION.md`), and GPT context packer (`scripts/pack_context.py` -> `handoff/context_pack.md`, ~20 KB).
- Implemented clean core package (`src/mamrl/`) and verified `unittest` (6/6 OK in 2.7 s) + pilot run (`configs/pilot_active_explore.json` -> `runs/pilot_active_low.json`): Gate-0 Oracle confirmed `VoI>0 = 100%`, `must_change_opening = 100%` (`mean_VoI = 0.5030`); `greedy` trapped at `+7.99` while `meta_full` reached `+26.24` and `ucb` `+28.31`.

## Active Decisions & Open Issues
- **Pilot diagnostic finding:** With `exploit_miss = -0.70` and `probe_cost = 0.15`, gambling on `EXPLOIT` at turn 0 has higher prior mean (`+0.10`) than `PROBE` (`-0.15`), letting UCB tie/edge Oracle (`+28.31` vs `+27.96`). Calibrating risk/probe structure (e.g., deeper mismatch penalty or multi-step/opponent-contingent probe) before locking Exp 1 is the immediate design decision.
- Legacy code stays ONLY in git tag `archive/v0.9-legacy-studies`; local `runs/` remains untouched (gitignored).

## Next Actionable Step
1. Share `handoff/context_pack.md` with GPT to critique Exp 1 & Exp 2 calibration (`docs/SPEC.md`), especially probe vs. risky-gamble separation and H1 diversity axes.
2. Lock Exp 1 (`Passive` vs. `Active`) and run multi-seed comparisons via `mamrl.runner`.
