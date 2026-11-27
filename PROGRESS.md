# Research status · 2026-11-27

## Current objective
Measure whether a meta-trained explorer uses eight interactions to collect more useful experience when the rules of a multiplayer game change. The latest study varies joint-investment reliability and a temporal prerequisite across four game families.

## Completed and verified
- Dynamics transfer v1 used the 36 frozen v2 training blocks, 2,304 fresh base games, six paired variants and two resets. Protocol: `research/dynamics_transfer/PROTOCOL.md`.
- The predeclared change in the full-minus-masked Teacher benefit was −0.000118 per assessment turn, 95% paired interval [−0.001682, +0.001424]. See `research/dynamics_transfer/evidence/summary.json`.
- At 96 turns, reducing success probability by .6 changed the total-return Teacher’s advantage over Student-only by −.399 [−.625, −.166]. Teacher versus UCB remained unresolved.
- Source and output hashes, role/family sampling and 864 full scalar trajectories were checked; the primary estimate and interval were independently reproduced. See `research/dynamics_transfer/evidence/verification.json`.
- Scripted exposure diagnostics and raw tapes are in `runs/dynamics_transfer_v1/scripted_exposure/`. Full game, action and reward tapes live in the parent run. Scripts measure how often an intervention becomes observable under specified play.
- Earlier v1 (24 trained blocks), v2 (36 trained blocks), and the 96-turn extension are indexed in `research/fast_exploration/README.md`. H1 training diversity and optimal information acquisition remain open.

## Active work
- Historical D-031 resume uses its sealed Python 3.14.4 / torch 2.15.0+cu132 CPU execution. All 25 primary pairs were previously completed. The local environment was restored separately from v2.
- Inspect current progress using `.venv/bin/python -m research.d031_completion.status runs/d031_completion_v2/study`; active log is `runs/d031_completion_v2/continuation_20261127.log`.
- Project-wide study status and evidence paths are in `research/project_control/studies.json`. Check using `.venv/bin/python -m research.project_control --check`.

## Next action
Verify D-031 completion receipts and extension contrasts, then link the final report and reproduction commands here.
