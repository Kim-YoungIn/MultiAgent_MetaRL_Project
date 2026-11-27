# Multi-Agent Meta-Learning and Adaptation

Can an agent select experiences that help it adapt to an unfamiliar game and a learning opponent? The project compares zero-shot performance, the value of experience after a fixed exploration budget, and total reward including the cost of exploration.

## Current research

The latest study evaluates 36 previously trained blocks on 2,304 fresh structured games. Lowering the probability that joint investment changes the state did little to alter the measured advantage of feedback-conditioned exploration over a matched masked Teacher. It reduced the 96-turn total-return Teacher's advantage over Student-only, particularly in reciprocity games. [Results, family comparisons and reproduction](research/dynamics_transfer/REPORT.md) · [Protocol](research/dynamics_transfer/PROTOCOL.md)

| To find | Start here |
|---|---|
| Current stage, open question and next action | [PROGRESS.md](PROGRESS.md) |
| Project-wide protocols, results, evidence, local data and run status | [Study registry](research/project_control/studies.json) |
| Teacher, Student, opponent, implementation and training commands | [Fast Exploration architecture](research/fast_exploration/architecture/README.md) |
| Earlier Fast Exploration results | [36-block v2](research/fast_exploration/results/v2/README.md) · [24-block v1](research/fast_exploration/results/README.md) · [v1 by seed](research/fast_exploration/results/SEED_RESULTS.md) |
| Structured game rules and simulator | [Game guide](STRUCTURED_GAMES.md) · [Environment](05_toy_models/mamrl/structured_games.py) |
| Historical recurrent exploration study, all 375 fits and 175 home evaluation cells completed | [D-031 results and verification](research/d031_completion/README.md) · [Diversity contrasts](research/d031_completion/evidence/diversity.png) |

## Managing and reproducing studies

Run from the repository root. The registry command uses the Python standard library. The reports link to each study's numerical dependencies and data-generation commands.

```bash
python3 -m research.project_control --check
python3 -m research.project_control --study dynamics-transfer-v1 --check
```

The first command checks the versioned evidence paths across studies and, when the corresponding local runs exist, compares selected receipts or manifests with versioned evidence. The second shows paths and checks for one study. Full trajectory and statistical replay are documented in each report. Raw trajectories, historical bundles and checkpoints live in the local `runs/` tree.

On resuming work, read [PROGRESS.md](PROGRESS.md), then open the study's registry entry and verify the evidence needed for the decision. For a new evaluation, add one registry entry, record its protocol and primary endpoint before inspecting the panel, and write to a distinct raw directory. If work stops halfway, use the study's receipts to recover completed units. When a study finishes, link its checked report and compact evidence in that entry and replace the progress snapshot. Historical contracts and opened panels retain their identities. [Project guidance](AGENTS.md)

## Scientific scope

- **H1, training diversity.** How does variation in training rules and opponent policies or learning dynamics affect adaptation? Task count, exposure per task, training compute and diversity can have different effects.
- **H2, shared structure.** Which reusable interaction patterns explain transfer? Rule syntax and numerical payoff similarity are candidate predictors. Their relation to strategy and information use must be measured.
- **Multi-agent adaptation.** Both agents' actions influence incentives. Controlled comparisons retain only the learner's experience or restore the opponent state. Continuous deployment measures what happens when both agents keep learning.

Earlier studies cover [structured-game and cross-context transfer](research/REPORT.md), [similarity of previous rule catalogs](research/rule_similarity/REPORT.md), [learning opponents](research/opponent_adaptation/REPORT.md) and [absolute exploration objectives](research/absolute_exploration/REPORT.md). The [registry](research/project_control/studies.json) indexes their implementations and verification evidence through the corresponding study guides.

The original proposal and course requirements are in [01_originals](01_originals/). Historical decisions, references, ideas, designs and implementations are in [00_management](00_management/), [02_references](02_references/), [03_ideas](03_ideas/), [04_design](04_design/) and [05_toy_models](05_toy_models/). The original D-031 entry point is `python -m experiment`. A historical v0.5 results bundle is available locally at `runs/archive/runs-v0.5.zip`; a clone containing the original commit can recover it with `git show ea2a646:runs.zip > runs-v0.5.zip`.
