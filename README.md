# Multi-Agent Meta-Learning and Adaptation

Can agents use a few interactions to adapt to unfamiliar rules and learning opponents? This project measures what exploration teaches the agent, how quickly that changes its decisions, and what the experience costs in reward.

## Start here

## Completed adaptation experiments

**Fast Exploration at 24 independent blocks.** The scaled Teacher–Student protocol compares learned, belief-blind, random and greedy exploration with K=0/4/8/16/24 and 32 assessment turns. The runnable entry point, benchmarks and artifact indexes are in the [scaled study guide](research/fast_exploration_v1/README.md).

**Absolute reward after exploration.** On 6,144 new structured games the cumulative-return opening beat both a gain-only target and an immediate-reward target. Its advantage in choosing a more useful numerical reward signal was unresolved when the game and opponent state were restored. [Study and reproduction](research/absolute_exploration/REPORT.md)

**Rule similarity and adaptation.** Compare nearest training-game similarity with paired adaptation gains in D-031 and in 3,072 structured test games. The historical full-budget continuation and its offline source restoration are tracked separately. [Similarity report](research/rule_similarity/REPORT.md) · [D-031 continuation](research/d031_completion/README.md)

**Direct comparison against a learning counterpart.** Across 3,072 fresh structured games, updating from eight interactions raised payoff for both the meta-trained and conventionally pretrained agents. Their absolute difference against the updating opponent was −0.00015, with a 97.5% interval of [−0.00312, +0.00271]. The paired test also found no resolved change in their relative update benefit when the counterpart's own learning was enabled. Ordinary pretraining yielded higher descriptive 24-turn deployment return. [Research report, data and reproduction](research/opponent_adaptation/REPORT.md)


A learned opponent prior improved decisions in an unvisited arena after four observations when incentives were correlated across arenas. The same prior hurt when that correlation reversed. In the supplied structured-game extension, a scratch learner gained more over the first eight probes than the learned reward prior. A fresh cross-size experiment found a small eight-probe advantage over pooled pretraining when transferring from three to six actions. [Integrated report and figures](research/REPORT.md)

The new studies use independent training catalogs, common-history queries against a continuing learning opponent, and separate live deployments. The extension experiments run the actual resource, sequence, reciprocity and mixed games. A two-turn exact decision calculation measures the value of observing reward and the cost of forcing a greedy opening. [Reproduction and raw-data guide](research/REPRODUCE.md) · [Architecture comparison](research/adapt_v1/DESIGN_REVIEW.md) · [D-031 evidence audit](research/adapt_v1/HISTORICAL_AUDIT.md)

## Scientific questions

**Central question:** Does meta-learning enable more effective adaptation in multi-agent settings, particularly when an agent must learn from a limited amount of experience with new rules or opponents? Under what conditions does that advantage appear, disappear, or reverse?

An initially strong policy is not necessarily a rapidly adapting policy. A useful evaluation must distinguish zero-shot performance, learning from additional experience, training-time priors, and the cost of adaptation. Adaptation may arise through recurrent state, explicit inference, parameter updates, learned exploration, or other mechanisms.

The project began with two related hypotheses:

- **H1 — Training diversity and adaptation.** How do differences in training rules, game families, opponents, and their learning dynamics affect adaptation to unseen interactions? More diversity is not assumed to be beneficial. The number of tasks, strategic variety, total training budget, and exposure per task should not be conflated. **This is the historical first research priority.**
- **H2 — Shared structure and transfer.** Can an agent learn reusable strategic, relational, or information-gathering structure that transfers across games? Similar rule syntax or payoff tables are not sufficient evidence of reusable knowledge. **This was reserved for possible expansion after evaluating H1 and the underlying adaptation mechanism.**

These are research hypotheses, not immutable specifications. A substantially better formulation can replace them if justified and recorded. Genuine multi-agent strategic dependence matters; opponent learning was an important original setting. Deliberately inducing an opponent to learn (opponent shaping) was not the central objective.

## Research freedom and methodology

No single learning algorithm, agent decomposition, game representation, opponent model, or evaluation method is prescribed. In particular, the current Teacher–Student split, PPO/Q-learning, fixed reward tensors, and D-031 protocol may be re-examined or replaced.

A new research effort should first compare theoretically credible approaches and design experiments able to distinguish their explanatory claims. This includes principled baselines, controlled access to task information, explicit train/development/final separation, independent random seeds, and appropriate treatment of adaptive opponents. Avoid interpreting an improvement from greater prior knowledge, compute, or exposure as proof of faster adaptation.

Changing the scientific design is allowed; changing the recorded conditions of an already completed experiment is not. Substantial revisions require their own protocol, version, and reproducible outputs. Negative and inconclusive findings are valid results.

## Repository layout

| Path | Purpose |
|---|---|
| Current status and next action | [PROGRESS.md](PROGRESS.md) |
| Latest study | [Fast Exploration](research/fast_exploration/README.md) |
| Architecture and reproduction | [Model, source map and commands](research/fast_exploration/architecture/README.md) · [v2 protocol](research/fast_exploration/architecture/V2_PROTOCOL.md) |
| Experiment results and plots | [v2 36-block and fixed-budget results](research/fast_exploration/results/v2/README.md) · [v1 by master seed](research/fast_exploration/results/SEED_RESULTS.md) |
| Simulator and game rules | [Structured games guide](STRUCTURED_GAMES.md) · [Simulator code](05_toy_models/mamrl/structured_games.py) |

## Work on the current study

Run from the repository root with the Python environment in the [study guide](research/fast_exploration/architecture/README.md). Use a new output directory for a new run.

```bash
export PYTHONPATH=05_toy_models:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -p 'test_fast_exploration*.py'
.venv/bin/python -m research.fast_exploration_v1.scale --split dev --output runs/repro_fast_dev --workers 2 --device cpu
```

The current Teacher learns to collect experiences for a Bayesian linear Student. The evaluation pairs methods on the same unseen games and opponent streams. Run outputs live under `runs/`; small evidence tables and figures live with the respective study. Original commands and source paths are part of each study's reproduction contract.

## Earlier studies

| Study | Evidence and reproduction |
|---|---|
| D-031, recurrent exploration in Memory-One games | [Continuation and completion status](research/d031_completion/README.md) · [Original protocol](04_design/Adaptation_Experiment.md) |
| Rule similarity and transfer | [Correlation results](research/rule_similarity/REPORT.md) |
| Adaptation against fixed and updating opponents | [Direct meta-training comparison](research/opponent_adaptation/REPORT.md) |
| Exploration objectives and the value of reward feedback | [Absolute-return opening](research/absolute_exploration/REPORT.md) · [Reward stakes](research/probe_stakes/REPORT.md) · [Two-turn planning](research/active_probe/REPORT.md) |
| Structured-game, cross-size and cross-context transfer | [Integrated report](research/REPORT.md) · [Reproduction](research/REPRODUCE.md) |

The original proposal and course requirements are in [01_originals](01_originals/). Historical decisions, literature, abandoned ideas and experiment contracts remain in [00_management](00_management/), [02_references](02_references/), [03_ideas](03_ideas/) and [04_design](04_design/). The D-031 entry point remains `python -m experiment`. Use [AGENTS.md](AGENTS.md) for research and session guidance.

The v0.5 results bundle is retained locally as `runs/archive/runs-v0.5.zip`. To retrieve the exact bundle from a clone with the original commit, run `git show ea2a646:runs.zip > runs-v0.5.zip` from the root. The original file manifest is a historical snapshot.
