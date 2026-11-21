# Multi-Agent Meta-Learning and Adaptation

Research repository for investigating **how meta-learning affects agents' adaptation to unfamiliar games, opponents, and strategic interactions**. The aim is to identify mechanisms and conditions of adaptation through well-founded, reproducible experiments—not to optimize a predetermined model or preserve an existing implementation.

> **Research scope is broader than the current code.** The repository contains Teacher–Student meta-exploration, recurrent PPO, tabular online learners, Memory-One games, and the D-031 study. These are historical experimental approaches, not mandatory components of future research.

## Completed adaptation experiments

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
| `PROGRESS.md` | Short, current session handoff: status, evidence, unresolved issues, next action |
| `AGENTS.md` | Research-agent principles, autonomy, and progress-update rules |
| `01_originals/` | Original proposal and course project guidance |
| `02_references/` | Literature and reference index |
| `03_ideas/` | Earlier ideas, critiques, and unadopted alternatives |
| `00_management/` | Historical decisions, implementation status, questions, and research notes |
| `04_design/` | Contracts for past and current experiment versions |
| `05_toy_models/mamrl/` | Existing simulator, generators, learners, and diagnostics |
| `experiment/` | D-031 training, evaluation, statistics, and reporting implementation |
| `research/` | Completed cross-context, structured-game and cross-size studies, exact oracle, figures and verification |
| `tests/` | Unit, interface, and experiment-contract tests |
| `runs/` | Local experiment artifacts, normally excluded from Git |

Read `PROGRESS.md` first when continuing work, then inspect the actual source, protocol, and artifacts that support the current task. A historical document's “FIXED” designation applies to its original experiment unless explicitly reaffirmed for a new design.

## Existing implementation: D-031 (historical study)

D-031 tests whether exploration trained to improve a downstream learner's experience-conditioned performance differs from reward-oriented pretraining. It uses simultaneous two-player Memory-One games with logical payoff rules, online learning opponents, and the following methods:

| Method | Implemented training objective |
|---|---|
| **M** | Difference between downstream returns with retained and erased student Q state (`G`) |
| **P** | Explorer's own mean reward during the exploration phase |
| **O** | M's objective with part of the explorer's observations masked |

A recurrent context-based actor gathers experience for a freshly initialized online Q-learning student; opponents use online Q-learning or Expected SARSA. Evaluation measures downstream performance across exploration budgets and compares adaptation-area estimates. **`G` is not the same as absolute downstream return or pure information gain.** The student and opponent can both continue learning during evaluation.

The authoritative protocol for that experiment is [`04_design/Adaptation_Experiment.md`](04_design/Adaptation_Experiment.md). The corresponding historical status is described in [`00_management/Implementation_Status.md`](00_management/Implementation_Status.md). The earliest completed full-budget cell did not establish superiority of M over P; remaining cells must not be presented as completed without evidence.

### Running the existing D-031 implementation

Run from the repository root on WSL/Linux with a compatible Python environment and PyTorch/NumPy installed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt

# Inspect the existing protocol, its methods, and compute budget.
.venv/bin/python -m experiment plan

# Small development-only integration run.
.venv/bin/python -m experiment run --dev --block all --output runs/my_d031_dev

# Reproduce the original primary protocol in a new output directory.
.venv/bin/python -m experiment prepare --output runs/my_d031
.venv/bin/python -m experiment run --block primary --output runs/my_d031

# Summarize completed outputs and run the repository tests.
.venv/bin/python -m experiment report --output runs/my_d031
PYTHONPATH=05_toy_models .venv/bin/python -m unittest discover -s tests -v
```

The implemented CLI supports `--block` (`primary`, `rules`, `opponents`, `families`, `actions`, `all`), `--catalog`, `--replica`, `--device cpu|cuda`, and `--defer-report`. Existing matched checkpoints can resume under the same recorded code and runtime configuration. Results are stored under `runs/<study>/` in `summary.md`, `report.json`, `panels/`, `cells/`, and per-trial artifacts.

The existing experimental design fixes 800 updates × 32 trials, K=H=24, five catalogs and five training replicas, and specific evaluation repetitions in `experiment/protocol.py`. **These are implementation defaults and limits, not scientifically justified maxima.** The CLI's `--catalog` and `--replica` select from fixed ranges; they do not increase those ranges. `--dev` runs a distinct short development protocol rather than an arbitrary-size configuration.

Earlier implementations remain accessible through `PYTHONPATH=05_toy_models python -m mamrl ...` and `python -m mamrl.meta_exploration ...`. For older commands and their exact defaults, see the [v0.5 README snapshot](https://github.com/Kim-YoungIn/MultiAgent_MetaRL_Project/blob/86386cebb75f95b5867f694d927491bc78c71f8c/README.md) and the associated experiment contracts.

## Scaling and redesigning experiments

**Training scale, number and diversity of tasks, action spaces, model capacity, evaluation horizons, replication, and computational infrastructure are part of experiment design.** If stronger evidence requires larger or qualitatively different experiments, extend or replace the implementation rather than treating the old protocol as a cap.

A new runner or research pipeline may need to support experiment-specific configurations; distinct architectures and adaptation mechanisms; scalable sampling and training; CPU/GPU and parallel execution as useful; resumable, versioned checkpoints; deterministic task manifests; and robust statistical aggregation. Select only the machinery justified by the research question—building a universal framework is not the objective.

Document actual availability separately from planned extensions. Record training interactions, optimization steps, evaluation cost, hardware/time requirements, seeds, source versions, task manifests, and the evidence underlying results. When comparing methods, make computational and informational budgets explicit. Keep previously evaluated test panels and protocols distinguishable from genuinely new final evaluations.

## Results, reproducibility, and continuation

The desired output is a defensible account of **whether, when, and why meta-learning improves adaptation**, including cases where it does not. Report initial performance, adaptation across experience levels, costs, variation across training seeds and task families, and limitations of interpretation. Testing and successfully running software are not themselves evidence for a scientific hypothesis.

Use `PROGRESS.md` as the single concise handoff for the *current* research session. Update it after substantial milestones and before stopping: current objective/stage, verified outcomes and artifact paths, active decisions, unresolved issues, and the next action. Keep it short and replace stale entries instead of accumulating a transcript. Historical details belong with versioned experiment artifacts and the existing research records.

For contributors and research agents, [`AGENTS.md`](AGENTS.md) describes the working principles. Neither that document nor this README should constrain independent theoretical exploration to the historical D-031 design.
