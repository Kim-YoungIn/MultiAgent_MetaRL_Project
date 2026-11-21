# Multi-Agent Meta-Learning Research — Agent Guidance

This repository investigates **how meta-learning affects agents' ability to adapt in multi-agent environments**. The objective is to produce a theoretically justified, empirically informative, and reproducible study—not to preserve a particular architecture or improve a preselected score.

## Research orientation

- Start from the scientific question, not the existing implementation. Define what constitutes a task, an opponent, adaptation, generalization, and a valid causal or comparative claim before committing to methods.
- The historical questions are **H1**, how training-game and opponent diversity affects adaptation to unseen tasks, and **H2**, whether reusable structure across games explains transfer. H1 was prioritized, with H2 contingent on the evidence. Both formulations can be criticized or refined; neither prescribes an algorithm.
- The original interest in learned exploration is important context, **not a requirement for a separate exploration policy**. Do not silently replace adaptation research with opponent shaping or another neighboring topic.
- Treat `01_originals/` as the original proposal and course requirements; `00_management/`, `04_design/`, `05_toy_models/`, `experiment/`, and past `runs/` as historical decisions, implementations, and evidence. Their labels such as FIXED describe previous experiment contracts, not permanent restrictions on new research. Preserve the reproducibility of those historical experiments.

## Independent, ambitious design

- Explore **fundamentally different formulations and architectures**, including approaches beyond the repository's current examples. Teacher–Student decomposition, recurrent meta-agents, learned optimizers, inference-based adaptation, PPO, Q-learning, Memory-One reward tensors, fixed within-task rules, and D-031 are all **optional choices**.
- Use current literature, theory, and appropriate computational tools to compare candidates in terms of expressive power, adaptation mechanism, information use, learning signal, strategic interaction, identification, compute cost, and the strength of conclusions they permit.
- Be willing to replace the environment, learning algorithm, experiment protocol, or substantial parts of the codebase if a stronger research design warrants it. Neither incremental patching nor wholesale rewriting is a goal in itself.
- Scale experiments when justified. Existing values for model size, action count, number of games or opponents, training updates, seeds, horizons, and hardware are **not research limits**. Extend the runner, configuration, batching, parallelism, checkpointing, or analysis pipeline where needed; do not claim unsupported functionality already exists.

## Scientific and engineering integrity

- Distinguish meta-learning effects from stronger priors, extra exposure, additional optimization, opponent dynamics, and evaluation leakage. Compare reasonable alternatives, not only weak baselines. Measure early performance, adaptation as experience accumulates, total costs, and failure conditions.
- Separate exploration of hypotheses and design from **precommitted final evaluations**. Never silently overwrite completed experiments or reinterpret opened test sets as untouched. Record substantial redesigns under a new version or protocol.
- Implement, run, and independently validate work wherever feasible. Tests passing, a model training, and a research hypothesis being supported are different claims. Preserve null and negative results; report failures and incomplete runs explicitly.
- Use the available capable models, tools, and subagents deliberately for independent scientific critique, focused implementation, and validation. Keep one owner for conflicting edits; assess evidence rather than agent agreement.

## Session continuity and documentation

- **Read `PROGRESS.md` at session start** alongside only the sources relevant to the task. Treat it as a compact navigation aid, not an authority overriding code, data, user instructions, or source documents.
- **Create `PROGRESS.md` if missing; update it after meaningful milestones and before ending a work session.** Replace obsolete status rather than appending a diary. Keep it concise (normally 15–30 lines): current objective and stage, completed work with evidence, active decisions and assumptions, blockers or unresolved issues, and the next actionable step. Include paths/commit IDs as useful; never mark unrun tests or experiments complete.
- Keep `README.md` focused on research scope, actual capabilities, reproduction, and extensibility. Update other management documents only when their current purpose requires it; avoid redundant logs, excessive instructions, and documentation growth that crowds out research.

**Success criterion:** a defensible, falsifiable, and reproducible account of when and why meta-learning improves—or fails to improve—adaptation in multi-agent settings.
