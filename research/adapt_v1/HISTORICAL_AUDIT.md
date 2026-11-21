# Historical audit

Audit date 2026-11-21. The categories below distinguish source statements from numerical checks. Paths are relative to the repository root.

## Research objectives and course requirements

- `01_originals/RL_TeamProject_Proposal_metaRL_MA.docx` asks whether learning across games produces faster adaptation to unseen rules and opponents. Its H1 proposes a diversity benefit, and H2 proposes that reusable interaction patterns matter. Both are explicitly working hypotheses. It calls for matched diversity budgets, early adaptation curves, initial and late returns, multiple seeds, held-out tasks, and feedforward and memory-reset controls. Recurrent RL²/PPO is an initial candidate. Learning opponents and alternative exploration are proposed follow-ups.
- `01_originals/The project guideline.txt` calls for rigorous scientific methodology, theoretical motivation, and investigation of successes and failures. It does not require novel algorithms or state-of-the-art results. The three-page original `01_originals/Evaluation_Plan(1).pdf` was subsequently decoded using its embedded font maps. It requires a five-minute presentation and three questions, peer review, slides, a code ZIP and a final report of up to five pages plus references, with a team contribution section.

## Historical design decisions and assumptions

- `04_design/Adaptation_Experiment.md` defines D-031 as a comparison of recurrent exploration policies trained on student downstream improvement (M) versus immediate exploration reward (P), with an observation-masked M (O). Each supplies experience to a tabular Q-learning student while the opponent also learns. `experiment/training.py` implements the M/P targets and O mask and `experiment/model.py` implements the shared per-action GRU. This is a comparison of training objectives and information access within a particular teacher–student system. It does not by itself isolate the effect of meta-learning across architectures.
- The predeclared primary outcome averages the increase from k=0 across k=0, 3, 6, 12, 24, with H=24 post-exploration interaction turns. The student and opponent continue learning during that evaluation. Exploration alters the opponent as well as student experience, so the natural outcome includes both effects. A 3×3 experience/opponent-state cross is a useful diagnostic, but interventions on constructed states need their own interpretation.
- The v2 final panel comprises 64 fresh and 64 exception rules per action size, and the documented collision checks report zero overlap with checked historical, training, and development fingerprints (`runs/d031_acceptance_20261121/panel_verification.json`). The panel has already been scored, so it is opened evidence for any later redesign. Fingerprint uniqueness does not establish strategic independence or prove coverage of every historical artifact.
- H1 rule-count conditions fix total training budget, making count and exposure per rule change together. Family breadth also changes training distribution. Such contrasts can estimate effects of those bundled interventions but do not isolate strategic diversity or an individual reusable primitive.

## Independently checked raw evidence

The following was recomputed by reading each gzip JSONL trial in `runs/d031_v2/cells/primary/c1/j1/n3/` and calculating the trapezoidal area from its five stored returns with weights (1.5, 3, 4.5, 9, 6)/24. Primary rows are fresh rules with the steady or SARSA training-opponent profile. The seven policy files each have 3,072 records, of which 1,024 meet this primary filter. Each method's primary outcomes cover 64 rules, two opponent profiles, and eight replicate seeds. They share one catalog and one trained-model seed, not 1,024 independently trained models.

| Method | Primary area | R(0) | R(3) | R(24) |
|---|---:|---:|---:|---:|
| M | 0.109972 | 0.149953 | 0.178158 | 0.335306 |
| P | 0.112573 | 0.149953 | 0.193442 | 0.336309 |
| O | 0.101831 | 0.149953 | 0.178554 | 0.334785 |
| Random | 0.101198 | 0.149953 | 0.182895 | 0.325942 |

The single paired M−P area contrast is −0.002601. For each M/P/O ledger, the audit counted 800 updates and 1,228,800 real plus 614,400 erasure simulation steps. The SHA-256 of each saved `training_state.pt` matched its ledger. Exactly three training ledgers occur under `runs/d031_v2/cells/`, all at primary/c1/j1. This supports the stored first-cell numbers and the reported 3/375 completed training runs. It does not establish the 25-pair predeclared inference, H1, H2, or general performance on learning opponents. The values above are descriptive for this opened panel and one training pair. No model-level uncertainty interval can be estimated from that pair alone.

In the stored two-turn audit (`runs/d031_v2/audit_n3.json`), recounting fresh uniform-opponent pairs using a 1e−10 tolerance gives 21/32 positive first-reward observation values and 1/32 positive costs from imposing an immediate-optimal opening. The extension diagnostic uses different opponents, state dynamics and hypotheses, so its frequencies do not estimate a treatment effect relative to D-031.

`runs/d031_v2_dev/` and the prior experiment folders contain development evidence. Their reported checks and earlier results require separate reproduction before use as quantitative findings. The audit did not rerun training, regenerate panels, validate every trajectory, or independently prove the panel collision report.
