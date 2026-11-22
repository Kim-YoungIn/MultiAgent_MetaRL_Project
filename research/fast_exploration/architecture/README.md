# Fast Exploration architecture

In v1, a recurrent Teacher selects the exploration actions in a new two-player structured game. The Bayesian linear Student observes those interactions, updates its reward predictions and takes over for assessment against a Q-learning opponent. The Teacher trains on the Student's absolute assessment reward. Random, greedy and belief-blind exploration share the same Student update rule.

The original model and prior objective are specified in the [v1 protocol](../../fast_exploration_v1/PROTOCOL.md). The [scale protocol](../../fast_exploration_v1/SCALE_PROTOCOL.md) fixes the training and evaluation budgets used for the [v1 results](../results/README.md). The [v2 protocol](V2_PROTOCOL.md) specifies the prior, feedback, learning-objective and fixed-budget comparisons. Follow-up experiments are tracked in the [experiment plan](EXPERIMENT_PLAN.md).

## Execution

Use Python with NumPy 2.5.4, PyTorch 2.14.1 and Matplotlib 3.11.2. The confirmed CUDA wheel version is 2.14.1+cu130. The CPU run used that wheel with its CPU device. PyTorch provides separate CPU and CUDA packages. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting deterministic CUDA training. An explicit CUDA request checks actual CUDA access.

```bash
export PYTHONPATH=05_toy_models:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mamrl_fast_mpl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m unittest discover -s tests -p 'test_fast_exploration*.py' -v
.venv/bin/python -m research.fast_exploration_v1.benchmark /tmp/fast_benchmark.json --cuda
.venv/bin/python -m research.fast_exploration_v1.scale --split dev --output runs/repro_fast_dev --workers 2 --device cpu
.venv/bin/python -m research.fast_exploration_v1.scale --split final --output runs/repro_fast_final --workers 8 --device cpu --sampling cpu
.venv/bin/python -m research.fast_exploration_v1.verify runs/repro_fast_final
.venv/bin/python -m research.fast_exploration_v1.export runs/repro_fast_final
```

Recompute paired AUC by game family and separate exploration and assessment rewards from an existing run in a separate output directory.

```bash
.venv/bin/python -m research.fast_exploration_v1.supplement runs/repro_fast_final --output runs/repro_fast_followup
```

Omit the benchmark's `--cuda` flag to compare only the CPU implementations. `--only-block 0` measures a single full-budget block in the same output directory. Reusing the same command without that flag resumes the remaining work. Checkpoints are atomic at 25-update boundaries and preserve both optimizers and the sampler state. The source and execution manifest binds a resumed run to its original device and protocol.

## Outputs

Each block has a JSON record, both Teacher checkpoints, and a compressed NPZ of all realized assessment turns and exploration totals per configuration, game, reset, method and K. The NPZ includes exact rule fingerprints, reset seeds and main training traces. `analysis/trials.csv.gz` contains per-trial returns and first-turn rewards. The block JSON provides the curve and seed mapping. `analysis/report.json` and its CSV counterparts provide ordinary and controlled comparisons, prior sensitivity, uncertainty and between-block variability. Three figures show the absolute adaptation curve, finite-difference slopes and individual assessment turns.

The repository's earlier batch design in `mamrl/meta_batch.py` supplied the separation of simultaneous rollout collection and recurrent gradient replay. This run uses a shared batched collector for training and assessment, a new vectorized structured simulator, the existing Student representation, and the atomic artifact utility from `experiment/artifacts.py`. GPU training and CPU/GPU sampling paths remain executable. The main run uses the measured faster path, CPU with eight workers.

## Turn-by-turn actions

Rebuild the actions in a separate directory from a completed CPU-sampling run and its recorded environment. Replay checks each assessment reward and exploration sum against the sealed payoffs before publication. The action archives describe reconstructed evaluation trajectories.

```bash
.venv/bin/python -m research.fast_exploration_v1.actions replay \
  runs/fast_exploration_scale_final runs/fast_exploration_scale_actions --workers 8
.venv/bin/python -m research.fast_exploration_v1.actions csv \
  runs/fast_exploration_scale_actions/block_000.npz --game 2 > game_actions.csv
```

The CSV includes both resets of the chosen zero-based game index. It identifies the policy, K, phase, turn, focal controller, both executed action IDs and their game-specific `invest`, `harvest` or `signal` roles. Each NPZ indexes joint actions by `[method, budget, game_reset, turn, player]`. The player order is focal then opponent, and `-1` pads exploration past K. Controlled assessment shares the corresponding `reward_auc_exploration` prefix. `complete.json` binds each block to its input rewards and checkpoint hashes.

## Code map

| Files | Responsibility |
|---|---|
| [scale.py](../../fast_exploration_v1/scale.py) | Split, parallel blocks, checkpoints and run manifest |
| [batch.py](../../fast_exploration_v1/batch.py), [simulator.py](../../fast_exploration_v1/simulator.py) | Warm-up, batched structured-game trajectories and online updates |
| [study.py](../../fast_exploration_v1/study.py), [prior.py](../../fast_exploration_v1/prior.py) | Teacher, scalar reference and Student prior fitting |
| [report.py](../../fast_exploration_v1/report.py), [supplement.py](../../fast_exploration_v1/supplement.py), [by_seed.py](../../fast_exploration_v1/by_seed.py) | Paired estimates, reward breakdown and seed plots |
| [verify.py](../../fast_exploration_v1/verify.py), [export.py](../../fast_exploration_v1/export.py), [actions.py](../../fast_exploration_v1/actions.py) | Integrity and scalar replay checks, per-trial returns and executed actions |

## Run v2

The v2 package uses the same CPU environment and simulator. Run from the repository root with the environment variables above. The development panel selected UCB β=0.15 for the sealed final run.

```bash
.venv/bin/python -m research.fast_exploration_v2.run --split dev --workers 4 --output runs/repro_v2_dev
.venv/bin/python -m research.fast_exploration_v2.verify runs/repro_v2_dev --calibrate
.venv/bin/python -m research.fast_exploration_v2.run --split final --workers 12 --ucb-beta 0.15 --output runs/repro_v2_final
.venv/bin/python -m research.fast_exploration_v2.verify runs/repro_v2_final --reference runs/repro_v2_dev
.venv/bin/python -m research.fast_exploration_v2.analyze runs/repro_v2_final --output runs/repro_v2_final/analysis
.venv/bin/python -m research.fast_exploration_v2.diagnostics runs/repro_v2_final
.venv/bin/python -m research.fast_exploration_v2.horizon runs/repro_v2_final --workers 12 \
  --reference runs/repro_v2_dev --output runs/repro_v2_horizon
.venv/bin/python -m research.fast_exploration_v2.horizon_ucb runs/repro_v2_final runs/repro_v2_horizon \
  --workers 8 --output runs/repro_v2_horizon_ucb
```

Reissuing the same v2 command resumes each incomplete block from its last 25-update checkpoint. Each block has both players' turn-level actions and evaluation rewards. `train_actions` is indexed by update, the first seven entries of `methods`, batch member, turn and player. Evaluation phase arrays are indexed by method, K, game reset, turn and player. The player order is focal then opponent. Probe arrays use `-1` for unused action slots and `NaN` for unused reward slots. The method and budget identify the handoff in `fixed_actions` and the two fixed-budget Teachers' `train_actions`. The `action_to_role` table maps action indices to invest, harvest and signal within each test game. The reset-assessment experiment stores the Student's selected `probe_features` directly for the common-query diagnostic. Other rollout features can be reconstructed from the executed actions, rule and reset seeds.
