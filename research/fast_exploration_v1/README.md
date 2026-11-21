# Scaled Fast Exploration

This is the scaled execution of the patched Fast Exploration architecture. The original protocol, prior fitting objective, Gaussian Student, warm Q-learning opponent and recurrent actor–critic are recorded in [PROTOCOL.md](PROTOCOL.md). The additional [scale protocol](SCALE_PROTOCOL.md) fixes the larger training and evaluation budgets. Existing D-031 and earlier Fast Exploration run directories contain separate experiments.

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

Omit the benchmark's `--cuda` flag to compare only the CPU implementations. `--only-block 0` measures a single full-budget block in the same output directory. Reusing the same command without that flag resumes the remaining work. Checkpoints are atomic at 25-update boundaries and preserve both optimizers and the sampler state. The source and execution manifest binds a resumed run to its original device and protocol.

## Outputs

Each block has a JSON record, both Teacher checkpoints, and a compressed NPZ of all realized assessment turns and exploration totals per configuration, game, reset, method and K. The NPZ includes exact rule fingerprints, reset seeds and main training traces. `analysis/trials.csv.gz` contains per-trial returns and first-turn rewards. The block JSON provides the curve and seed mapping. `analysis/report.json` and its CSV counterparts provide ordinary and controlled comparisons, prior sensitivity, uncertainty and between-block variability. Three figures show the absolute adaptation curve, finite-difference slopes and individual assessment turns.

The repository's earlier batch design in `mamrl/meta_batch.py` supplied the separation of simultaneous rollout collection and recurrent gradient replay. This run uses a shared batched collector for training and assessment, a new vectorized structured simulator, the existing Student representation, and the atomic artifact utility from `experiment/artifacts.py`. GPU training and CPU/GPU sampling paths remain executable. The main run uses the measured faster path, CPU with eight workers.
