# Reproduce the adaptation studies

**Absolute reward after exploration.** On 6,144 new structured games the cumulative-return opening beat both a gain-only target and an immediate-reward target. Its advantage in choosing a more useful numerical reward signal was unresolved when the game and opponent state were restored. [Study and reproduction](absolute_exploration/REPORT.md)

The follow-up studies relate rule similarity to measured adaptation and resume the full D-031 budget from its sealed archive. The original exploratory reward-stakes protocol has also been executed and verified. [Similarity results and reproduction](rule_similarity/REPORT.md) · [D-031 continuation](d031_completion/README.md) · [Reward-stakes experiment](probe_stakes/REPORT.md)

The subsequent direct comparison between meta-training and ordinary pretraining against fixed and updating opponents has its own run commands, raw data guide and independent verification. The learned two-turn planning experiment is documented separately. [Direct opponent-adaptation experiment](opponent_adaptation/REPORT.md) · [Two-turn planning experiment](active_probe/REPORT.md)


Run from the repository root on Linux. This checkout contains the supplied structured-game extension, including its compatibility change and tests. The new experiments require NumPy; the analysis figures require Matplotlib. The measured environment used Python 3.14.4, NumPy 2.5.4 and Matplotlib 3.11.2. The historical regression suite also uses PyTorch where available.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4 matplotlib==3.11.2
export PYTHONPATH="05_toy_models:."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mamrl_figures

.venv/bin/python -m research.adapt_v1.run runs/repro_context --split final --workers 6
.venv/bin/python -m research.adapt_v1.analyze runs/repro_context
.venv/bin/python -m research.adapt_v1.verify runs/repro_context

.venv/bin/python -m research.structured_transfer.run runs/repro_structured --blocks 16 --seed 67834219
.venv/bin/python -m research.structured_transfer.analyze runs/repro_structured
.venv/bin/python -m research.structured_transfer.verify runs/repro_structured
.venv/bin/python -m research.structured_transfer.verify_independent runs/repro_structured

.venv/bin/python -m research.structured_oracle.test_oracle
.venv/bin/python -m research.structured_oracle.run runs/repro_two_turn
.venv/bin/python -m research.structured_oracle.verify runs/repro_two_turn

.venv/bin/python -m unittest research.structured_scale.test_scale -v
.venv/bin/python -m research.structured_scale.run runs/repro_scale --blocks 16 --seed 91827361
.venv/bin/python -m research.structured_scale.analyze runs/repro_scale
.venv/bin/python -m research.structured_scale.verify runs/repro_scale
.venv/bin/python -m research.structured_scale.verify_independent runs/repro_scale

.venv/bin/python -m research.figure runs/repro_context runs/repro_structured runs/repro_figure.png
```

The two-turn runner creates its output directory exclusively. For another execution choose a fresh directory. The two-arena runner accepts a separate development split or a custom seed, block count and games per stratum. Its manifest binds the source, NumPy/Python versions and configuration. It uses a filesystem lock, atomic block publication and checksum verification on resume. Both structured learning studies can resume completed blocks and retain an incomplete block for inspection. Their fitted priors are small coefficient arrays inside each block's results. Matching historical sources at the same path matters for reusing the original structured manifests, which include absolute paths for some source files. A new output directory creates a new manifest using local paths.

## Results and independent checks

| Local final output | What to inspect |
|---|---|
| `runs/adapt_v1_final/` | Source snapshot, manifests, 12 training archives per block (trajectories, likelihoods and fitted priors), probe/deployment arrays, `analysis/primary_inference.csv` and `verification.json`. Independent code reconstructs likelihoods, both fits, paired outcomes and bootstrap intervals. |
| `runs/structured_transfer_v1/` | Exact training/test specs, raw features, joint actions and payoffs, learned coefficients, one-step action values, deployments and analysis. `verification.json` covers all trajectories with training actions held fixed. A second implementation reconstructs the primary and replays one game per motif, with every controller. |
| `runs/structured_oracle_v1/` | All generator and reset seeds, per-action immediate and two-turn values, motif summaries. The independent verifier reconstructs all 288 pair gaps and brute-forces one pair for each of 12 motif/setting combinations using the original environment. |
| `runs/structured_scale_v1/` | 16 training fits, game specs, trajectories, per-state action values, live deployments and `analysis.json`. Independent verification checks every selected payoff, reconstructs the primary interval and replays one game per motif/setting with all six policies. |

Compact tables and figures for the [integrated report](REPORT.md) are in [evidence](evidence/). Raw files under `runs/` are local and git-ignored. From the repo root, `sha256sum -c research/evidence/raw.sha256` checks the retained numerical artifacts of all four completed experiments. These are preservation hashes collected after evaluation. The study protocols separately record the timing and scope of each analysis.

For the full repository regression tests use `PYTHONPATH=05_toy_models:. .venv/bin/python -m unittest discover -s tests -v` in a training-capable environment. This session ran 223 tests, with 220 passing and three CUDA-only tests skipped. The structured extension's 17 tests were included. The seven new two-turn and cross-size tests ran separately and passed. Initial development of the two-arena runner stopped at process startup when the default multiprocessing forkserver could not bind a socket. The explicit Linux fork runner completed development and final execution. Fresh reruns reproduced all raw numerical block archives for the two-arena study and all 48 training, test and numerical files in each of the two structured learning studies byte for byte.
