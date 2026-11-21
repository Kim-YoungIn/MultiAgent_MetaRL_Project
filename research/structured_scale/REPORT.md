# Cross-size structured-game transfer

An empirical-Bayes prior trained on three-action, depth-three structured games was evaluated without refitting on six-action, depth-five games. At eight shared probe interactions its mean expected one-step reward was **0.01189**, compared with **0.00813** for an adaptive pooled-ridge prior trained on the same data. The prespecified paired difference was **0.00376**, with a 95% percentile interval of **[0.00071, 0.00702]** over 16 independently sampled training catalogs and paired test sets. This result supports a small advantage for this empirical-Bayes estimate over this pooled estimate under the specified generator and prior family. It does not establish an advantage over all pretrained learners or the learning of a general adaptation procedure.

The endpoint uses the same continuing environment and learning-opponent state for all methods. Expected current reward averages over the opponent's action distribution. Each method selects its action using its own predictions from the common history.

| Six actions, depth five | k=0 | k=8 | k=16 |
|---|---:|---:|---:|
| Empirical Bayes, adaptive | −0.02371 | 0.01189 | 0.01478 |
| Pooled prior, adaptive | −0.02054 | 0.00813 | 0.01697 |
| Empirical Bayes, frozen | −0.02371 | −0.02272 | −0.01945 |
| Scratch, adaptive | −0.02536 | 0.00898 | 0.01512 |
| State input masked, adaptive | −0.02371 | 0.00982 | 0.01251 |
| Lags two to five masked, adaptive | −0.02386 | 0.01081 | 0.01849 |

The frozen and ablation rows are descriptive. Compared with the frozen predictor, the adaptive predictor benefits from its updates at k=8 in this common-history comparison. The pooled adaptive model is slightly better at k=0 and k=16. In three-action depth-three games the k=8 means were 0.02166 and 0.02015 for adaptive empirical Bayes and adaptive pooled, respectively. Complete curves by motif appear in `runs/structured_scale_v1/curves.csv` and `curves.png`.

Separate 24-turn six-action deployments yielded mean cumulative focal rewards of 0.648 for adaptive empirical Bayes, 0.542 for adaptive pooled, 0.462 for scratch and −0.418 for frozen empirical Bayes. These means are descriptive, with no deployment confidence intervals. Methods can change subsequent opponent behavior and environment state through their own actions. Opponent mean cumulative absolute Q change in six-action deployment was 0.414 for adaptive empirical Bayes and 0.398 for adaptive pooled. This confirms updates occurred, without demonstrating opponent competence.

The state and long-history ablations use the same fitted prior with masked features throughout online updates and action selection. The long-history ablation removes lags 2–5, so it tests more than history beyond the training depth. The training games contain no observations at lags 4–5. The prior coefficients for those features therefore start at zero, although online updates may learn them in depth-five evaluation. This experiment does not identify whether the newly available lag-4/5 information itself helped, nor a causal effect of changing game size or depth. The generator's n=3 and n=6 instances share seeds across settings but are not strategically matched transformations.

Sixteen independent training blocks each contain 32 mixed-family games and 1,024 interactions. Each of three settings has 384 test games, balanced across four motifs. The run contains 16,384 training, 18,432 probe, and 165,888 live deployment turns. Block elapsed times sum to 14.75 seconds on the recorded WSL CPU environment, excluding analysis and verification. Game specifications, probe traces, learned priors, deployment actions and rewards, opponent updates, random seeds, and source hashes are in `runs/structured_scale_v1/`. A separate replay reconstructed all training transitions and fitted priors, and four test games per block and setting, one per family. It checked 3,072 probe turns, 4,608 method-query decisions, 27,648 deployment turns and the primary bootstrap interval. It accepted action-value ties within 10⁻¹² to account for numerical differences between sequential and closed-form updates. See `runs/structured_scale_v1/verification.json`. Exact fingerprint uniqueness does not rule out semantic or action-relabeling equivalence.

The generator has one public state for resource and reciprocity, two public states for sequence, and two public states plus hidden trust for mixed. Only resource and mixed have stochastic gates. Missing public states are explicitly padded with zero. A single action-equivariant feature representation and shared fitted prior support both action counts. Tie-breaking uses a random action permutation shared across methods. A separate one-block development rerun exactly reproduced its game specifications, traces and result arrays. The underlying reward kernel was compared with every simultaneous action pair for each motif at all three sizes, and regression tests confirmed relabeling equivariance, missing-state padding and equality of sequential and closed-form posterior updates.

The model is a hand-specified Gaussian linear immediate-reward predictor with a learning Q opponent. The comparison leaves open longer-horizon planning, active exploration, unfamiliar strategic families, new opponent algorithms, and neural meta-learning. Training with richer history and varying action count would require its own controlled experiment.

## Reproduction

Run from the repository root with the existing environment, using a fresh output directory.

```bash
PYTHONPATH=.:05_toy_models OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest research.structured_scale.test_scale -v
PYTHONPATH=.:05_toy_models OPENBLAS_NUM_THREADS=1 .venv/bin/python research/structured_scale/run.py runs/structured_scale_reproduction --blocks 16 --seed 91827361
PYTHONPATH=.:05_toy_models OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv/bin/python research/structured_scale/analyze.py runs/structured_scale_reproduction
PYTHONPATH=.:05_toy_models OPENBLAS_NUM_THREADS=1 .venv/bin/python research/structured_scale/verify.py runs/structured_scale_reproduction
```

The runner saves independently reproducible blocks and checks source hashes and configuration before resume. If a block was interrupted before its completion marker, the incomplete directory is retained and must be moved aside before resuming. This experiment is a prospective follow-up to the opened structured-transfer v1 study and uses fresh seeds and its own protocol.
