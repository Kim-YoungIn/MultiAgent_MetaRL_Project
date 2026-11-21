# Structured-game transfer experiment

This study tests a hierarchical prior for fast reward prediction in the supplied `StructuredGame` simulator. The predeclared difference in gain from zero to eight shared probe interactions was −0.0151 for the mixed-family empirical-Bayes predictor relative to a scratch online predictor, with a 95% paired block bootstrap interval of [−0.0282, −0.0029]. On this endpoint, the scratch model's selected-action value increased more. This is a change in expected one-step reward as the shared game and opponent states also evolve. It is not an estimate of the effect of observing history alone.

| Method | k=0 | k=8 | k=16 | k=8 minus k=0 |
|---|---:|---:|---:|---:|
| Mixed empirical Bayes with updates | −0.04385 | −0.00167 | 0.02052 | 0.04218 |
| Scratch online regression | −0.05051 | 0.00677 | 0.02059 | 0.05727 |
| Mixed empirical Bayes frozen | −0.04385 | −0.03665 | −0.02460 | 0.00720 |
| Pooled pretraining with updates | −0.04339 | −0.00400 | 0.02209 | 0.03938 |

At eight interactions, the mixed adaptive model's absolute expected reward exceeds its frozen counterpart by 0.03498 and the pooled adaptive model by 0.00233. These are descriptive comparisons on the same state. Against scratch, its absolute value is lower by 0.00844 at k=8, while its k=0 value is higher by 0.00666. Thus the negative gain contrast reflects both differences, and the frozen comparison suggests that online updates help this fitted predictor under the probe distribution. None of these differences proves that the learned population prior beats ordinary pretraining or that its adaptation rule was learned.

The prespecified descriptive comparison of mixed-family versus resource-only training gives −0.00823 for the gain contrast, with 95% interval [−0.02347, 0.00681]. The arms have equal interaction budgets, but the interventions also change per-family exposure, so the comparison does not isolate diversity from distribution and exposure. The interval includes both directions. `curves.csv` gives each motif and budget.

In separate 32-step deployment, mean cumulative focal return was 0.992 for adaptive mixed empirical Bayes, 0.889 for scratch, 0.948 for adaptive pooled, and −0.927 for frozen mixed. These are descriptive means without deployment confidence intervals. Different actions produce different state and opponent trajectories. The opponent's mean cumulative absolute Q update during these rollouts ranged from 0.905 to 0.960 across methods, establishing parameter updates but not strategic competence. All task parameters and outcomes are under the specified artificial generator distribution.

Each of 16 blocks fits independent 32-game mixed and resource training catalogs, with 1,024 focal interactions per arm. The final panel comprises 512 held-out generated games, balanced over four motifs. The simulation used 32,768 training turns, 8,192 shared probe turns, and 114,688 method-specific deployment turns. The saved block runtimes sum to 13.5 seconds on the recorded WSL CPU environment. This excludes analysis and verification. Every held-out game and full training trajectory is retained under `runs/structured_transfer_v1/`. An independent replay reconstructed all 32,768 training transitions, 8,192 probe transitions, 14,336 method-query decisions, 114,688 deployment transitions, fitted weights, and the reported bootstrap interval. It allowed a tolerance of 10⁻¹² in action-value ties because closed-form and sequential linear algebra differ at machine precision. It did not reconstruct the training opponent random draws. See `runs/structured_transfer_v1/verification.json`. Exact fingerprint disjointness covers serialized specifications, and says nothing about equivalence after action relabeling.

## Material protocol deviations

The generator creates one public state for resource and reciprocity, two public states for sequence, and two public states plus hidden trust for mixed. Only resource and mixed include stochastic transition gates. The original protocol described every game as having three states and stochastic transitions. NumPy broadcasting also duplicated a single public state into both model state features in resource and reciprocity. This duplication changes effective regularization of those models, and state encoding differs systematically across families. It is present for all methods and the final run has been preserved. These details limit attribution of the diversity comparison to game motifs. This report records the correction and the pre-evaluation protocol is preserved unchanged.

The model predicts immediate reward using hand-selected history and state features, a Gaussian pseudo-likelihood, and a fixed regularization strength. It cannot establish effective long-horizon planning or active exploration, and the learning opponent uses only last joint actions. Performance in these generated families does not establish H1 or H2 in general. Run the exact source and inspect complete raw results before extending the conclusion.

## Reproduction

From the repository root, using Python with NumPy and the existing MAMRL package path:

```bash
PYTHONPATH=05_toy_models OPENBLAS_NUM_THREADS=1 .venv/bin/python research/structured_transfer/run.py runs/structured_transfer_reproduction --blocks 16 --seed 67834219
PYTHONPATH=05_toy_models OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv/bin/python research/structured_transfer/analyze.py runs/structured_transfer_reproduction
PYTHONPATH=05_toy_models OPENBLAS_NUM_THREADS=1 .venv/bin/python research/structured_transfer/verify.py runs/structured_transfer_reproduction
PYTHONPATH=05_toy_models .venv/bin/python -m unittest discover -s tests -p 'test_structured_games.py' -v
```

The final manifest stores hashes of the protocol and execution sources, the Python and NumPy versions, and the random seed. A run can be resumed while its source and configuration remain identical. An incomplete block is retained and needs to be moved aside before resuming.
