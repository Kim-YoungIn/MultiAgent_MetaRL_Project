# Fast Exploration v1 — prospective, separate from v0.7.1/D-031

## Question and scope

Can a meta-trained recurrent exploration Teacher collect *more useful experience*
for a freshly initialized Student in an unseen structured multi-agent game than
random and reward-greedy exploration, under equal per-game probe budgets?
Does a **Gaussian reward-model prior trained for post-update absolute utility**
change that outcome? These are separate effects; a better prior alone is not
proof of learned exploration. This is a new, exploratory experiment, not a
reanalysis or modification of the original D-031 or v0.7.1 reports.

## Fixed architecture

- Game: original three-action, depth-three resource/sequence/reciprocity/mixed
  `StructuredEnv`, balanced over families. Neither learner is given the hidden
  rule or opponent's Q table. Each game has a simultaneously acting epsilon-
  greedy Q-learning opponent, warmed for 32 random turns in that game (without
  revealing those turns to the focal agent), then still learning during play.
- Student: existing action-relative 20-feature *Bayesian linear immediate-reward
  predictor*. Its coefficient prior is `w_a ~ N(mu,I)` independently per action;
  an observed focal reward updates only that action via online ridge. The
  Student chooses the predicted best immediate reward in assessment and
  continues learning except in the controlled diagnostic. Student state resets
  each game, while the shared `mu` is retained. **This is not the old D-031 Q
  learner** and is not a long-horizon optimal controller.
- Teacher: a new 32-unit GRU actor with the D-031 recurrent action architecture,
  extended from 5 to 7 input features to receive the Bayesian Student's
  predicted reward and posterior predictive uncertainty for each action. The
  comparison **Belief-Blind Teacher** receives zeros in those two positions
  but has identical network capacity and training budget. The learned Teacher
  uses stochastic actions and is trained by Monte Carlo recurrent actor-critic
  on downstream *absolute* Student reward.
  It selects `K` exploration actions; Student receives the same transitions
  and updates passively. Teacher weights transfer across games; its recurrent
  state resets per game.
- Baselines: the equally trained belief-blind Teacher, uniformly random
  exploration, and Student greedy exploration. All use the *same trained
  Student prior and learning rule* and face paired game/reset/opponent
  and transition random streams;
  only action-conditioned trajectories may differ.

## Prior training and its objective

Fit the Gaussian prior **mean** `mu` on a disjoint training panel of uniform
probe histories. At each checkpoint `k in {0,2,4,8,16}`, replay the exact online
ridge posterior for each candidate prior. The objective is a trapezoidal AUC
of **absolute expected next reward**; it is *not* a difference relative to
scratch, frozen, or a lowered zero-shot baseline. Implemented with an off-
policy inverse-propensity estimator `3 * pi_mu(a_k | h_k) * r_k` because each
training action is uniform over three arms. `pi_mu` is a temperature-five
softmax of the posterior value predictions: a **differentiable surrogate** for
the Student's actual greedy choice. Add `0.01 ||mu||^2` regularization.
The training objective uses only sampled action rewards and public features;
it does **not** use the simulator's full-action `truth` table. This objective
estimates one-step utility on *random-policy histories*, not true on-policy
24-turn downstream AUC; therefore evaluation must check for objective mismatch.
For a fair same-data comparator, also retain the v0.7 pooled regression prior;
zero mean is a third control. The two trained Teachers are held fixed in these prior
sensitivity comparisons: they do not measure jointly reoptimized systems.

## Teacher training target

For the sampling grid `K in {0,2,4,8,16}`, choose positive K with probability
proportional to its normalized trapezoid AUC weight. K=0 is independent of
Teacher and contributes no policy gradient. For each sampled game, collect K
Teacher-controlled steps, reset the episode but **retain** the student's
posterior and the opponent's learned Q, and evaluate H=24 Student greedy steps.
Bootstrap the last exploration opponent TD update toward the new assessment
reset observation rather than accidentally treating the exploration boundary
as termination.
The Monte Carlo reward/critic target is **absolute downstream mean payoff**,
not retained-minus-erased Q return or visible-minus-masked prediction gain.
Each teacher is trained with recurrent REINFORCE, a learned critic baseline,
entropy encouragement and gradient clipping. It is **not PPO**, though it
reuses the recurrent actor architecture. Trained models do not receive test
hidden rules, simulator counterfactual labels or test rewards during training.

## Evaluation and diagnostics

Per held-out game, run K={0,2,4,8,16} for learned, belief-blind, random
and greedy explorers, starting at paired game/opponent/reset and exogenous random streams. After K
steps, assess for H=24, with a fresh assessment episode, retaining Student and
opponent learning states. Compute `R(K)` and the normalized *absolute* AUC.
Compare learned-minus-blind, learned-minus-random and learned-minus-greedy
AUC within **independent train/test blocks** (not independent turns). Record zero-shot equality and the
exploration + downstream cumulative return for each K. Do not claim a
positive result unless a fresh final panel supports it.

A separate **controlled** evaluation restores the opponent's pre-exploration
Q and freezes both Student and opponent updating during assessment
(assessment episode state is already reset). Its contrast focuses on the Student coefficients retained
from Teacher experience rather than opponent shaping and assessment-time
learning. It does not prove the Teacher intentionally acquired information.
Actual deployed return includes exploration cost and must be reported beside
AUC. Prior comparisons under the same already-trained teacher are sensitivity
analyses, **not** evidence that Bayesian inference itself caused exploration.

## Adaptation and per-turn plots

Three plots are produced automatically (when Matplotlib is installed) from the
**held-out block records**, not from training losses or invented expected values:

1. `figures/adaptation_curve.png`: x = Teacher probe count K, y = Student mean
   realized reward during the fixed H=24 assessment after handoff. All policy
   curves and pointwise 95% intervals resample independently trained blocks.
   The area to y=0 under the Bayesian Teacher curve visualizes the *absolute*
   AUC, including negative payoff regions. Legends report each policy's
   normalized AUC. These y-values average *actual* assessment returns, not
   a model's reward predictions.
2. `figures/adaptation_slopes.png`: finite differences
   `(R(K2)-R(K1))/(K2-K1)`, not derivatives of the network's loss or evidence
   of greater information acquisition. Slopes are estimated between the
   prespecified checkpoints, not individual training optimizer updates.
3. `figures/assessment_reward_k8.png`: for a fixed K=8, x = Student assessment
   turn 1..H, y = realized reward at that particular assessment turn. This is
   not the same x-axis as the adaptation curve; ongoing Student/opponent
   learning can change this within-assessment curve. Change K with
   `python -m research.fast_exploration_v1.figure --output RUN --k-for-steps 4`.

Corresponding CSVs are written beside figures: `adaptation_curve.csv`,
`adaptation_slopes.csv`, `auc.csv` and `assessment_reward_k8.csv`. Uncertainty
bands are *descriptive pointwise* 95% block bootstrap intervals, not simultaneous
bands and not proof of a positive method contrast. AUC is recomputed by
trapezoidal integration of each individual block's displayed R(K), checked
against the stored AUC, and then averaged across blocks. The logged `R(K)`
includes assessment-time Student/opponent updating in the natural setting;
the controlled curves are retained separately in raw JSON. More detailed
turn-by-turn changes should be interpreted accordingly.

## Limitations and reproduction rules

AUC reflects high performance *early*, not the derivative itself. The Student
is myopic, its feature map is researcher-specified, and the learned prior is a
Gaussian reward-model prior, not a posterior over full hidden rules. Inferred
opponent behaviour remains outside that Gaussian. Family diversity is fixed,
so this experiment alone does not prove H1; feature sharing alone does not
prove H2. The current final preset is a research run, not a preregistered
confirmatory test and its inference intervals are **descriptive**. Keep
historical sealed experiments unchanged. All blocks are independently fitted;
`manifest.json` guards against reusing data with changed code/config.

## Commands (from the repository root, WSL/Linux)

```bash
# Use an existing CUDA-enabled .venv; do NOT reinstall the CPU-pinned
# requirements-training.txt into that environment.
.venv/bin/python -c "import torch; print(torch.__version__,torch.cuda.is_available(),torch.version.cuda)"
.venv/bin/python -m pip install numpy matplotlib
export PYTHONPATH=05_toy_models:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -p "test_fast_exploration_v1.py" -v
.venv/bin/python -m research.fast_exploration_v1.study --split dev --device cuda --output runs/fast_exploration_cuda_dev
.venv/bin/python -m research.fast_exploration_v1.study --split final --device cuda --output runs/fast_exploration_cuda_final
.venv/bin/python -m json.tool runs/fast_exploration_final/report.json
```

The study writes JSON curves, AUC/return outcomes, fitted prior coefficients and
both Teacher checkpoints per independent block. To generate plots install
Matplotlib in the Python environment and run the study; the same images and
CSVs can be regenerated later from already-completed blocks without retraining:

```bash
.venv/bin/python -m pip install matplotlib
.venv/bin/python -m research.fast_exploration_v1.figure --output runs/fast_exploration_final
```
 Change `--seed` and `--output` to
create an independent replication. Never reinterpret a development panel as
fresh confirmatory test evidence.

## PyTorch CUDA training / reproducibility

Run with `--device auto` (default), `--device cuda`, or `--device cpu`. The
explicit CUDA option **fails** if `torch.cuda.is_available()` is false; there
is no silent CPU fallback. CUDA runs place the fitted prior's PyTorch
optimization tensors and the recurrent actor-critic's batched replay,
autograd, and optimizer steps on the GPU. Python/NumPy game generation,
online Bayesian student updates, opponent Q updates, and per-game sampling
and evaluation remain on CPU. The sampling actor is a CPU snapshot of the
GPU-trained actor: this avoids slow synchronization for tiny individual
GRU forwards, while every policy-gradient update is recomputed in a batched
GPU graph. One K is sampled per training update (shared within the batch)
with the original AUC-weighted distribution; this preserves the expected
objective without padding unequal trajectories. GPU prior fitting uses FP32
for throughput on consumer GPUs; CPU fitting retains FP64. Device, CUDA runtime,
GPU name, and PyTorch version enter the run manifest, so previously sealed
outputs cannot silently be resumed across different hardware configurations.

The repository's historical `requirements-training.txt` pins **CPU-only**
PyTorch. Do **not** install that file after setting up CUDA in this experiment.
For an NVIDIA WSL machine with compatible driver, install a CUDA PyTorch
wheel (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu130`)
and verify `torch.cuda.is_available()`. The pip wheel includes the runtime;
`nvidia-smi`'s CUDA-version column indicates driver capability, not the
installed PyTorch wheel's runtime. Record timings with both `--device cuda`
and `--device cpu`: this small GRU and CPU environment are not guaranteed to
run faster overall on a GPU; batch size and prior optimization dominate
possible acceleration.

## Post-CUDA audit

Direct evaluation of a CUDA-resident Teacher samples with a CPU-local generator
from a CPU copy of the action distribution, then computes log probabilities on
the policy device. This supports both direct `evaluate(...)` and the normal
CPU sampling-copy path. The test suite checks both routes (CUDA tests are
skipped on machines with no GPU). The manifest fingerprints the environment,
opponent, student model, posterior fitting, plotting and protocol sources.
Plots refuse incomplete or corrupted sealed runs. Because manifest contents
changed, resume after this patch requires a **new output directory**; already
completed historical outputs should be left unchanged.
