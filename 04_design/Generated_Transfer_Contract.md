# Generated logical games / first transfer experiment — v1 (2026-11-14)

## Scope and preserved conditions

Build one connected generator→MA task manifest→common-policy training→held-out evaluation→HTML path.
This is a first bounded transfer pilot, not a prerequisite diagnostic and not proof that H1 holds.
CPU; n=3 by default; Memory-One symmetric common payoff F; four observation fields;
independent uniform initial joint action; hidden fixed horizon; learner memory persists across
episodes, opponents reset each episode, all learner memory resets per trial. No self-play required.
New generation and experiment entry point are explicit; old CLI/checkpoints retain behavior.

## Logical grammar (worker-owned logical_rules.py and compile dispatch in rules.py)

Immutable dataclasses:
- Predicate(op, left=None, right=None, children=()): eq/ne compare operands;
  and/or have exactly2 children; not exactly1. Operand is one of
  'own','other','prev_own','prev_other' or an integer action constant. Bool is not an action constant.
- Payoff(op, value=0.0, condition=None, children=()): const uses finite value;
  if uses condition and exactly2 payoff children; add exactly2 children; scale uses finite value and1 child.
- LogicalRule(n, root): n>=2; action constants in range. Validate depth<=6 and combined node count<=63.
- evaluate_logical_rule(rule,previous,actions) returns rewards from player-local perspective;
  compile_logical_rule returns existing RewardTensor global axes. compile_rule dispatches on LogicalRule.
- rule_to_dict/rule_from_dict: versioned strict JSON-safe serialization with validation; no eval.
- generate_logical_rule(n=3,seed=0,structure='atom'): structures atom/and/or/nested.
  Sample relations/operands and finite payoff values using local RNG. nested uses composed predicates
  (and/or/not), not old named reward templates. Bounded expression generation including if/add/scale.
  Defaults and exact sampling recorded in source and serialized rule; determinism across processes.
  Do not sample individual tensor entries. No promise every sampled rule is useful/nondegenerate.
  The first sampler includes current-action equality in every payoff; this is a bounded distribution bias.

## Manifest and distance (main-owned generated_tasks.py)

Seeded catalog with12 training rules from atom/and/or,6 fresh-rule test rules from those structures,
6 nested-structure test rules. Retain generation/rejection counts. Reject constant/fully own-independent
payoffs for this MA treatment suite before model results; do not filter on learner performance.
Require dependence on current own and current other somewhere in common table.
Existing isolated diagnostic families remain separate controls.
All accepted rules have unique common payoff tensors even after positive affine normalization and
consistent common action permutation. Shared relation vocabulary, new composition is not new primitive discovery.
Raw fingerprints and canonical fingerprints saved. Catalog normalizes actual reward to common range[-1,1]
using ONE positive affine transform of all entries, saved in manifest; not state-wise normalization.
This defines the actual played task. For reward-reactive opponents (win_stay's zero threshold),
the transform can change behavior relative to the source rule; do not claim joint-task strategy preservation.

Distances: mean absolute common-table difference (raw/standardized); canonical permutation minimum;
logical token multiset Jaccard dissimilarity (not semantic distance). Save nearest training rule distances.
One common consistent permutation applied to all4 action axes. Opponent policies are NOT relabeled by this
catalog distance, so quotient distance describes rule equivalence, not entire task equivalence.

Controlled rule-diversity×opponent-diversity four conditions:
- low rules = first4 accepted train rules; high = all12; low opponent = copy_previous;
- high opponents = random,copy_previous,avoid_previous.
- hold out combinations when (rule_index + opponent_index)%3==0, except ALL copy anchors are retained.
  Every high-rule rule and every high opponent still appears in training.
- test subsets: combination (held-out train-rule/opponent pairs), fresh_rules (6×3),
  structural (6×3 nested), novel_opponent (12 train rules×win_stay).
- Record seen-rule/seen-opponent/seen-pair by condition: the common combination set is pure
  new pairing only for high/high. Other conditions can also encounter new rules/opponents there.
- Nested changes predicate and reward-expression composition together; not a depth-only treatment.
- training validation uses TRAIN tasks only. Final held-out results do not select checkpoints or tune.
Manifest stores per-condition exact pairs, train/test labels, IDs, grammar, tensors/hashes and distances.
No rule/opponent/condition ID, AST, horizon or tensor reaches learner forward.

## Training and evaluation (main-owned training.py / generated_experiment.py)

Reuse full-trial PPO via optional task_sampler(seed)->(rule,opponent_kind),
evaluation_tasks=sequence[(rule,kind)]. Default legacy training path unchanged.
Sample uniformly from each condition's explicit task pairs. Same initialization per learning seed,
same total interactions for all4 conditions. Note per-task exposure differs; cannot isolate pure diversity.
Equal budget means optimizer training rollout actions. Frozen training-only validation task counts
differ by condition and final evaluation is separate; neither updates weights or selects checkpoints.
Default pilot: seeds7/17/27,40updates,8trials/update,2episodes×12steps,H32,action_set_v1.
4conditions×3seeds=12runs. No model/grammar/tuning selection on held-out results.

win_stay opponent: uses own reward; start uniform; reward>0 repeats own previous action;
otherwise uniform among other actions. Acts on pre-step observation only, RNG local, reset per episode.
Available through make_opponent; no task information supplied.

Evaluate initial/final/episode-reset neural, Random, online Q, and simple always repeat/copy baselines.
Each task has fixed exogenous seeds shared across conditions; baseline updates/priors stated.
Store per-task raw rewards, turn/episode curves, first/early/late rewards, current opponent actions,
train membership and distance. Expected finite-horizon task-aware best-response for memory-one opponents
via dynamic programming is an informed upper reference; NOT Nash regret or evidence of identifiability.
Compute posterior-greedy family-informed reference over finite manifest hypotheses on a bounded
representative subset if feasible; report stronger prior and exact scope. No active-exploration claim
without separate action-dependent information/value evidence. Do not require neural success for completion.

Output: exclusive new output directory, protocol/manifest/evaluation JSON, initial/final checkpoints,
source snapshots/hashes and standalone Korean report. Serialize enough to independently reconstruct
rules/tasks and reevaluate checkpoints. Runs ignored by Git; users execute independently via README.
Report equal-budget condition differences on the SAME test tasks, learning-seed variability and
per-task replication variability separately; no significance/causality claim from seed3 alone.
Overall trial_standard_error is pooled descriptive, not a stratified Monte Carlo or seed uncertainty.
Early/late curves include state occupancy changes; do not call the difference pure adaptation.
Default task reps=2; no aggregate pooling across splits without explicit split weights.

## Validation and interpretation

Independent grammar truth tables, symmetry, serialization rejection, local RNG, exhaustive n3/4/5
compile checks, duplicates/permutation/normalization, split membership, learner info boundary,
simultaneity, PPO replay likelihood, opponent reset, DP hand fixture, checkpoint reevaluation and
CLI overwrite refusal. Main integrates and reruns all checks. Small CLI test does not replace full pilot.
H1/H2 remain hypotheses; floor performance limits interpretation rather than refuting diversity.
An informative finite-hypothesis reference is not a proof arbitrary unseen grammar is identifiable.
An analytic two-rule information-value witness (r000/r005, copy opponent, reset state(1,0),horizon4)
was found after generating the catalog and frozen before neural results. It is an existence example,
not a preregistered representative sample. Preserve first reward in return while hiding it from
inference to isolate its information value; later observations remain available.
