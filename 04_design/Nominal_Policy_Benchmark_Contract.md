# Nominal Action Policy / Functional Benchmarks — v1

## Scope and decisions

Correct the known ordinal-label bias while preserving ordinal_v1 as a reference.
CPU, four-field observation, hidden RuleSpec/opponent ID, random reset, fixed hidden horizon,
simultaneous actions and finite-trial PPO remain unchanged. This milestone establishes three
functional benchmarks, not H1/H2 or novel-rule generalization.

## New model (action_set.py; worker)

`ActionSetActorCritic(hidden_size=32, seed=0, *, memoryless=False)` subclasses RecurrentActorCritic
for existing Policy/collector compatibility, but initializes its own shared modules:
GRUCell(5,H), actor Linear(H,H)/Tanh/Linear(H,1), critic Linear(H,H)/Tanh/Linear(H,1).
No scalar action indices, one-hot fixed dimensions or action-specific trainable parameters.
For each candidate a, inputs are [a==previous_own, a==previous_opponent,
tanh(previous_own_reward/4), episode_start_flag, 1/n].

- `initial_hidden(*, n)` -> zeros float32 CPU (n,H).
- `forward_observation(obs, hidden, *, n)` -> logits(n), value scalar, next_hidden(n,H).
  Shared GRU updates each row; shared actor scores each row; critic consumes mean(next_hidden,dim=0).
- memoryless=True validates but zeroes incoming hidden before each forward. This is a trained
  reactive/zero-hidden-every-action ablation; not an independent larger feedforward algorithm.
- Arbitrary bijection p maps hidden rows as h_internal[p(a)]=h_physical[a] and observed action IDs
  through p. logits/next_hidden transform the same way; value is invariant (numerical tolerance).
- Parameters independent of n; n=3/4/5 execute with the same weights. Hidden state is nH rather
  than ordinal H: state capacity/parameter counts must be reported; architecture comparisons are
  not a single-factor causal encoding ablation. No static ID prior; constant privileged label
  memorization at trial start is not available. Evaluate experience-based action tracking instead.

## Compatibility/checkpoints (recurrent.py; worker)

- ordinal RecurrentActorCritic.initial_hidden accepts optional n keyword, retaining no-arg behavior.
  RecurrentPolicy.initial_state passes n to the model.
- Existing format_version=1 checkpoint save/load remains exact ordinal behavior.
- New format_version=2: architecture={kind:'action_set_v1'|'action_set_ff_v1',hidden_size:H},state_dict.
  Exact keys/shapes/float32 CPU/finite checks; weights_only load; refuse existing files.
- Shared Policy adapter supports both model shapes; policy state contains no rule/opponent ID or horizon.
- TrainConfig.architecture defaults ordinal_v1 for historical library compatibility. New training CLI
  defaults action_set_v1, with explicit --architecture ordinal_v1 to reproduce historical runs.
- collect_trial/replay_trial always call initial_hidden(n=spec.n), including reset ablation.

## Logical benchmark rules (main)

A `hidden_action`: existing action_bonus, target uniform 0..n-1. Random opponent. Informed search
reference removes failed actions and repeats a rewarded action; no target is passed to the policy.

B `probe_memory` (n>=3): hidden mode repeat/copy fixed throughout trial. New RuleTerm kinds
probe_repeat/probe_copy, coefficient scales the following payoff for each player:
  if current own == chosen reference (previous own / previous opponent): +1;
  elif current own == the other reference: -3;
  otherwise: -.25 for repeat mode / -.5 for copy mode (an informative probe).
When both references coincide they are correct +1; other actions still reveal mode via probe reward.
Generator's existing default family/primitive sampling is unchanged; these are explicit templates.
Reachable full-history memory witness must reach the SAME current observation (own0,opp1,reward1,
startFalse) in both modes but have different next optimal actions0/1. Prefixes include observed
probe feedback, so full histories distinguish modes while current observation does not.

C `opponent_response`: coordination reward current own==current opponent (+1 else0), fixed rule.
Opponent kind is either copy_previous (copies learner previous action) or avoid_previous (uniformly
chooses an action other than learner previous action). Hidden kind fixed throughout trial.
Best-response expectations are1 and1/(n-1), respectively. Report per-kind results/regret separately.
No active information action is necessary in C; it is an opponent-identification benchmark.

## Opponents / training (main)

opponents.py contains make_opponent(kind) -> existing Policy API, kinds random/copy_previous/avoid_previous.
`collect_trial(..., opponent_kind='random')` records opponent_kind in TrialRollout (default preserves semantics).
TrainConfig adds opponent_mode='random'|'copy_previous'|'avoid_previous'|'mixed'. Mixed samples
copy/avoid using a separate RNG derived from the existing task seed, without consuming new default RNG draws.
Rule family probe_memory samples repeat/copy; opponent_response always generates coordination rule.
`evaluate(...,opponent_kinds=None)` uses Random by default; supplied list must align with specs.
Training evaluations match config opponent distribution; legacy default numerical paths remain unchanged.
No opponent kind is passed to neural forward or policy observations.

## Benchmark reference API (main, benchmarks.py)

BenchmarkTask(family, rule, opponent_kind, label) is researcher metadata; labels/modes/RuleSpec are
never passed to learner constructors. Learner constructors receive only family prior, n and seed.
Reference learners: Random, reactive repeat-after-positive/otherwise-switch, tabular Q,
family-informed belief. Belief has exact FAMILY likelihood knowledge, not task parameter knowledge;
label as informed reference, not a learned method with equal prior knowledge.
Q uses full four-field observation as its discrete key, alpha=.5,gamma=.99,epsilon=.1,
uniform random tie breaking. Q/belief reset at trial start and retain across episodes.
Q updates during evaluation; recurrent weights are frozen. Distinguish these adaptation mechanisms.

Reference interface: act(obs), observe(obs,action,reward,final_next_obs),
transition(obs,action,reward,next_action_obs,trial_end), episode_boundary(final_obs), reset_memory().
All inputs are learner-visible data; no other-player reward, rule or horizon. Q bootstraps across
intermediate reset from the next episode's ACTUAL reset observation; final trial bootstrap is0.
Neural boundary consumes the final observation once without an action/RNG draw, matching training.
A reset-memory ablation clears memory but retains action RNG stream.

BenchmarkTask sampling balances target/mode/opponent strata. Same external seeds across methods.
Record task-wise return, per-step mean, episode2 first-action reward, early/late reward and regret
against the task-aware best-response bound (can be negative in individual stochastic samples).
Do not pool reward across families; B contains larger penalties. Stratum-level results retained.

## Execution plan

Implement and test model equivariance, checkpoint compatibility, reward truth tables and actual
memory witness first. Then paired pilot with seeds7/17/27, new recurrent and trained reactive variants,
3 families,40 updates each (18 training runs). Other PPO defaults retained, evaluation48 trials/family,
root918273. Save model/initial checkpoints, full config, task descriptions, metrics/raw returns and HTML.
No success threshold in CI; report failures unchanged. Generic Generator expansion remains subsequent.
