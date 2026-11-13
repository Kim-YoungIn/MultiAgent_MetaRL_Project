# Action Label / Gradient Audit — v1

## Scope

Analyze the fixed-target action-1 learning asymmetry without replacing the baseline model,
reward semantics, PPO objective, or CPU configuration. Preserve earlier artifacts.
This is a diagnostic intervention, not an adopted policy architecture revision.

## Predeclared comparisons

- n=3, hidden32, learning seeds7/17/27, physical fixed targets0/1/2, 40updates,
  8trials/update,2episodes×12steps,3epochs,Adam .003; remaining PPO defaults unchanged.
- Physical-to-internal permutations p(a)=(a+k)%3 for k=0,1,2; full3×3×3 matrix27new training runs.
  Re-run identity conditions through the same current path rather than relying on historical checkpoints.
- Map own/opponent action labels in Observation through p. Reward/start flag unchanged.
  Return logits in physical order: physical_logits[a]=internal_logits[p(a)].
- All mappings are fixed independently of the target. No target information is passed to policies.
- Same seed gives identical initial internal weights. Physical environment/opponent seed streams are matched.
  Learner trajectories and categorical sampling outcomes need not match after relabeling.
- Fixed-target evaluation uses48trials with seed918273. Preserve initial/trained/Random comparisons,
  initial-to-trained change, and group both by physical target and by internal target=p(physical target).
- Permutation changes both GRU observation encoding and actor candidate encoding; do not attribute its
  causal effect to the actor alone. Only these three cyclic mappings are tested.

## Production extension

`train(config, *, checkpoint_callback=None, model_factory=None)`.
None preserves the original construction exactly. Factory receives hidden_size/seed keywords,
returns a RecurrentActorCritic-compatible module. Used only for analysis variants.
Baseline config, checkpoint format, inference CLI and default architecture remain unchanged.

## mamrl.label_audit (worker ownership)

- `PermutedActorCritic(hidden_size=32, seed=0, *, permutation=(0,1,2))` subclass of RecurrentActorCritic.
  `permutation` is a validated tuple bijection of 0..n-1. Forward checks n==len(permutation),
  validates original observation before mapping, then returns physical-order logits.
  No extra trainable parameters or state_dict keys. initial_hidden/value contract retained.
- `reward_audit(n=3)` → dict. Exhaustively compare tensor/step rewards for all targets to
  hand predicate r_i=1[current_action_i==target], independent of previous actions.
  Return passed, tensor_entries_checked, environment_steps_checked; fail loudly on disagreement.
- `actor_gradient_audit(model, *, n=3)` → dict for an unpermuted base model's internal coordinates.
  Enumerate n² observations own/opponent, reward0,start=True,zero hidden; compute GRU outputs no_grad.
  Copy actor and all combined inputs to float64. Objective J_t is mean p(target=t).
  For every target report probability, norm(mean gradient), mean(norm per-observation gradient),
  per-observation norms and finite-difference check of largest mean-gradient coordinate at eps1e-4/1e-5/1e-6.
  Finite differences and autograd use the same detached-hidden float64 objective.
  Include n3 linear ordinal-slope fixture derivatives at w=0 for softmax(w*[0,.5,1]):[-1/6,0,+1/6].
  This fixture is not the full neural model. Preserve global RNG and source parameters/grad buffers.
- `loss_gradient_audit(model, config)` → first-update PPO batch diagnostics (no optimizer update).
  Reproduce training task/rollout seeds, GAE and batch advantage normalization.
  Decompose raw policy/value/entropy gradients and weighted gradients on actor/GRU/critic groups;
  report full total norm, global clip multiplier and shared-GRU policy/value cosine (None if zero norm).
  Value and entropy are raw .5MSE and mean entropy; weights are value_coefficient and -entropy_coefficient.
  Use real clipped surrogate policy objective, constant old logprob/GAE targets.
  Do not infer optimization causality merely from norm or scalar loss size.
- `render_label_audit_report(record)` → offline escaped HTML summary plus detailed raw results.

## mamrl.audit_labels (main ownership)

CLI default fresh output directory, full matrix. Tiny test overrides seeds/updates/evaluation_repeats/
trials_per_update/max_steps/hidden_size; every override stored in protocol.json.
Artifacts: audit.json,report.html,protocol.json, per-run internal_weights.pt and raw JSON.
Weights are in internal coordinates. Restoring a physical policy requires its recorded permutation;
these internal weights must not be presented as an ordinary physical-label checkpoint.
Factory/loss audits have independent tests; do not enforce a learning-success threshold in CI.

Record schema_version1: protocol,reward_audit,initial_gradient_audits(list),loss_gradient_audits(list),
runs(list),aggregates(dict). Run fields: seed,physical_target,internal_target,permutation,shift,config,
comparisons(initial/random/trained),initial_to_trained,checkpoint,checkpoint_sha256,raw_record,history,
evaluation_specs,evaluation_seed,trained_gradient_audit.
Initial gradient audits: {seed,audit}; loss audits: {seed,physical_target,audit} for baseline initial models.
Aggregates by_internal_target and by_physical_target: mean_reward,sd_across_runs,run_count,
mean_initial_to_trained. Raw seed/permutation data retained, repeated conditions not called independent learning seeds.
