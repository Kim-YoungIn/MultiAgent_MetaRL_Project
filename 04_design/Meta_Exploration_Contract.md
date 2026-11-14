# Meta exploration with learning opponents — first protocol (2026-11-14)

## Purpose and preserved scope

D-021/D-022: improve EXPLORATION through reinforcement learning, measured by the usefulness of
its experience for a controlled adapting learner, in a game against an opponent that ALSO LEARNS.
Use the existing symmetric logical generator/tensor/4-field observation, n=3, CPU, private fixed
episode horizon, random independent initial joint action. No literal parent network, evolutionary
selection, parameter generation or self-play is required. Old whole-trial RL² runs are comparators.

## Roles and inner trial (worker: adaptive_agents.py, meta_exploration_rollout.py)

OnlineQ(n,seed,alpha=.5,gamma=.95,epsilon=.2): local Random; table key is the four observation
fields; zeros for unseen observations. distribution(obs,epsilon=None) is epsilon/n plus uniform
greedy tie probability. act(obs,epsilon=None); update(obs,action,reward,next_obs,terminal=False)
performs Q+=alpha*(reward+gamma*max Q(next_obs)*(not terminal)-Q). Validate inputs and action.
snapshot() returns JSON-safe sorted entries and RNG state digest (not live mutable references).
clone() copies table AND RNG. Track updates, nonzero parameter changes, sum absolute Q changes,
and action distribution L1 changes at the updated observation. Plain random/copy policies do not
meet the learning-opponent requirement. Strategy changes can stop when a particular task is solved.

Opponent profiles: steady(alpha=.3,epsilon=.15,gamma=.95), rapid(.7,.25,.95) for training;
heldout(.5,.1,.95) for new learning-profile evaluation, not a new opponent algorithm.
Both players receive only local observations/reward, not rule or other player's current action.

run_meta_trial(rule,teacher_factory,*,seed,steps=12,opponent_profile='steady') -> JSON-safe dict.
teacher_factory(*,n,seed,student) creates an object with act(observation)->action.
student is provided only to implement the epsilon-greedy reference from its own experience;
neural teacher uses only observation and private recurrent state, never a rule, opponent profile or horizon.
Generic protocol allows history-derived learner state, not privileged task information.

At trial start create one zero-table student(alpha=.5,gamma=.95) and one zero-table opponent;
teacher recurrent memory also resets. Seed streams independently for teacher/student/opponent,
exploration episode reset and assessment reset. Return their seed values.

1. Exploration episode, K=steps: teacher chooses P0 actions; P1 chooses from its ONLINE Q policy.
   Both choose on the same pre-step observations. After each step, student learns off-policy
   from P0 experience and opponent learns from OWN experience/reward. At the episode boundary,
   bootstrap from the pre-created assessment reset observation. No artificial extra timestep.
2. Assessment episode, H=steps: student controls P0 using greedy Q with random ties(epsilon0),
   and BOTH student/opponent continue updating by the same algorithm. Teacher no longer acts.
   Retain both learned tables across this episode reset; final finite trial target bootstrap0.
3. Paired erased-memory assessment: clone the SAME opponent table/RNG at end of exploration,
   and the SAME student RNG but erase only student Q table. Use the SAME assessment reset and
   random streams, with the same continued-learning protocol. Subsequent opponent trajectories
   may diverge in response to changed P0 actions; do not replay fixed opponent actions.

Primary meta reward = assessment P0 mean reward - paired erased-student P0 mean reward.
Exploration rewards are NOT added to this signal. This target measures retained experience's
conditional downstream value, NOT pure information gain or immunity to opponent shaping.
Report absolute assessment, erased, exploration and total gameplay returns, both players' learning
audits, full local observations/actions/rewards and phase boundaries. Erased branch is a researcher
control: H extra interactions per trial for meta-training/scoring, charged explicitly for every method.
Actual path K+H; control H; total scoring simulation cost K+2H.

## Meta training (main: meta_exploration.py)

Teacher is current action_set_v1 H32 for a first controlled reference; student/opp are OnlineQ.
theta is fixed within an inner trial. Sample training logical rules uniformly from12 train catalog
rules and opponent profile uniformly from steady/rapid. Every trial initializes fresh student/opp.
Use terminal downstream meta reward as the return target for EVERY exploration action,
no discount through an invented phase boundary. Recompute teacher action likelihoods by full
exploration-sequence replay from zero recurrent state. PPO3 epochs,clip.2,value coeff.5,
entropy.01,learningrate.003,gradclip.5. Batch normalization of (G-old_value) advantages.
No gradient is propagated through Q updates or assessment trajectories; policy gradient credits
the sampled exploration sequence with its downstream score. Critic estimates that meta score.
8 trials/update,40updates,3 seeds7/17/27; no post-hoc tuning on final evaluation tasks.
Only train tasks may be used for update-progress checks. Use final weights, not best heldout score.

## Final evaluation and artifacts

Freeze meta theta, but ALWAYS permit new student/opponent inner learning. Comparators:
initial teacher, trained teacher, random exploration, epsilon-greedy(studentepsilon.2), repeat.
Same student initialization/hyperparameters, opponent initialization/profile, resets and K/H per trial;
different exploration may cause different subsequent opponent behavior, which is a valid effect.
Record stable per-task IDs/seeds so traversal order or model changes do not change exogenous seeds.

Splits: train_rule_reference(12rules×2profiles),fresh_rules(6×2),structural(6×2),
new_learning_profile(12train rules×heldout profile). Same fixed rules from prior manifest;
no model-success filtering. 2 reps/task by default. Fresh/nested are distinct catalog rules;
nested changes predicate+payoff composition together. Opponent profile variation is not strategy diversity
independent of initialization/learning speed. Per-split learning seed SD and paired task differences;
pooling fixed tasks is descriptive, not significance. Always compare meta G AND absolute assessment.

Output new exclusive directory: protocol,manifest,history,checkpoint initial/final,per-trial JSON,
source snapshots/hashes,aggregate and Korean HTML. Keep runs ignored; README independent execution.
Include audits proving opponent value/strategy changed (not merely receiving update calls), student
experience affects assessment behavior, teacher receives only permitted observations, and teacher
training targets depend on downstream control difference, not exploration reward.

## Independence, verification and interpretation

Validator owns independent tests. Hand Q/Bellman/update/distribution checks, RNG clone isolation,
simultaneous actions, phase/episode/trial reset and retained learning, no task/horizon leak,
same initialized paired controls/opponent divergence allowed, full-sequence PPO likelihood and
teacher gradient/weights from meta reward, corruption/overwrite/serialization/checkpoint reevaluation.
Main integrates and runs a full3seed experiment after small end-to-end check.
Failure to beat random/epsilon is a result; do not silently change objective/learner/opponent budget.
This experiment tests one teacher architecture+one tabular learning algorithm, not all meta exploration.
Finite difference controls retained Q rather than all learned information; state occupancy, ongoing
assessment learning and opponent shaping still affect returns. No H1/H2 or population claims.
