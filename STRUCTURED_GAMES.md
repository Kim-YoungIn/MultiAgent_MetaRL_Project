# Structured finite-state games — optional MAMRL extension

This extension generates **executable**, seeded two-player games with bounded
joint-action history and persistent finite state. It keeps existing
`RewardTensor`, `RuleSpec`, `MemoryOneEnv`, and D-031 game definitions intact.
It is a game-generator/rollout extension, **not a validated new Meta-RL result**.

The extension is present in this checkout. Subsequent experiments used this simulator for state-preserving evaluation, cross-size transfer and exact two-turn planning. [Experimental results](research/REPORT.md) · [Reproduction](research/REPRODUCE.md)

## Installing into an unmodified v0.5 checkout

From the repository root, apply the supplied patch:

```bash
git apply --check structured_games.patch
git apply structured_games.patch
```

The patch adds `05_toy_models/mamrl/structured_games.py` and
`tests/test_structured_games.py`, and makes a three-line compatibility change
to `05_toy_models/mamrl/meta_exploration_rollout.py`. Existing old-game code paths
retain their behavior.

## Quick start

```bash
PYTHONPATH=05_toy_models python -m mamrl.structured_games \
  --n 3 --depth 3 --family mixed --seed 17 --steps 32 \
  --output /tmp/example_structured_game.json

PYTHONPATH=05_toy_models python -m unittest \
  discover -s tests -p 'test_structured_games.py' -v
```

Programmatic usage:

```python
from mamrl.structured_games import (
    StructuredEnv, StructuredGame, StructuredQ,
    generate_structured_rule, generate_catalog, from_legacy,
)
from mamrl.rollout import run_episode
from mamrl.policy import RandomPolicy

spec = generate_structured_rule(
    n=4, memory_depth=3, seed=42, family='mixed',
    state_levels=(3, 4, 3), stochastic=True,
)
assert StructuredGame.from_dict(spec.to_dict()) == spec

result = run_episode(
    StructuredEnv(spec, max_steps=80),
    (RandomPolicy(), RandomPolicy()),
    seed=5, policy_seeds=(10, 11),
)
print(result.total_rewards)

# Persistent state is available to the simulator; only public variables
# enter policy observations. Hidden variables are never sent to agents.
obs = result.transitions[-1].next_observations[0]
print(obs.public_state, obs.action_history)

catalog = generate_catalog(count=16, seed=9, memory_depth=3)
print([(game.family, game.fingerprint()) for game in catalog])
```

A legacy rule remains exactly the same payoff function:

```python
from mamrl.rules import generate_rule
legacy = from_legacy(generate_rule(n=3,seed=7))
assert legacy.legacy_tensor().n == 3
```

## Meta-exploration compatibility

After applying the compatibility patch, the existing teacher factory and
unbatched rollout can accept a `StructuredGame` directly:

```python
from mamrl.structured_games import generate_structured_rule
from mamrl.meta_exploration import RandomExplorer
from mamrl.meta_exploration_rollout import run_meta_trial

spec = generate_structured_rule(seed=17,memory_depth=3)
trial = run_meta_trial(spec,RandomExplorer,seed=91,steps=12,
                       assessment_steps=24,opponent_profile='sarsa')
print(trial['meta_reward'],trial['assessment_mean'])
```

`meta_batch.collect_batch` also accepts `(StructuredGame, seed, profile)` jobs.
Existing `OnlineQ` and `ContextActor` run through this interface but **ignore
`public_state` and the added `action_history` fields**. For a state-aware
Q baseline, use `StructuredQ`, which keys on the observed state/history and
retains the original Q-learning update method. This is *not* automatically
plugged into D-031 training. The existing `experiment/` panel builder and
training entry decoder still accept the legacy relational rule schema; a new
experiment protocol/manifest and appropriately state-aware models are needed
before drawing research conclusions with these games.

The patched legacy meta trial **resets persistent environment state at the
exploration/assessment boundary** (a reset-state transfer measurement). For
continuous adaptation, a separate continuous-state evaluation protocol is
required. The `StructuredEnv` itself maintains state **within** each episode.

## Defined semantics

For joint actions `(a,b)`, previous action history `h` (most recent first),
and old shared state `s`, the executable game is:

```text
reward_i = scale * (
    base[i, previous_a, previous_b, a, b]
    + sum(clause.value for each locally matched reward clause)
)

new_state_j = clamp(
    old_state_j + sum(matching transition deltas),
    0, variable.levels - 1
)
```

All clauses read the **old** state/history and the **current** simultaneous
joint action. Transition deltas are combined before clipping. Stochastic
transition gates use the environment's local seeded RNG. Generated rewards
are conservatively scaled to [-1, 1] for every possible state/history; this
uniform positive scaling does not change the preference ordering *within*
a game. `from_legacy` preserves the original tensor without scaling.

Observations extend the existing frozen `Observation` dataclass with:

- `public_state`: one value per state variable, or `None` if hidden;
- `action_history`: at most `memory_depth` previous joint actions in the
  **player's own perspective**, most recent first.

The four existing observation fields and the `StepResult` layout are unchanged.
At episode reset, persistent state returns to configured initial values and
the history is initialized with seeded random joint actions. The current game
rule is fixed within an episode; `reset(rule=...)` permits changing it between
episodes when the action count is unchanged.

The generator has four motifs: `resource`, `sequence`, `reciprocity`, and
`mixed`. Role assignments and payoff coefficients vary by seed. The `mixed`
family combines investment/harvest state transitions, a multi-step sequence,
and an occasionally hidden trust variable. Parameters, actions, memory depth,
state capacities, and transition stochasticity are configurable. Rules are
sampled once, then executed deterministically except for explicitly declared
stochastic transition gates.

## Scope and limitations

- Catalog uniqueness is checked with **exact serialized fingerprints**, not
  action-relabeling-invariant game equivalence or strategic equivalence.
  For H1/H2 studies, separately audit payoff, transition, strategic, and
  relabeling equivalences *before* exposing held-out results.
- The generator is a controlled set of templates, not a proof of active
  information value. Measure information-gathering costs and strategic
  differentiation for each experimental distribution.
- A larger `memory_depth` does not automatically imply longer effective
  dependence: inspect which predicates are sampled and which states are
  reachable. A state-aware learner may face a much larger key space.
- Public state and player history are visible to new-aware agents, but old
  teacher/student modules see only the original four fields. For research
  on partially observable adaptation, explicitly specify which fields the
  model can use.
- The package uses only Python's standard library **plus the existing MAMRL
  core**. Neural meta training still requires PyTorch as before.
