"""Controlled inner trial with a learning student and learning opponent.

Each teacher action is chosen before the simultaneous opponent action. The
learner and opponent observe and update only from their own experience. The
erased-student control branches from the same learned opponent and reset.
"""

from __future__ import annotations

from dataclasses import asdict
import random
import statistics

from .adaptive_agents import OnlineQ, ExpectedSARSA
from .common import positive_integer, validate_action, validate_seed
from .environment import MemoryOneEnv
from .rules import compile_rule
from .tensor import RewardTensor
from .structured_games import StructuredGame, StructuredEnv


OPPONENT_PROFILES = {
    'steady': {'alpha': .3, 'epsilon': .15, 'gamma': .95},
    'rapid': {'alpha': .7, 'epsilon': .25, 'gamma': .95},
    'heldout': {'alpha': .5, 'epsilon': .1, 'gamma': .95},
    'sarsa': {'alpha': .5, 'epsilon': .2, 'gamma': .95, 'algorithm':'expected_sarsa'},
    'sarsa_heldout': {'alpha': .7, 'epsilon': .1, 'gamma': .95, 'algorithm':'expected_sarsa'},
}

OPPONENT_SUITES = {
    'q_learning': {'train':('steady','rapid'),'heldout':'heldout'},
    'mixed_learning': {'train':('steady','sarsa'),'heldout':'sarsa_heldout'},
}


def _make_env(tensor, steps, seed):
    env = StructuredEnv(tensor, max_steps=steps) if isinstance(tensor, StructuredGame) else MemoryOneEnv(tensor, max_steps=steps)
    return env, env.reset(seed=seed)


def _event(observations, actions, result, learning_next, terminal, opponent_probabilities):
    return {'observations':[asdict(item) for item in observations],
            'actions':list(actions), 'rewards':list(result.rewards),
            'next_observations':[asdict(item) for item in result.observations],
            'learning_next_observations':[asdict(item) for item in learning_next],
            'terminated':result.terminated, 'truncated':result.truncated,
            'terminal':terminal,
            'opponent_action_probabilities':list(opponent_probabilities)}


def _mean(events, player=0):
    return statistics.mean(item['rewards'][player] for item in events)


def run_meta_trial(rule, teacher_factory, *, seed, steps=12, opponent_profile='steady', assessment_steps=None, state_capture=None):
    """Drive one stream with a local teacher; identical contract to batched trials."""
    if not callable(teacher_factory):
        raise TypeError('teacher_factory must be callable')
    stream = meta_trial_stream(rule, seed=seed, steps=steps, opponent_profile=opponent_profile,assessment_steps=assessment_steps,state_capture=state_capture)
    observation, n, teacher_seed, student = next(stream)
    teacher = teacher_factory(n=n, seed=teacher_seed, student=student)
    if not callable(getattr(teacher, 'act', None)):
        raise TypeError('teacher_factory must return an object with act(observation)')
    while True:
        try:
            observation, _, _, _ = stream.send(teacher.act(observation))
        except StopIteration as result:
            return result.value


def meta_trial_stream(rule, *, seed, steps=12, opponent_profile='steady', assessment_steps=None, state_capture=None):
    """Collect exploration, downstream assessment, and a paired Q-erasure control.

    Assessment is run twice from identical environment and opponent RNG/table
    state. On the control branch, only student Q values are erased; the student's
    RNG and its accumulated audit counters are retained. Both agents keep
    learning, so future opponent actions are allowed to diverge between branches.
    """
    positive_integer(steps, 'steps')
    assessment_steps=steps if assessment_steps is None else positive_integer(assessment_steps,'assessment_steps')
    if validate_seed(seed) is None:
        raise TypeError('trial requires an explicit seed')
    if opponent_profile not in OPPONENT_PROFILES:
        raise ValueError('unknown opponent profile')
    tensor = rule if isinstance(rule, (RewardTensor, StructuredGame)) else compile_rule(rule)
    seed_rng = random.Random(seed)
    names = ('teacher', 'student', 'opponent', 'exploration_reset', 'assessment_reset')
    seeds = {name:seed_rng.getrandbits(63) for name in names}
    student = OnlineQ(tensor.n, seeds['student'], alpha=.5, gamma=.95, epsilon=.2)
    profile=OPPONENT_PROFILES[opponent_profile]
    opponent_class=ExpectedSARSA if profile.get('algorithm')=='expected_sarsa' else OnlineQ
    opponent = opponent_class(tensor.n, seeds['opponent'], **{k:v for k,v in profile.items() if k!='algorithm'})
    student_initial = student.snapshot()
    opponent_initial = opponent.snapshot()
    exploration_env, observations = _make_env(tensor, steps, seeds['exploration_reset'])
    assessment_env, assessment_start = _make_env(tensor, assessment_steps, seeds['assessment_reset'])
    exploration = []
    for index in range(steps):
        # Simultaneous actions: neither learner sees the other's current action.
        a0 = validate_action((yield (observations[0], tensor.n, seeds['teacher'], student)), tensor.n)
        opponent_probs = opponent.distribution(observations[1])
        a1 = opponent.act(observations[1])
        result = exploration_env.step(a0, a1)
        learning_next = assessment_start if index == steps-1 else result.observations
        student.update(observations[0], a0, result.rewards[0], learning_next[0], terminal=False)
        opponent.update(observations[1], a1, result.rewards[1], learning_next[1], terminal=False)
        exploration.append(_event(observations, (a0,a1), result, learning_next, False, opponent_probs))
        observations = result.observations

    student_before = student.snapshot()
    opponent_before = opponent.snapshot()
    erased_student, erased_opponent = student.clone(), opponent.clone()
    erased_student.erase_values()
    erased_student_before = erased_student.snapshot()
    erased_opponent_before = erased_opponent.snapshot()
    erased_env, erased_start = _make_env(tensor, assessment_steps, seeds['assessment_reset'])
    if erased_start != assessment_start:
        raise RuntimeError('paired assessment reset mismatch')

    if state_capture is not None:
        state_capture.update(student=student.clone(),opponent=opponent.clone(),
                             tensor=tensor,seeds=dict(seeds),assessment_steps=assessment_steps)

    assessment = assess_pair(assessment_env, assessment_start, student, opponent,assessment_steps)
    erased_assessment = assess_pair(erased_env, erased_start, erased_student, erased_opponent,assessment_steps)
    exploration_mean = _mean(exploration)
    assessment_mean = _mean(assessment)
    erased_mean = _mean(erased_assessment)
    return {'n':tensor.n, 'seed':seed, 'opponent_profile':opponent_profile, 'seeds':seeds,
            'exploration':exploration, 'assessment':assessment, 'erased_assessment':erased_assessment,
            'exploration_mean':exploration_mean, 'assessment_mean':assessment_mean,
            'erased_mean':erased_mean, 'meta_reward':assessment_mean-erased_mean,
            'gameplay_mean':statistics.mean((exploration_mean,assessment_mean)) if steps==assessment_steps else (steps*exploration_mean+assessment_steps*assessment_mean)/(steps+assessment_steps),
            'exploration_return':sum(item['rewards'][0] for item in exploration),
            'assessment_return':sum(item['rewards'][0] for item in assessment),
            'erased_return':sum(item['rewards'][0] for item in erased_assessment),
            'opponent_exploration_mean':_mean(exploration,1),
            'opponent_assessment_mean':_mean(assessment,1),
            'opponent_erased_mean':_mean(erased_assessment,1),
            'student_initial':student_initial, 'opponent_initial':opponent_initial,
            'student_before_assessment':student_before, 'opponent_before_assessment':opponent_before,
            'erased_student_before_assessment':erased_student_before,
            'erased_opponent_before_assessment':erased_opponent_before,
            'student_final':student.snapshot(), 'opponent_final':opponent.snapshot(),
            'erased_student_final':erased_student.snapshot(),
            'erased_opponent_final':erased_opponent.snapshot(),
            'phase_steps':{'exploration':steps,'assessment':assessment_steps,'erased_assessment':assessment_steps},
            'environment_steps_with_control':steps+2*assessment_steps,
            'opponent_nonuniform_action_events':sum(
                sum(abs(prob-1/tensor.n) for prob in event['opponent_action_probabilities'])>1e-12
                for event in exploration+assessment),
            'opponent_nonuniform_control_events':sum(
                sum(abs(prob-1/tensor.n) for prob in event['opponent_action_probabilities'])>1e-12
                for event in erased_assessment)}

def assess_pair(env, initial, learner, adversary, steps, *, learn_student=True, learn_opponent=True):
    observations = initial
    records = []
    for index in range(steps):
        # Student always assesses greedily; adversary follows its learning profile.
        a0 = learner.act(observations[0], epsilon=0.0)
        opponent_probs = adversary.distribution(observations[1])
        a1 = adversary.act(observations[1])
        result = env.step(a0,a1)
        terminal = index == steps-1
        if learn_student:
            learner.update(observations[0],a0,result.rewards[0],result.observations[0],terminal=terminal)
        if learn_opponent:
            adversary.update(observations[1],a1,result.rewards[1],result.observations[1],terminal=terminal)
        records.append(_event(observations,(a0,a1),result,result.observations,terminal,opponent_probs))
        observations = result.observations
    return records
