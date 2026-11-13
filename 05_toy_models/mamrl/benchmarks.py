"""Small, explicit tests of action tracking, information/memory, and opponents.

Task metadata is researcher-only. Reference learners receive a family prior,
never a particular rule, task label, opponent kind, seed schedule or horizon.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import random
import statistics

from .common import finite_reward, positive_integer, validate_action, validate_n, validate_seed
from .environment import MemoryOneEnv
from .opponents import make_opponent
from .rollout import EpisodeResult, Transition
from .rules import RuleSpec, RuleTerm, compile_rule


@dataclass(frozen=True)
class BenchmarkTask:
    family: str
    rule: RuleSpec
    opponent_kind: str
    label: str


def benchmark_tasks(family, *, n=3, count=48):
    n = validate_n(n)
    count = positive_integer(count, 'count')
    if family == 'hidden_action':
        variants = [BenchmarkTask(family, RuleSpec(n,(RuleTerm('action_bonus',1,a),)),
                                  'random',f'target={a}') for a in range(n)]
    elif family == 'probe_memory':
        if n < 3:
            raise ValueError('probe_memory requires n >= 3')
        variants = [BenchmarkTask(family,RuleSpec(n,(RuleTerm('probe_'+mode,1),)),
                                  'random',mode) for mode in ('repeat','copy')]
    elif family == 'opponent_response':
        variants = [BenchmarkTask(family,RuleSpec(n,(RuleTerm('coordination',1),)),kind,kind)
                    for kind in ('copy_previous','avoid_previous')]
    else:
        raise ValueError('unknown benchmark family')
    if count % len(variants):
        raise ValueError('count must be a multiple of the task strata count')
    return tuple(variants * (count // len(variants)))


class RandomLearner:
    def __init__(self, *, family, n, seed):
        self.n = validate_n(n)
        self.family = family
        self.rng = random.Random(validate_seed(seed))

    def act(self, observation):
        return self.rng.randrange(self.n)

    def observe(self, observation, action, reward, final_next_obs):
        pass

    def transition(self, observation, action, reward, next_action_obs, trial_end):
        pass

    def episode_boundary(self, final_observation):
        pass

    def reset_memory(self):
        pass


class ReactiveLearner(RandomLearner):
    """A family-informed reactive reference, with no stored task evidence."""
    def act(self, observation):
        own, other = observation.previous_own_action, observation.previous_opponent_action
        if self.family == 'probe_memory':
            if own == other:
                return own
            if observation.previous_own_reward == -.25:
                return own
            if observation.previous_own_reward == -.5:
                return other
            return self.rng.choice([a for a in range(self.n) if a not in (own,other)])
        if not observation.episode_start_flag and observation.previous_own_reward > 0:
            return own
        return self.rng.choice([a for a in range(self.n) if a != own])


class QLearner(RandomLearner):
    """Online tabular Q using precisely the four policy observation fields."""
    alpha = .5
    gamma = .99
    epsilon = .1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_memory()

    @staticmethod
    def key(observation):
        return (observation.previous_own_action, observation.previous_opponent_action,
                observation.previous_own_reward, observation.episode_start_flag)

    def values(self, observation):
        return self.q.setdefault(self.key(observation), [0.0]*self.n)

    def act(self, observation):
        values = self.values(observation)
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n)
        maximum = max(values)
        return self.rng.choice([a for a in range(self.n) if values[a] == maximum])

    def transition(self, observation, action, reward, next_action_obs, trial_end):
        values = self.values(observation)
        bootstrap = 0.0 if trial_end else max(self.values(next_action_obs))
        values[action] += self.alpha * (reward + self.gamma*bootstrap - values[action])

    def reset_memory(self):
        self.q = {}


class BeliefLearner(RandomLearner):
    """Family likelihood reference; task parameters and rules are not given."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_memory()

    def reset_memory(self):
        if self.family == 'hidden_action':
            self.possible = set(range(self.n))
        elif self.family == 'probe_memory':
            self.possible = {'repeat','copy'}
        elif self.family == 'opponent_response':
            self.possible = {'copy_previous','avoid_previous'}
        else:
            raise ValueError('unknown reference family')

    @property
    def identified(self):
        return len(self.possible) == 1

    def optimal_actions(self, observation):
        if not self.identified:
            return set()
        value = next(iter(self.possible))
        if self.family == 'hidden_action':
            return {value}
        if self.family == 'probe_memory':
            return {observation.previous_own_action if value == 'repeat' else observation.previous_opponent_action}
        own = observation.previous_own_action
        return {own} if value == 'copy_previous' else set(range(self.n))-{own}

    def act(self, observation):
        if self.identified:
            return self.rng.choice(sorted(self.optimal_actions(observation)))
        if self.family == 'hidden_action':
            return self.rng.choice(sorted(self.possible))
        if self.family == 'opponent_response':
            return observation.previous_own_action  # prior expected reward .5 vs .5/(n-1)
        own, other = observation.previous_own_action, observation.previous_opponent_action
        if own == other:
            return own
        return self.rng.choice([a for a in range(self.n) if a not in (own,other)])

    def observe(self, observation, action, reward, final_next_obs):
        if self.family == 'hidden_action':
            self.possible = {action} if reward > 0 else self.possible-{action}
        elif self.family == 'opponent_response':
            copied = final_next_obs.previous_opponent_action == observation.previous_own_action
            self.possible &= {'copy_previous' if copied else 'avoid_previous'}
        else:
            own, other = observation.previous_own_action, observation.previous_opponent_action
            def expected(mode):
                reference, alternative = (own,other) if mode == 'repeat' else (other,own)
                return 1.0 if action == reference else (-3.0 if action == alternative else (-.25 if mode=='repeat' else -.5))
            self.possible = {mode for mode in self.possible if expected(mode) == reward}
        if not self.possible:
            raise ValueError('feedback violates the declared benchmark family')


class NeuralLearner(RandomLearner):
    def __init__(self, *, model, **kwargs):
        super().__init__(**kwargs)
        import torch
        self.model = model
        self.generator = torch.Generator(device='cpu').manual_seed(kwargs['seed'] % 2**63)
        self.reset_memory()

    def reset_memory(self):
        self.hidden = self.model.initial_hidden(n=self.n)

    def act(self, observation):
        import torch
        with torch.no_grad():
            logits, _, self.hidden = self.model.forward_observation(observation, self.hidden, n=self.n)
            probabilities = torch.log_softmax(logits,dim=0).exp()
            return int(torch.multinomial(probabilities,1,generator=self.generator).item())

    def episode_boundary(self, final_observation):
        import torch
        with torch.no_grad():
            _, _, self.hidden = self.model.forward_observation(final_observation,self.hidden,n=self.n)


@dataclass(frozen=True)
class BenchmarkTrial:
    episodes: tuple[EpisodeResult, ...]
    learner_seed: int
    environment_seeds: tuple[int, ...]
    opponent_seeds: tuple[int, ...]
    post_identification_correct: int = 0
    post_identification_opportunities: int = 0
    probe_count: int = 0


def run_benchmark_trial(task, learner_factory, *, seed, episodes_per_trial=2, max_steps=12,
                        reset_memory_between_episodes=False):
    if seed is None:
        raise TypeError('an explicit seed is required')
    validate_seed(seed)
    positive_integer(episodes_per_trial,'episodes_per_trial')
    positive_integer(max_steps,'max_steps')
    if type(reset_memory_between_episodes) is not bool:
        raise TypeError('reset_memory_between_episodes must be bool')
    seeds = random.Random(seed)
    learner_seed = seeds.getrandbits(63)
    env_seeds = tuple(seeds.getrandbits(63) for _ in range(episodes_per_trial))
    opp_seeds = tuple(seeds.getrandbits(63) for _ in range(episodes_per_trial))
    n = task.rule.n
    learner = learner_factory(family=task.family,n=n,seed=learner_seed)
    tracker = BeliefLearner(family=task.family,n=n,seed=0)
    rule = compile_rule(task.rule)
    envs = [MemoryOneEnv(rule,max_steps=max_steps) for _ in range(episodes_per_trial)]
    initial = [env.reset(seed=s) for env,s in zip(envs,env_seeds)]
    episodes = []
    correct = opportunities = probes = 0
    for e,env in enumerate(envs):
        observations = initial[e]
        opponent = make_opponent(task.opponent_kind)
        state = opponent.initial_state(n=n,seed=opp_seeds[e])
        transitions = []
        totals = [0.0,0.0]
        for step in range(max_steps):
            own = observations[0]
            action = validate_action(learner.act(own),n)
            if tracker.identified:
                opportunities += 1
                correct += action in tracker.optimal_actions(own)
            if task.family == 'probe_memory' and action not in (own.previous_own_action,own.previous_opponent_action):
                probes += 1
            other = opponent.act(observations[1],state,n=n)
            state = other.state
            result = env.step(action,other.action)
            learner.observe(own,action,result.rewards[0],result.observations[0])
            tracker.observe(own,action,result.rewards[0],result.observations[0])
            last = step == max_steps-1
            trial_end = last and e == episodes_per_trial-1
            # The next actionable state across an internal truncation is the reset state.
            next_action_obs = initial[e+1][0] if last and not trial_end else result.observations[0]
            learner.transition(own,action,result.rewards[0],next_action_obs,trial_end)
            transitions.append(Transition(observations,(action,other.action),result.rewards,
                                          result.observations,result.terminated,result.truncated))
            for p in (0,1):
                totals[p] += result.rewards[p]
            observations = result.observations
        episodes.append(EpisodeResult(tuple(transitions),tuple(totals)))
        if e < episodes_per_trial-1:
            learner.episode_boundary(observations[0])
            if reset_memory_between_episodes:
                learner.reset_memory()
    return BenchmarkTrial(tuple(episodes),learner_seed,env_seeds,opp_seeds,correct,opportunities,probes)


def best_response_bound(task):
    return 1/(task.rule.n-1) if task.opponent_kind=='avoid_previous' else 1.0


def _stratified_se(values,labels):
    groups = {label:[v for v,l in zip(values,labels) if l==label] for label in set(labels)}
    if any(len(v)<2 for v in groups.values()):
        return None
    return math.sqrt(sum((len(v)/len(values))**2*statistics.variance(v)/len(v) for v in groups.values()))


def evaluate_benchmark(learner_factory,tasks,*,seed=918273,episodes_per_trial=2,max_steps=12,
                       reset_memory_between_episodes=False):
    tasks = tuple(tasks)
    if not tasks or len({t.family for t in tasks})!=1:
        raise ValueError('evaluate one nonempty benchmark family at a time')
    validate_seed(seed)
    seeds = random.Random(seed)
    trial_seeds = [seeds.getrandbits(63) for _ in tasks]
    trials = [run_benchmark_trial(t,learner_factory,seed=s,episodes_per_trial=episodes_per_trial,
                                  max_steps=max_steps,reset_memory_between_episodes=reset_memory_between_episodes)
              for t,s in zip(tasks,trial_seeds)]
    rewards = [[step.rewards[0] for ep in trial.episodes for step in ep.transitions] for trial in trials]
    steps = episodes_per_trial*max_steps
    returns = [sum(r) for r in rewards]
    means = [r/steps for r in returns]
    labels = [t.label for t in tasks]
    opportunities = sum(t.post_identification_opportunities for t in trials)
    fingerprints = [hashlib.sha256(json.dumps(asdict(t),sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in tasks]
    groups = {}
    for label in sorted(set(labels)):
        indices = [i for i,x in enumerate(labels) if x==label]
        groups[label] = {'trials':len(indices),'mean_reward_per_step':statistics.mean(means[i] for i in indices),
                         'best_response_bound':best_response_bound(tasks[indices[0]]),
                         'mean_regret_per_step':statistics.mean(best_response_bound(tasks[i])-means[i] for i in indices)}
    return {
        'trials':len(trials),'steps_per_trial':steps,'episodes_per_trial':episodes_per_trial,'max_steps':max_steps,
        'mean_reward_per_step':statistics.mean(means),'standard_error':_stratified_se(means,labels),
        'by_task':groups,'trial_returns':returns,'trial_seeds':trial_seeds,'task_fingerprints':fingerprints,
        'task_labels':labels,'mean_reward_by_episode':[statistics.mean(t.episodes[e].total_rewards[0]/max_steps for t in trials) for e in range(episodes_per_trial)],
        'mean_reward_by_turn':[statistics.mean(r[k] for r in rewards) for k in range(steps)],
        'early_reward':statistics.mean(sum(r[:min(3,steps)])/min(3,steps) for r in rewards),
        'late_reward':statistics.mean(sum(r[-min(3,steps):])/min(3,steps) for r in rewards),
        'episode2_first_action':statistics.mean(t.episodes[1].transitions[0].rewards[0] for t in trials) if episodes_per_trial>1 else None,
        'post_identification_accuracy':sum(t.post_identification_correct for t in trials)/opportunities if opportunities else None,
        'post_identification_opportunities':opportunities,'probe_count':sum(t.probe_count for t in trials),
        'probe_rate':sum(t.probe_count for t in trials)/(len(trials)*steps),
        'mean_regret_per_step':statistics.mean(best_response_bound(t)-m for t,m in zip(tasks,means)),
    }


def paired_benchmark_difference(left,right):
    for metrics in (left,right):
        steps = positive_integer(metrics['steps_per_trial'],'steps_per_trial')
        episodes = positive_integer(metrics['episodes_per_trial'],'episodes_per_trial')
        horizon = positive_integer(metrics['max_steps'],'max_steps')
        count = positive_integer(metrics['trials'],'trials')
        if steps != episodes*horizon:
            raise ValueError('steps_per_trial must equal episodes_per_trial * max_steps')
        if any(len(metrics[key]) != count for key in
               ('trial_returns','trial_seeds','task_fingerprints','task_labels')):
            raise ValueError('paired task/seed/return lengths must match trial count')
    for key in ('steps_per_trial','episodes_per_trial','max_steps','trial_seeds','task_fingerprints','task_labels'):
        if left[key]!=right[key]:
            raise ValueError(f'paired benchmark mismatch: {key}')
    steps = left['steps_per_trial']
    if not left['trial_returns'] or len(left['trial_returns'])!=len(right['trial_returns']):
        raise ValueError('paired return length mismatch')
    differences = [(finite_reward(a)-finite_reward(b))/steps for a,b in zip(left['trial_returns'],right['trial_returns'])]
    return {'mean':statistics.mean(differences),'standard_error':_stratified_se(differences,left['task_labels']),
            'trial_differences':differences}


def probe_memory_witness():
    result = []
    for mode,start,actions in (('repeat',(2,1),((0,2),(0,1))),('copy',(1,2),((0,0),(0,1)))):
        seed = 0
        while True:
            rng = random.Random(seed)
            if (rng.randrange(3),rng.randrange(3))==start:
                break
            seed += 1
        rule = RuleSpec(3,(RuleTerm('probe_'+mode,1),))
        tensor = compile_rule(rule)
        env = MemoryOneEnv(tensor,max_steps=3)
        obs = env.reset(seed=seed)
        observations = [asdict(obs[0])]
        rewards = []
        for own,other in actions:
            step = env.step(own,other)
            obs = step.observations
            observations.append(asdict(obs[0])); rewards.append(step.rewards[0])
        values = [tensor[0,0,1,a,0] for a in range(3)]
        result.append({'mode':mode,'initial_state':list(start),'reset_seed':seed,
                       'prefix_actions':[list(x) for x in actions],'observations':observations,
                       'rewards':rewards,'final_observation':asdict(obs[0]),
                       'optimal_actions':[a for a in range(3) if values[a]==max(values)]})
    return result
