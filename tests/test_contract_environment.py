"""Environment and policy checks derived from the written public contract."""
import dataclasses
import itertools
import random
import unittest

from mamrl.environment import MemoryOneEnv, Observation
from mamrl.tensor import RewardTensor
from mamrl.policy import PolicyOutput, RandomPolicy
from mamrl.rollout import run_episode


def labelled_tensor(n):
    return RewardTensor(n, [10000*p + 1000*x + 100*y + 10*a + b
                            for p, x, y, a, b in itertools.product(range(2), range(n), range(n), range(n), range(n))])


class EnvironmentContract(unittest.TestCase):
    def test_reset_observations_and_global_reward_order(self):
        for n in (3,4,5):
            env = MemoryOneEnv(labelled_tensor(n), max_steps=2)
            obs = env.reset(seed=123)
            x, y = obs[0].previous_own_action, obs[1].previous_own_action
            self.assertEqual(obs[0].previous_opponent_action, y)
            self.assertEqual(obs[1].previous_opponent_action, x)
            self.assertEqual((obs[0].previous_own_reward, obs[1].previous_own_reward), (0,0))
            self.assertTrue(obs[0].episode_start_flag and obs[1].episode_start_flag)
            self.assertEqual({f.name for f in dataclasses.fields(Observation)},
                             {'previous_own_action', 'previous_opponent_action',
                              'previous_own_reward', 'episode_start_flag'})
            with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
                obs[0].episode_start_flag = False
            a, b = 1, 2
            first = env.step(a,b)
            rewards = (1000*x + 100*y + 10*a + b, 10000 + 1000*x + 100*y + 10*a + b)
            self.assertEqual(first.rewards, rewards)
            self.assertEqual(first.observations, (Observation(a,b,rewards[0],False), Observation(b,a,rewards[1],False)))
            self.assertFalse(first.terminated)
            self.assertFalse(first.truncated)
            second = env.step(2,0)
            self.assertEqual(second.rewards, (1000*a + 100*b + 20, 10000 + 1000*a + 100*b + 20))
            self.assertFalse(second.terminated)
            self.assertTrue(second.truncated)
            with self.assertRaises(RuntimeError):
                env.step(0,0)

    def test_single_step_limit_and_reset(self):
        env = MemoryOneEnv(labelled_tensor(3), max_steps=1)
        with self.assertRaises(RuntimeError):
            env.step(0, 0)
        initial = env.reset(seed=9)
        self.assertTrue(env.step(0, 1).truncated)
        with self.assertRaises(RuntimeError):
            env.step(0, 1)
        self.assertEqual(env.reset(seed=9), initial)
        self.assertTrue(env.step(0, 1).truncated)

    def test_invalid_actions_do_not_change_state_or_time(self):
        n = 3
        for actions in ((-1,0), (0,n), (True,1), (1,1.0), (0,'1')):
            left = MemoryOneEnv(labelled_tensor(n), max_steps=1)
            right = MemoryOneEnv(labelled_tensor(n), max_steps=1)
            left.reset(seed=17)
            right.reset(seed=17)
            with self.subTest(actions=actions), self.assertRaises((TypeError, ValueError)):
                left.step(*actions)
            self.assertEqual(left.step(1,2), right.step(1,2))

    def test_local_rng_reproducible_including_unseeded_resets(self):
        saved = random.getstate()
        first = MemoryOneEnv(labelled_tensor(4), max_steps=3)
        second = MemoryOneEnv(labelled_tensor(4), max_steps=3)
        self.assertEqual(first.reset(seed=22), second.reset(seed=22))
        self.assertEqual([first.reset() for _ in range(15)], [second.reset() for _ in range(15)])
        self.assertEqual(random.getstate(), saved)
        for seed in (-1, True, 1.5, '4'):
            with self.subTest(seed=seed), self.assertRaises((TypeError,ValueError)):
                first.reset(seed=seed)

    def test_rule_change_at_reset_and_shape_rejection(self):
        env = MemoryOneEnv(labelled_tensor(3), max_steps=1)
        env.reset(seed=11)
        replacement = RewardTensor(3, [7] * (2*3**4))
        env.reset(seed=11, rule=replacement)
        self.assertEqual(env.step(1,2).rewards, (7,7))
        with self.assertRaises(ValueError):
            env.reset(seed=11, rule=labelled_tensor(4))
        for invalid in (0, -1, True, 1.5, '2'):
            with self.subTest(max_steps=invalid), self.assertRaises((ValueError, TypeError)):
                MemoryOneEnv(labelled_tensor(3), max_steps=invalid)


class CountingPolicy:
    """State is intentionally supplied by the runner, and all calls are observable."""
    def __init__(self, offset):
        self.offset = offset
        self.seen = []
        self.initial_calls = []

    def initial_state(self, *, n, seed=None):
        self.initial_calls.append((n, seed))
        return 0

    def act(self, observation, state, *, n):
        self.seen.append((observation, state, n))
        return PolicyOutput((state + self.offset) % n, state + 1)


class PolicyAndRunnerContract(unittest.TestCase):
    def test_shared_policy_object_gets_separate_per_player_state(self):
        env = MemoryOneEnv(labelled_tensor(3), max_steps=3)
        shared = CountingPolicy(1)
        result = run_episode(env, (shared, shared), seed=71, policy_seeds=(5, 6))
        self.assertEqual([state for _, state, _ in shared.seen], [0,0,1,1,2,2])
        self.assertEqual(shared.initial_calls, [(3,5), (3,6)])
        self.assertEqual([t.actions for t in result.transitions], [(1,1),(2,2),(0,0)])

    def test_policy_isolation_boundaries_and_joint_action_history(self):
        env = MemoryOneEnv(labelled_tensor(3), max_steps=3)
        p0, p1 = CountingPolicy(0), CountingPolicy(2)
        episode = run_episode(env, (p0,p1), seed=41, policy_seeds=(10,20))
        self.assertEqual(len(episode.transitions), 3)
        self.assertEqual(p0.initial_calls, [(3,10)])
        self.assertEqual(p1.initial_calls, [(3,20)])
        self.assertEqual([s for _,s,_ in p0.seen], [0,1,2])
        self.assertEqual([s for _,s,_ in p1.seen], [0,1,2])
        for i, transition in enumerate(episode.transitions):
            self.assertEqual(transition.actions, (i % 3,(i+2) % 3))
            self.assertEqual(transition.observations, (p0.seen[i][0],p1.seen[i][0]))
            self.assertEqual((p0.seen[i][2],p1.seen[i][2]), (3,3))
            if i:
                self.assertEqual(transition.observations, episode.transitions[i-1].next_observations)
            self.assertFalse(transition.terminated)
            self.assertEqual(transition.truncated, i == 2)
        self.assertEqual(episode.total_rewards, tuple(sum(t.rewards[p] for t in episode.transitions) for p in (0,1)))
        self.assertEqual(episode.transitions[0].observations[1].previous_opponent_action,
                         episode.transitions[0].observations[0].previous_own_action)
        run_episode(env, (p0,p1), seed=41, policy_seeds=(10,20))
        self.assertEqual([s for _,s,_ in p0.seen], [0,1,2,0,1,2])
        self.assertEqual([s for _,s,_ in p1.seen], [0,1,2,0,1,2])

    def test_random_policy_seeds_and_no_global_rng_use(self):
        saved = random.getstate()
        p = RandomPolicy()
        obs = Observation(0,1,0,True)
        one = p.initial_state(n=5, seed=73)
        two = p.initial_state(n=5, seed=73)
        seq1, seq2 = [], []
        for _ in range(25):
            out1 = p.act(obs, one, n=5)
            out2 = p.act(obs, two, n=5)
            seq1.append(out1.action)
            seq2.append(out2.action)
            one, two = out1.state, out2.state
        self.assertEqual(seq1, seq2)
        self.assertTrue(all(isinstance(a, int) and 0 <= a < 5 for a in seq1))
        self.assertEqual(random.getstate(), saved)
        for seed in (True, -1, 1.5, '2'):
            with self.subTest(seed=seed), self.assertRaises((ValueError,TypeError)):
                p.initial_state(n=3, seed=seed)

    def test_runner_requires_two_policies(self):
        env = MemoryOneEnv(labelled_tensor(3), max_steps=1)
        for policies in ((), (RandomPolicy(),), (RandomPolicy(),)*3):
            with self.subTest(policies=policies), self.assertRaises((ValueError,TypeError)):
                run_episode(env, policies)


if __name__ == '__main__':
    unittest.main()
