"""Exercise the real generator/compiler/environment/policy chain end-to-end."""
import unittest

from mamrl.environment import MemoryOneEnv
from mamrl.policy import RandomPolicy
from mamrl.rollout import run_episode
from mamrl.rules import compile_rule, generate_rule


class GeneratedGameIntegration(unittest.TestCase):
    def test_real_generated_games_random_policies_and_reproducible_transitions(self):
        for n in (3, 4, 5):
            for family in ('zero_sum', 'coordination', 'memory_one', 'mixed'):
                with self.subTest(n=n, family=family):
                    spec = generate_rule(n=n, seed=37, family=family)
                    tensor = compile_rule(spec)
                    env = MemoryOneEnv(tensor, max_steps=7)
                    episode = run_episode(env, (RandomPolicy(), RandomPolicy()),
                                          seed=53, policy_seeds=(71, 89))
                    repeated_tensor = compile_rule(generate_rule(n=n, seed=37, family=family))
                    repeated = run_episode(MemoryOneEnv(repeated_tensor, max_steps=7),
                                           (RandomPolicy(), RandomPolicy()),
                                           seed=53, policy_seeds=(71, 89))
                    self.assertEqual(tensor, repeated_tensor)
                    self.assertEqual(episode, repeated)
                    self.assertEqual(len(episode.transitions), 7)
                    totals = [0, 0]
                    for step, transition in enumerate(episode.transitions):
                        o0, o1 = transition.observations
                        a0, a1 = transition.actions
                        p0, p1 = o0.previous_own_action, o1.previous_own_action
                        self.assertEqual(o0.previous_opponent_action, p1)
                        self.assertEqual(o1.previous_opponent_action, p0)
                        self.assertEqual((o0.episode_start_flag, o1.episode_start_flag),
                                         (step == 0, step == 0))
                        self.assertTrue(0 <= a0 < n and 0 <= a1 < n)
                        expected = (tensor[0, p0, p1, a0, a1], tensor[1, p0, p1, a0, a1])
                        self.assertEqual(transition.rewards, expected)
                        next0, next1 = transition.next_observations
                        self.assertEqual((next0.previous_own_action, next0.previous_opponent_action,
                                          next0.previous_own_reward, next0.episode_start_flag),
                                         (a0, a1, expected[0], False))
                        self.assertEqual((next1.previous_own_action, next1.previous_opponent_action,
                                          next1.previous_own_reward, next1.episode_start_flag),
                                         (a1, a0, expected[1], False))
                        self.assertFalse(transition.terminated)
                        self.assertEqual(transition.truncated, step == 6)
                        for player in (0, 1):
                            totals[player] += expected[player]
                        if step:
                            self.assertEqual(transition.observations,
                                             episode.transitions[step - 1].next_observations)
                    self.assertEqual(episode.total_rewards, tuple(totals))


if __name__ == '__main__':
    unittest.main()
