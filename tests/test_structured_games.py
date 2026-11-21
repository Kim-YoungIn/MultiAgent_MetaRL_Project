"""Independent contracts for the optional structured-game extension."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
import random
import subprocess
import sys
import unittest

from mamrl.adaptive_agents import OnlineQ
from mamrl.environment import MemoryOneEnv, Observation
from mamrl.policy import RandomPolicy
from mamrl.rollout import run_episode
from mamrl.rules import RuleSpec, RuleTerm, compile_rule
from mamrl.structured_games import (
    Predicate, RewardClause, StateVariable, StructuredEnv, StructuredGame,
    StructuredObservation, StructuredQ, TransitionClause,
    from_legacy, generate_catalog, generate_structured_rule,
)
from mamrl.tensor import RewardTensor


def zero(n=3):
    return RewardTensor(n, (0 for _ in range(2*n**4)))


class LegacyCompatibility(unittest.TestCase):
    def test_wrap_legacy_tensor_exact_rewards_and_observations(self):
        source = RuleSpec(3, (RuleTerm('coordination', 2), RuleTerm('copy_opponent', -1)))
        tensor = compile_rule(source)
        for original in (tensor, source):
            game = from_legacy(original)
            self.assertEqual(game.legacy_tensor(), tensor)
            standard = MemoryOneEnv(tensor, max_steps=9)
            extended = StructuredEnv(game, max_steps=9)
            for seed in (0, 41, 829):
                old = standard.reset(seed=seed)
                new = extended.reset(seed=seed)
                for a, b in zip(old, new):
                    self.assertEqual((a.previous_own_action, a.previous_opponent_action,
                                      a.previous_own_reward, a.episode_start_flag),
                                     (b.previous_own_action, b.previous_opponent_action,
                                      b.previous_own_reward, b.episode_start_flag))
                actions = [(i % 3, (2 * i + 1) % 3) for i in range(9)]
                for a,b in actions:
                    result_a, result_b = standard.step(a,b), extended.step(a,b)
                    self.assertEqual(result_a.rewards, result_b.rewards)
                    self.assertEqual(result_a.truncated, result_b.truncated)
                    for x, y in zip(result_a.observations, result_b.observations):
                        self.assertEqual((x.previous_own_action, x.previous_opponent_action,
                                          x.previous_own_reward, x.episode_start_flag),
                                         (y.previous_own_action, y.previous_opponent_action,
                                          y.previous_own_reward, y.episode_start_flag))

    def test_existing_runner_and_random_policy_work(self):
        env = StructuredEnv(generate_structured_rule(seed=29), max_steps=24)
        result = run_episode(env, (RandomPolicy(), RandomPolicy()),
                             seed=83, policy_seeds=(9,10))
        self.assertEqual(len(result.transitions), 24)
        self.assertTrue(result.transitions[-1].truncated)
        self.assertTrue(isinstance(result.transitions[0].observations[0], Observation))
        for p in (0,1):
            self.assertAlmostEqual(result.total_rewards[p],
                sum(t.rewards[p] for t in result.transitions), places=10)


    def test_meta_rollout_compatibility_with_legacy_rules_and_new_games(self):
        from mamrl.meta_exploration import RandomExplorer
        from mamrl.meta_exploration_rollout import run_meta_trial
        source = RuleSpec(3, (RuleTerm('coordination', 2),))
        old = run_meta_trial(source, RandomExplorer, seed=87, steps=5,
                             assessment_steps=7, opponent_profile='sarsa')
        new = run_meta_trial(from_legacy(source), RandomExplorer, seed=87, steps=5,
                             assessment_steps=7, opponent_profile='sarsa')
        for field in ('exploration_mean', 'assessment_mean', 'meta_reward',
                      'erased_mean', 'gameplay_mean'):
            self.assertEqual(old[field], new[field])
        game = generate_structured_rule(seed=15,memory_depth=3)
        extended = run_meta_trial(game, RandomExplorer, seed=87, steps=5,
                                  assessment_steps=7, opponent_profile='sarsa')
        self.assertEqual(len(extended['exploration']),5)
        self.assertEqual(len(extended['assessment']),7)
        self.assertIn('public_state', extended['exploration'][0]['observations'][0])
        self.assertEqual(extended['environment_steps_with_control'],19)

    def test_batched_ppo_teacher_path_accepts_structured_rules(self):
        try:
            from mamrl.action_set import ActionSetActorCritic
            from mamrl.meta_batch import collect_batch, replay_batch
        except ImportError:
            self.skipTest('PyTorch not installed')
        game=generate_structured_rule(seed=10,memory_depth=3)
        policy=ActionSetActorCritic(hidden_size=8,seed=11)
        batch=collect_batch(policy,[(game,121,'steady'),(game,122,'sarsa')],
                            steps=4,assessment_steps=6)
        self.assertEqual(len(batch),2)
        self.assertEqual(len(batch[0].actions),4)
        self.assertEqual(tuple(x.numel() for x in replay_batch(policy,batch)),(8,8,8))

    def test_original_online_q_and_extended_q_use_same_observation_contract(self):
        game = generate_structured_rule(seed=91)
        env = StructuredEnv(game, max_steps=5)
        obs = env.reset(seed=9)[0]
        legacy_student = OnlineQ(game.n, seed=1)
        aware_student = StructuredQ(game.n, seed=1)
        self.assertEqual(len(legacy_student._key(obs)), 4)
        self.assertEqual(len(aware_student._key(obs)), 6)
        out = env.step(0,1)
        for student in (legacy_student, aware_student):
            student.update(obs, 0, out.rewards[0], out.observations[0])
            self.assertEqual(student.updates, 1)
            self.assertEqual(student.snapshot()['entries'][0]['values'][0],
                             student.values(obs)[0])
        self.assertIn('public_state', aware_student.snapshot()['entries'][0]['observation'])
        self.assertEqual(aware_student.clone().snapshot(), aware_student.snapshot())


class StateAndTimeContracts(unittest.TestCase):
    def test_reward_and_transition_are_applied_from_old_state(self):
        game = StructuredGame(
            n=3, memory_depth=1, base=zero(),
            states=(StateVariable('resource',3),),
            rewards=(RewardClause(5, (Predicate('own_is',action=1),
                                     Predicate('state_min', state='resource',threshold=1))),),
            transitions=(TransitionClause('resource',1,(Predicate('own_is',action=0),),scope='both'),
                         TransitionClause('resource',-1,(Predicate('own_is',action=1),),scope='any')),
        )
        env = StructuredEnv(game,max_steps=4)
        env.reset(seed=2)
        self.assertEqual(env.step(0,0).rewards, (0,0))
        self.assertEqual(env.public_snapshot, (1,))
        self.assertEqual(env.step(1,2).rewards, (5,0))
        self.assertEqual(env.public_snapshot, (0,))
        self.assertEqual(env.step(1,1).rewards, (0,0))
        self.assertEqual(env.public_snapshot, (0,))
        self.assertFalse(env.step(0,2).terminated)
        with self.assertRaises(RuntimeError):
            env.step(0,0)

    def test_two_step_memory_changes_rewards_with_same_last_pair(self):
        game = StructuredGame(
            n=3, memory_depth=2, base=zero(),
            rewards=(RewardClause(7,(Predicate('own_count',action=1,window=2,count=2),)),),
        )
        # Two trajectories end with the same immediate previous pair but
        # different second-to-last pair. Reward on next action must differ.
        a, b = StructuredEnv(game,max_steps=4), StructuredEnv(game,max_steps=4)
        a.reset(seed=4); b.reset(seed=4)
        a.step(1,0); b.step(2,0)
        a.step(1,0); b.step(1,0)
        self.assertEqual(a.step(0,0).rewards[0], 7)
        self.assertEqual(b.step(0,0).rewards[0], 0)

    def test_hidden_state_remains_private_and_visible_state_is_public(self):
        game = StructuredGame(
            n=3, memory_depth=1, base=zero(),
            states=(StateVariable('visible',2), StateVariable('hidden',2,public=False)),
            transitions=(TransitionClause('hidden',1,(Predicate('own_is',action=1),),scope='any'),),
            rewards=(RewardClause(4,(Predicate('state_min',state='hidden',threshold=1),)),),
        )
        env = StructuredEnv(game,max_steps=4)
        first = env.reset(seed=1)
        self.assertEqual(first[0].public_state, (0, None))
        second = env.step(1,0)
        self.assertEqual(second.rewards, (0,0))
        self.assertEqual(second.observations[0].public_state, (0,None))
        self.assertEqual(env.step(2,0).rewards, (4,4))

    def test_player_exchange_symmetry(self):
        game = generate_structured_rule(n=4,seed=7,memory_depth=3,family='mixed')
        a,b = StructuredEnv(game,max_steps=10), StructuredEnv(game,max_steps=10)
        a.reset(seed=8); b.reset(seed=8)
        # Exchange all player-indexed information, not just current actions.
        b._history = tuple((y,x) for x,y in a._history)
        actions = ((1,2),(0,3),(2,2),(0,1),(3,2),(0,0))
        for x,y in actions:
            ra, rb = a.step(x,y), b.step(y,x)
            self.assertAlmostEqual(ra.rewards[0], rb.rewards[1])
            self.assertAlmostEqual(ra.rewards[1], rb.rewards[0])
            self.assertEqual(a._state,b._state)
            self.assertEqual(ra.observations[0].public_state,
                             rb.observations[1].public_state)

    def test_generated_games_change_real_state(self):
        game = generate_structured_rule(seed=2, family='mixed')
        env = StructuredEnv(game, max_steps=8)
        env.reset(seed=6)
        invest, harvest, _ = game.roles
        for _ in range(4):
            env.step(invest, invest)
        self.assertGreater(env._state['resource'], 0)
        self.assertGreater(env._state['progress'], 0)
        before = env._state['resource']
        env.step(harvest, harvest)
        self.assertLess(env._state['resource'], before)


    def test_stochastic_transitions_reproduce_under_fixed_seed(self):
        game=generate_structured_rule(seed=12,family='resource',stochastic=True)
        x,y=StructuredEnv(game,max_steps=24),StructuredEnv(game,max_steps=24)
        self.assertEqual(x.reset(seed=91),y.reset(seed=91))
        invest, harvest, _=game.roles
        sequence=[(invest,invest)]*15 + [(harvest,harvest)]*9
        for a,b in sequence:
            self.assertEqual(x.step(a,b),y.step(a,b))
        self.assertEqual(x.public_snapshot,y.public_snapshot)

    def test_atomic_bad_actions(self):
        game = generate_structured_rule(seed=4,stochastic=True)
        for invalid in (-1,True,3,'0'):
            left = StructuredEnv(game,max_steps=2)
            right = StructuredEnv(game,max_steps=2)
            left.reset(seed=17);right.reset(seed=17)
            with self.assertRaises((TypeError,ValueError)):
                left.step(0,invalid)
            self.assertEqual(left.step(0,1),right.step(0,1))


class GenerationContracts(unittest.TestCase):
    def test_families_are_deterministic_and_rules_have_dynamic_state(self):
        before = random.getstate()
        for family in ('resource','sequence','reciprocity','mixed'):
            for n in (3,4,5):
                a = generate_structured_rule(n=n,seed=17,memory_depth=3,family=family,
                                             stochastic=True)
                b = generate_structured_rule(n=n,seed=17,memory_depth=3,family=family,
                                             stochastic=True)
                self.assertEqual(a,b)
                self.assertTrue(a.transitions)
                self.assertTrue(a.rewards)
                self.assertEqual(a.fingerprint(),b.fingerprint())
                self.assertEqual(a,StructuredGame.from_dict(json.loads(json.dumps(a.to_dict()))))
                # Generator bounds all rewards in [-1, 1] for every reachable state.
                env=StructuredEnv(a,max_steps=12)
                env.reset(seed=3)
                for _ in range(12):
                    outcome=env.step(0,1)
                    self.assertLessEqual(max(map(abs,outcome.rewards)),1.000000001)
        self.assertEqual(before, random.getstate())

    def test_catalog_is_reproducible_and_exact_duplicates_excluded(self):
        a=generate_catalog(count=16,seed=18,n=3,memory_depth=3)
        b=generate_catalog(count=16,seed=18,n=3,memory_depth=3)
        self.assertEqual(a,b)
        self.assertEqual(len({x.fingerprint() for x in a}),16)
        self.assertEqual({x.family for x in a},{'resource','sequence','reciprocity','mixed'})

    def test_invalid_specs_rejected(self):
        with self.assertRaises(ValueError):
            StructuredGame(3,1,zero(),rewards=(RewardClause(1,(Predicate('repeat',lag=2),)),))
        with self.assertRaises(ValueError):
            StructuredGame(3,1,zero(),rewards=(RewardClause(1,(Predicate('state_min',state='x'),)),))
        with self.assertRaises(ValueError):
            StructuredGame(3,1,zero(),transitions=(TransitionClause('x',1),))
        with self.assertRaises(ValueError):
            StructuredGame(3,1,zero(),rewards=(RewardClause(1,(Predicate('own_is',action=3),)),))
        with self.assertRaises(ValueError):
            StructuredGame(3,1,zero(),rewards=(RewardClause(1,(Predicate('matching_count',window=2),)),))
        with self.assertRaises(ValueError):
            generate_structured_rule(n=2)
        with self.assertRaises(ValueError):
            generate_structured_rule(seed=0,state_levels=(1,))
        with self.assertRaises(ValueError):
            StructuredGame.from_dict({'version':99})

    def test_determinism_across_python_hash_seeds(self):
        script = ("from mamrl.structured_games import generate_structured_rule; "
                  "print(generate_structured_rule(seed=19,stochastic=True).fingerprint())")
        outputs=[]
        for hash_seed in ('4','888'):
            env=dict(os.environ,PYTHONHASHSEED=hash_seed)
            outputs.append(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
        self.assertEqual(outputs[0],outputs[1])

    def test_memory_one_adapter_rejects_invalid_downcast(self):
        with self.assertRaises(ValueError):
            generate_structured_rule(seed=0).legacy_tensor()


if __name__=='__main__':
    unittest.main()
