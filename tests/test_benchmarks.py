"""Independent truth tables, reachable memory witness and opponent behavior."""
from itertools import product
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

from mamrl.environment import MemoryOneEnv
from mamrl.policy import RandomPolicy
from mamrl.rules import RuleSpec, RuleTerm, compile_rule, generate_rule


def seeded_initial_pair(n, pair):
    for seed in range(10000):
        rng = random.Random(seed)
        if (rng.randrange(n), rng.randrange(n)) == pair:
            return seed
    raise AssertionError('failed to find a reset seed')


class BenchmarkRulesContract(unittest.TestCase):
    def test_probe_mode_payoffs_for_all_states_actions_players(self):
        for n in (3,4,5):
            for mode, probe_reward in (('probe_repeat',-.25),('probe_copy',-.5)):
                tensor = compile_rule(RuleSpec(n,(RuleTerm(mode,1),)))
                scaled = compile_rule(RuleSpec(n,(RuleTerm(mode,2),)))
                for p0,p1,a0,a1 in product(range(n),repeat=4):
                    for player in (0,1):
                        previous = (p0,p1)
                        actions = (a0,a1)
                        ref = previous[player] if mode == 'probe_repeat' else previous[1-player]
                        other = previous[1-player] if mode == 'probe_repeat' else previous[player]
                        a = actions[player]
                        expected = 1 if a == ref else (-3 if a == other else probe_reward)
                        self.assertEqual(tensor[player,p0,p1,a0,a1], expected)
                        self.assertEqual(scaled[player,p0,p1,a0,a1], 2*expected)

    def test_reachable_identical_observation_has_different_optimal_actions(self):
        seed = seeded_initial_pair(3,(0,1))
        histories = {}
        for kind in ('probe_repeat','probe_copy'):
            env = MemoryOneEnv(compile_rule(RuleSpec(3,(RuleTerm(kind,1),))),max_steps=4)
            start = env.reset(seed=seed)
            self.assertEqual((start[0].previous_own_action,start[0].previous_opponent_action),(0,1))
            transitions = []
            for actions in ((2,0),(0,0),(0,1)):
                step = env.step(*actions)
                transitions.append(step)
            histories[kind] = transitions
            final = transitions[-1].observations[0]
            self.assertEqual((final.previous_own_action,final.previous_opponent_action,
                              final.previous_own_reward,final.episode_start_flag), (0,1,1.,False))
        self.assertEqual(histories['probe_repeat'][0].rewards[0],-.25)
        self.assertEqual(histories['probe_copy'][0].rewards[0],-.5)
        self.assertEqual(histories['probe_repeat'][-1].observations[0],histories['probe_copy'][-1].observations[0])
        for kind, correct, other in (('probe_repeat',0,1),('probe_copy',1,0)):
            tensor = compile_rule(RuleSpec(3,(RuleTerm(kind,1),)))
            self.assertEqual(tensor[0,0,1,correct,0],1)
            self.assertEqual(tensor[0,0,1,other,0],-3)

    def test_generator_is_unchanged_by_explicit_probe_modes(self):
        # New benchmark-only terms must not enter mixed generator primitive sampling.
        allowed = {'coordination','repeat_own','copy_opponent','action_bonus','cyclic_zero_sum'}
        for seed in range(30):
            spec = generate_rule(3,seed=seed,family='mixed')
            self.assertTrue(all(term.kind in allowed for term in spec.terms))


class BenchmarkOpponentsContract(unittest.TestCase):
    def test_best_response_values_for_copy_and_avoid(self):
        from mamrl.opponents import make_opponent
        from mamrl.environment import Observation
        class EnumeratingRandom(random.Random):
            def __init__(self):
                super().__init__(0)
                self.i = 0
            def randrange(self, stop):
                value = self.i % stop
                self.i += 1
                return value
        for n in (3,4,5):
            for previous_learner in range(n):
                obs = Observation(0,previous_learner,0.,False)
                copy_action = make_opponent('copy_previous').act(obs,random.Random(1),n=n).action
                self.assertEqual(float(copy_action == previous_learner), 1.)
                for learner_action in range(n):
                    opponent = make_opponent('avoid_previous')
                    rng = EnumeratingRandom()
                    hits = sum(opponent.act(obs,rng,n=n).action == learner_action for _ in range(n-1))
                    expected = 0. if learner_action == previous_learner else 1/(n-1)
                    self.assertAlmostEqual(hits/(n-1),expected)

    def test_copy_and_avoid_previous_and_local_rng(self):
        from mamrl.opponents import make_opponent
        from mamrl.environment import Observation
        before = random.getstate()
        for n in (3,4,5):
            for previous_learner in range(n):
                opponent_obs = Observation(0,previous_learner,0.,False)
                copy = make_opponent('copy_previous')
                avoid = make_opponent('avoid_previous')
                copy_state = copy.initial_state(n=n,seed=23)
                avoid_a = avoid.initial_state(n=n,seed=23)
                avoid_b = avoid.initial_state(n=n,seed=23)
                copy_out = copy.act(opponent_obs,copy_state,n=n)
                self.assertEqual(copy_out.action, previous_learner)
                seq_a,seq_b = [],[]
                for _ in range(60):
                    a = avoid.act(opponent_obs,avoid_a,n=n)
                    b = avoid.act(opponent_obs,avoid_b,n=n)
                    avoid_a,avoid_b = a.state,b.state
                    seq_a.append(a.action); seq_b.append(b.action)
                self.assertEqual(seq_a,seq_b)
                self.assertTrue(all(action in range(n) and action != previous_learner for action in seq_a))
                self.assertEqual(set(seq_a), set(range(n))-{previous_learner})
        self.assertEqual(random.getstate(),before)
        with self.assertRaises((TypeError,ValueError)):
            make_opponent('unknown')


class BenchmarkReferenceContract(unittest.TestCase):
    def test_balanced_task_strata_and_official_memory_witness(self):
        from mamrl.benchmarks import benchmark_tasks, probe_memory_witness
        hidden = benchmark_tasks('hidden_action',n=3,count=6)
        self.assertEqual([t.rule.terms[0].action for t in hidden], [0,1,2,0,1,2])
        self.assertEqual([t.opponent_kind for t in hidden], ['random']*6)
        probe = benchmark_tasks('probe_memory',n=3,count=6)
        self.assertEqual([t.rule.terms[0].kind for t in probe], ['probe_repeat','probe_copy']*3)
        opponents = benchmark_tasks('opponent_response',n=3,count=6)
        self.assertEqual([t.opponent_kind for t in opponents], ['copy_previous','avoid_previous']*3)
        for family in ('hidden_action','probe_memory','opponent_response'):
            with self.subTest(family=family), self.assertRaises(ValueError):
                benchmark_tasks(family,n=3,count=5)
        witness = probe_memory_witness()
        self.assertEqual(len(witness),2)
        finals = []
        for item in witness:
            env = MemoryOneEnv(compile_rule(RuleSpec(3,(RuleTerm('probe_'+item['mode'],1),))),max_steps=3)
            obs = env.reset(seed=item['reset_seed'])
            self.assertEqual([obs[0].previous_own_action,obs[1].previous_own_action],item['initial_state'])
            observations,rewards = [asdict(obs[0])],[]
            for actions in item['prefix_actions']:
                step = env.step(*actions)
                observations.append(asdict(step.observations[0])); rewards.append(step.rewards[0])
            self.assertEqual(observations,item['observations'])
            self.assertEqual(rewards,item['rewards'])
            self.assertEqual(observations[-1],item['final_observation'])
            self.assertEqual(item['optimal_actions'],[0] if item['mode']=='repeat' else [1])
            finals.append(observations[-1])
        self.assertEqual(finals[0],finals[1])
        self.assertEqual(finals[0],{'previous_own_action':0,'previous_opponent_action':1,
                                    'previous_own_reward':1.,'episode_start_flag':False})
        self.assertNotEqual(witness[0]['rewards'][0],witness[1]['rewards'][0])

    def test_benchmark_runner_leaks_only_family_n_seed_and_bootstraps_reset_observation(self):
        from mamrl.benchmarks import benchmark_tasks, run_benchmark_trial
        task = benchmark_tasks('hidden_action',n=3,count=3)[1]
        calls = []
        class Spy:
            def act(self,obs):
                calls.append(('act',obs)); return 0
            def observe(self,obs,action,reward,final):
                calls.append(('observe',obs,action,reward,final))
            def transition(self,obs,action,reward,next_obs,trial_end):
                calls.append(('transition',obs,action,reward,next_obs,trial_end))
            def episode_boundary(self,final):
                calls.append(('boundary',final))
            def reset_memory(self):
                calls.append(('reset',))
        factory_args = []
        def factory(*,family,n,seed):
            factory_args.append({'family':family,'n':n,'seed':seed})
            return Spy()
        trial = run_benchmark_trial(task,factory,seed=23,episodes_per_trial=2,max_steps=2)
        self.assertEqual(factory_args,[{'family':'hidden_action','n':3,'seed':trial.learner_seed}])
        self.assertEqual([x[0] for x in calls],
                         ['act','observe','transition','act','observe','transition','boundary',
                          'act','observe','transition','act','observe','transition'])
        last_first_ep = [c for c in calls if c[0]=='transition'][1]
        first_second_obs = trial.episodes[1].transitions[0].observations[0]
        self.assertEqual(last_first_ep[4],first_second_obs)
        self.assertFalse(last_first_ep[5])
        self.assertTrue([c for c in calls if c[0]=='transition'][-1][5])
        boundary = [c for c in calls if c[0]=='boundary'][0]
        self.assertEqual(boundary[1],trial.episodes[0].transitions[-1].next_observations[0])
        reset_calls = []
        def reset_factory(*,family,n,seed):
            instance = Spy()
            original = instance.reset_memory
            def reset():
                reset_calls.append(True); original()
            instance.reset_memory = reset
            return instance
        run_benchmark_trial(task,reset_factory,seed=23,episodes_per_trial=2,max_steps=2,
                            reset_memory_between_episodes=True)
        self.assertEqual(reset_calls,[True])

    def test_q_bootstrap_and_belief_use_only_observable_feedback(self):
        from mamrl.benchmarks import QLearner, BeliefLearner
        from mamrl.environment import Observation
        q = QLearner(family='hidden_action',n=3,seed=7)
        old = Observation(0,1,0.,False)
        reset_obs = Observation(2,1,0.,True)
        last_obs = Observation(1,1,1.,False)
        q.values(reset_obs)[1] = 2.
        q.values(last_obs)[2] = 100.
        q.transition(old,0,1.,reset_obs,False)
        self.assertAlmostEqual(q.values(old)[0],.5*(1+.99*2))
        q.transition(old,1,1.,last_obs,True)
        self.assertAlmostEqual(q.values(old)[1],.5)
        self.assertIn((0,1,0.,False),q.q)
        belief = BeliefLearner(family='hidden_action',n=3,seed=7)
        belief.observe(old,0,0.,last_obs)
        self.assertNotIn(0,belief.possible)
        for _ in range(10):
            self.assertIn(belief.act(old),(1,2))
        belief.observe(old,1,1.,last_obs)
        self.assertEqual(belief.possible,{1})
        self.assertEqual(belief.act(old),1)

        # Probe feedback identifies the mode; later both histories can have
        # the exact same current observation but require different actions.
        common = Observation(0,1,1.,False)
        probe_obs = Observation(0,1,0.,True)
        for reward,expected_mode,action in ((-.25,'repeat',0),(-.5,'copy',1)):
            learner = BeliefLearner(family='probe_memory',n=3,seed=7)
            learner.observe(probe_obs,2,reward,Observation(2,0,reward,False))
            self.assertEqual(learner.possible,{expected_mode})
            self.assertEqual(learner.act(common),action)

        for opponent_action,expected_kind in ((0,'copy_previous'),(1,'avoid_previous')):
            learner = BeliefLearner(family='opponent_response',n=3,seed=7)
            previous = Observation(0,2,0.,False)
            learner.observe(previous,1,float(1==opponent_action),
                            Observation(1,opponent_action,float(1==opponent_action),False))
            self.assertEqual(learner.possible,{expected_kind})

    def test_paired_benchmark_provenance_and_family_separation(self):
        from mamrl.benchmarks import (benchmark_tasks, evaluate_benchmark,
                                      paired_benchmark_difference, RandomLearner)
        tasks = benchmark_tasks('opponent_response',n=3,count=4)
        baseline = evaluate_benchmark(RandomLearner,tasks,seed=7,episodes_per_trial=2,max_steps=2)
        self.assertEqual(set(baseline['by_task']),{'copy_previous','avoid_previous'})
        self.assertEqual(baseline['by_task']['copy_previous']['best_response_bound'],1.)
        self.assertEqual(baseline['by_task']['avoid_previous']['best_response_bound'],.5)
        self.assertEqual(paired_benchmark_difference(baseline,baseline)['trial_differences'],[0.]*4)
        left,right = deepcopy(baseline),deepcopy(baseline)
        left['trial_returns'],right['trial_returns'] = [2,4,6,8],[0,0,0,0]
        paired = paired_benchmark_difference(left,right)
        self.assertEqual(paired['trial_differences'],[.5,1.,1.5,2.])
        self.assertAlmostEqual(paired['mean'],1.25)
        self.assertAlmostEqual(paired['standard_error'],(1/8)**.5)
        different_order = evaluate_benchmark(RandomLearner,tuple(reversed(tasks)),seed=7,episodes_per_trial=2,max_steps=2)
        with self.assertRaises(ValueError):
            paired_benchmark_difference(baseline,different_order)
        wrong_boundaries = evaluate_benchmark(RandomLearner,tasks,seed=7,episodes_per_trial=1,max_steps=4)
        with self.assertRaises(ValueError):
            paired_benchmark_difference(baseline,wrong_boundaries)
        both_bad = deepcopy(baseline)
        both_bad['task_fingerprints'] = both_bad['task_fingerprints'][:2]
        both_bad['task_labels'] = both_bad['task_labels'][:2]
        with self.assertRaises(ValueError):
            paired_benchmark_difference(both_bad,both_bad)
        both_bad = deepcopy(baseline)
        both_bad['steps_per_trial'] = 3
        with self.assertRaises(ValueError):
            paired_benchmark_difference(both_bad,both_bad)
        different_family = benchmark_tasks('probe_memory',n=3,count=2)
        with self.assertRaises(ValueError):
            evaluate_benchmark(RandomLearner,(tasks[0],different_family[0]),seed=7,episodes_per_trial=2,max_steps=2)


@unittest.skipIf(torch is None, 'optional training dependency torch is not installed')
class BenchmarkCollectorContract(unittest.TestCase):
    def test_training_cli_benchmark_family_defaults_and_metadata(self):
        from mamrl.recurrent import load_checkpoint
        from mamrl.recording import load_record
        from mamrl.training import TrainConfig,evaluate
        self.assertEqual(TrainConfig().architecture,'ordinal_v1')
        with TemporaryDirectory() as tmp:
            for family in ('probe_memory','opponent_response'):
                output = Path(tmp)/family
                cmd = [sys.executable,'-m','mamrl.train','--output-dir',str(output),
                       '--family',family,'--hidden-size','4','--updates','1',
                       '--trials-per-update','1','--episodes-per-trial','2','--max-steps','2',
                       '--epochs','1','--evaluation-trials','2','--evaluation-interval','1','--seed','13']
                result = subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy())
                self.assertEqual(result.returncode,0,result.stderr)
                record = json.loads((output/'training.json').read_text(encoding='utf-8'))
                config = record['config']
                self.assertEqual(config['architecture'],'action_set_v1')
                expected_mode = 'mixed' if family=='opponent_response' else 'random'
                self.assertEqual(config['opponent_mode'],expected_mode)
                expected_kinds = ['copy_previous','avoid_previous'] if family=='opponent_response' else ['random']*2
                self.assertEqual(record['evaluation_opponent_kinds'],expected_kinds)
                specs = [RuleSpec(s['n'], tuple(RuleTerm(**t) for t in s['terms']))
                         for s in record['evaluation_task_specs']]
                allowed = {'coordination'} if family=='opponent_response' else {'probe_repeat','probe_copy'}
                self.assertTrue(all(spec.terms[0].kind in allowed for spec in specs))
                replay = evaluate(load_checkpoint(output/'policy.pt'),specs,episodes_per_trial=2,max_steps=2,
                                  seed=record['evaluation_seed'],opponent_kinds=expected_kinds)
                self.assertEqual(replay,record['comparisons']['trained'])
                sample = load_record(output/'sample_episode/run.json')
                self.assertEqual(sample['metadata']['policies'][0],'RecurrentPolicy')
                self.assertEqual(sample['metadata']['policies'][1],
                                 'CopyPreviousPolicy' if family=='opponent_response' else 'RandomPolicy')

    def test_neural_benchmark_runner_matches_training_collector(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.benchmarks import benchmark_tasks, run_benchmark_trial, NeuralLearner
        from mamrl.training import collect_trial
        tasks = (benchmark_tasks('hidden_action',n=3,count=3)[1],
                 benchmark_tasks('probe_memory',n=3,count=2)[0],
                 *benchmark_tasks('opponent_response',n=3,count=2))
        for memoryless in (False,True):
            model = ActionSetActorCritic(hidden_size=4,seed=11,memoryless=memoryless)
            def factory(*,family,n,seed):
                return NeuralLearner(model=model,family=family,n=n,seed=seed)
            for task in tasks:
                for reset in (False,True):
                    with self.subTest(memoryless=memoryless,family=task.family,
                                      opponent=task.opponent_kind,reset=reset):
                        direct = collect_trial(model,task.rule,episodes_per_trial=2,max_steps=3,seed=129,
                                               opponent_kind=task.opponent_kind,reset_hidden=reset)
                        benchmark = run_benchmark_trial(task,factory,episodes_per_trial=2,max_steps=3,seed=129,
                                                        reset_memory_between_episodes=reset)
                        self.assertEqual(direct.environment_seeds,benchmark.environment_seeds)
                        self.assertEqual(direct.opponent_seeds,benchmark.opponent_seeds)
                        self.assertEqual(direct.learner_seeds[0],benchmark.learner_seed)
                        self.assertEqual(direct.episodes,benchmark.episodes)

    def test_benchmark_cli_artifacts_and_restored_weights(self):
        from mamrl.benchmarks import (BenchmarkTask,NeuralLearner,benchmark_tasks,evaluate_benchmark)
        from mamrl.recurrent import load_checkpoint
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'benchmark'
            cmd = [sys.executable,'-m','mamrl.benchmark_run','--output-dir',str(output),
                   '--seeds','11','--updates','1','--evaluation-trials','6',
                   '--trials-per-update','1','--max-steps','2','--hidden-size','4']
            run = subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy())
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)['runs'],6)
            record_path,report_path = output/'benchmark.json', output/'report.html'
            record = json.loads(record_path.read_text(encoding='utf-8'))
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(record['runs']),6)
            families = ('hidden_action','probe_memory','opponent_response')
            architectures = ('action_set_v1','action_set_ff_v1')
            self.assertEqual({(r['family'],r['architecture']) for r in record['runs']},
                             {(f,a) for f in families for a in architectures})
            for family in families:
                self.assertEqual(set(record['references'][family]),
                                 {'random','reactive','q_learning','informed','informed_reset'})
            self.assertEqual(record['task_checks']['bounds']['opponent_copy'],1.)
            self.assertEqual(record['task_checks']['bounds']['opponent_avoid'],.5)
            self.assertEqual(record['protocol']['seeds'],[11])
            self.assertEqual(record['protocol']['evaluation_trials'],6)
            for name,digest in record['protocol']['source_sha256'].items():
                self.assertEqual(hashlib.sha256((output/'sources'/name).read_bytes()).hexdigest(),digest)
            for row in record['runs']:
                with self.subTest(family=row['family'],architecture=row['architecture']):
                    self.assertEqual(row['seed'],11)
                    self.assertEqual(row['hidden_state_elements'],12)
                    self.assertEqual(row['config']['architecture'],row['architecture'])
                    checkpoint = output/row['checkpoint']
                    self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),row['checkpoint_sha256'])
                    self.assertTrue((output/row['initial_checkpoint']).is_file())
                    self.assertEqual(json.loads((output/row['raw_record']).read_text(encoding='utf-8')),row)
                    model = load_checkpoint(checkpoint)
                    def factory(*,family,n,seed):
                        return NeuralLearner(model=model,family=family,n=n,seed=seed)
                    tasks = benchmark_tasks(row['family'],n=3,count=6)
                    self.assertEqual(json.loads(json.dumps([asdict(t) for t in tasks])),row['evaluation_tasks'])
                    replay = evaluate_benchmark(factory,tasks,seed=row['evaluation_seed'],
                                                episodes_per_trial=2,max_steps=2)
                    self.assertEqual(replay,row['comparisons']['trained'])
                    if row['architecture']=='action_set_ff_v1':
                        self.assertEqual(row['memory_benefit']['mean'],0.)
                        self.assertEqual(row['memory_benefit']['trial_differences'],[0.]*6)
            report = report_path.read_text(encoding='utf-8')
            self.assertNotRegex(report.lower(),r'<\s*script\b|@import\b|https?://')
            self.assertIn('probe_memory',report)
            self.assertIn('0.5',report)
            first = record_path.read_bytes()
            retry = subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy())
            self.assertNotEqual(retry.returncode,0)
            self.assertEqual(record_path.read_bytes(),first)

    def test_mixed_opponents_do_not_consume_task_rng_stream(self):
        from mamrl.training import TrainConfig, evaluation_opponents, sample_task, train
        config = TrainConfig(n=3, family='opponent_response', architecture='action_set_v1',
                             hidden_size=4, updates=2, trials_per_update=2, episodes_per_trial=2,
                             max_steps=2, epochs=1, evaluation_trials=4, evaluation_interval=2, seed=29)
        seed_traces = []
        for mode in ('random','mixed'):
            seen = []
            def tracked_sample(n, family, seed, **kwargs):
                seen.append(seed)
                return sample_task(n, family, seed, **kwargs)
            with mock.patch('mamrl.training.sample_task',side_effect=tracked_sample):
                train(replace(config,opponent_mode=mode))
            seed_traces.append(seen)
        self.assertEqual(seed_traces[0],seed_traces[1])
        self.assertIsNone(evaluation_opponents(config))
        self.assertEqual(evaluation_opponents(replace(config,opponent_mode='mixed')),
                         ('copy_previous','avoid_previous','copy_previous','avoid_previous'))

    def test_opponent_conditions_and_matched_exogenous_streams(self):
        from mamrl.recurrent import RecurrentActorCritic
        from mamrl.training import collect_trial, evaluate
        spec = RuleSpec(3,(RuleTerm('coordination',1),))
        model = RecurrentActorCritic(hidden_size=4,seed=5)
        for kind in ('copy_previous','avoid_previous','random'):
            learned = collect_trial(model,spec,episodes_per_trial=2,max_steps=4,seed=100,opponent_kind=kind)
            random_trial = collect_trial(None,spec,episodes_per_trial=2,max_steps=4,seed=100,opponent_kind=kind)
            self.assertEqual(learned.opponent_kind,kind)
            self.assertEqual(learned.environment_seeds, random_trial.environment_seeds)
            self.assertEqual(learned.opponent_seeds, random_trial.opponent_seeds)
            for episode in learned.episodes:
                for transition in episode.transitions:
                    a0,a1 = transition.actions
                    prev0 = transition.observations[0].previous_own_action
                    if kind == 'copy_previous':
                        self.assertEqual(a1,prev0)
                    if kind == 'avoid_previous':
                        self.assertNotEqual(a1,prev0)
                    self.assertEqual(transition.rewards, (float(a0==a1),float(a0==a1)))
                    self.assertEqual(set(vars(transition.observations[0])) if hasattr(transition.observations[0],'__dict__') else
                                     set(transition.observations[0].__slots__),
                                     {'previous_own_action','previous_opponent_action','previous_own_reward','episode_start_flag'})
        with self.assertRaises((TypeError,ValueError)):
            evaluate(model,[spec,spec],episodes_per_trial=2,max_steps=2,seed=3,opponent_kinds=['random'])


if __name__ == '__main__':
    unittest.main()
