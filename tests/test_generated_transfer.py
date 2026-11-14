"""Independent checks of controlled opponents and generated-task training interfaces."""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mamrl.environment import Observation
from mamrl.opponents import make_opponent
from mamrl.rules import RuleSpec, RuleTerm, compile_rule

try:
    import torch
except ImportError:
    torch = None


class WinStayOpponentContract(unittest.TestCase):
    def test_start_is_uniform_seeded_then_win_stay_lose_switch(self):
        p = make_opponent('win_stay')
        for n in (3,4,5):
            for seed in range(60):
                state = p.initial_state(n=n,seed=seed)
                start = Observation(1,2,0,True)
                action = p.act(start,state,n=n).action
                self.assertEqual(action, random.Random(seed).randrange(n))
                for reward in (0,-1,-.5):
                    out = p.act(Observation(1,2,reward,False), state,n=n)
                    self.assertNotEqual(out.action,1)
                    self.assertIn(out.action,range(n))
                for reward in (.1,1,2):
                    out = p.act(Observation(1,2,reward,False), state,n=n)
                    self.assertEqual(out.action,1)

    def test_lose_switch_is_uniform_over_alternatives(self):
        # Random(seed) first randrange(n-1) is the documented random draw.
        p = make_opponent('win_stay')
        for own in range(3):
            observed = []
            for seed in range(100):
                state = p.initial_state(n=3,seed=seed)
                got = p.act(Observation(own,0,-1,False),state,n=3).action
                raw = random.Random(seed).randrange(2)
                expected = raw + (raw >= own)
                self.assertEqual(got,expected)
                observed.append(got)
            self.assertEqual(set(observed),set(range(3))-{own})

    def test_global_rng_is_untouched_and_reset_same_seed(self):
        p = make_opponent('win_stay')
        old = random.getstate()
        obs = Observation(1,2,0,True)
        seq=[]
        for _ in range(2):
            state=p.initial_state(n=3,seed=99)
            seq.append([p.act(obs,state,n=3).action for _ in range(6)])
        self.assertEqual(*seq)
        self.assertEqual(random.getstate(),old)


@unittest.skipUnless(torch is not None, 'optional torch training dependency')
class GeneratedTrainingContract(unittest.TestCase):
    def test_collector_never_passes_task_or_horizon_and_actions_are_simultaneous(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.training import collect_trial

        class InspectingActor(ActionSetActorCritic):
            def __init__(self):
                super().__init__(hidden_size=4,seed=4)
                self.seen=[]
            def forward_observation(self,observation,hidden,*,n):
                self.seen.append(observation)
                self_fields={f.name for f in dataclasses.fields(observation)}
                if self_fields != {'previous_own_action','previous_opponent_action',
                                   'previous_own_reward','episode_start_flag'}:
                    raise AssertionError('observation leaked extra data')
                return super().forward_observation(observation,hidden,n=n)

        spec=RuleSpec(3,(RuleTerm('coordination',1),))
        model=InspectingActor()
        a=collect_trial(model,compile_rule(spec),episodes_per_trial=2,max_steps=5,
                        seed=712,opponent_kind='copy_previous')
        b=collect_trial(None,compile_rule(spec),episodes_per_trial=2,max_steps=5,
                        seed=712,opponent_kind='copy_previous')
        self.assertEqual(a.environment_seeds,b.environment_seeds)
        self.assertEqual(a.opponent_seeds,b.opponent_seeds)
        self.assertEqual(len(a.episodes),2)
        self.assertEqual(sum(o.episode_start_flag for o in a.events),2)
        for episode in a.episodes:
            self.assertTrue(episode.transitions[0].observations[0].episode_start_flag)
            self.assertEqual(episode.transitions[0].observations[0].previous_own_reward,0)
            for step in episode.transitions:
                # The opponent copies the learner's previous action, not this simultaneous action.
                self.assertEqual(step.actions[1], step.observations[1].previous_opponent_action)
                self.assertEqual(step.next_observations[0].previous_own_action,step.actions[0])
                self.assertEqual(step.next_observations[0].previous_opponent_action,step.actions[1])
                self.assertEqual(step.next_observations[0].previous_own_reward,step.rewards[0])
            self.assertTrue(episode.transitions[-1].truncated)

    def test_custom_training_sampler_and_evaluation_are_distinct_and_replay_likelihood(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.training import TrainConfig, train, collect_trial, replay_trial
        rule1 = compile_rule(RuleSpec(3,(RuleTerm('coordination',1),)))
        rule2 = compile_rule(RuleSpec(3,(RuleTerm('repeat_own',1),)))
        seen=[]
        def sampler(seed):
            seen.append(seed)
            return rule1,'copy_previous'
        config=TrainConfig(n=3,architecture='action_set_v1',hidden_size=4,updates=1,
                           trials_per_update=2,episodes_per_trial=2,max_steps=2,
                           epochs=1,evaluation_trials=2,evaluation_interval=1)
        result=train(config,task_sampler=sampler,evaluation_tasks=((rule2,'win_stay'),))
        self.assertEqual(len(seen),2)
        self.assertEqual(len(result.evaluations),2)
        self.assertEqual(result.evaluations[0]['trials'],1)
        self.assertEqual(result.evaluations[-1]['trials'],1)
        trial=collect_trial(result.model,rule2,episodes_per_trial=2,max_steps=3,
                            seed=3,opponent_kind='win_stay')
        logs,values,entropy=replay_trial(result.model,trial)
        self.assertTrue(torch.allclose(logs,torch.tensor(trial.old_log_probs),atol=1e-6,rtol=1e-6))
        self.assertEqual(len(entropy),6)
        with self.assertRaises(ValueError):
            train(config,task_sampler=sampler,evaluation_tasks=())
        with self.assertRaises((TypeError,ValueError)):
            train(config,task_sampler=sampler,evaluation_tasks=((rule2,'unknown'),))


class GeneratedManifestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mamrl.generated_tasks import build_manifest
        cls.before = random.getstate()
        cls.manifest = build_manifest(n=3,seed=20261114)
        cls.after = random.getstate()

    def test_seed_reproducibility_and_strata_counts(self):
        from mamrl.generated_tasks import build_manifest, tasks_from_manifest, CONDITIONS, SPLITS
        m = self.manifest
        self.assertEqual(self.before,self.after)
        self.assertEqual(build_manifest(n=3,seed=20261114),m)
        self.assertEqual(len(m['rules']),24)
        self.assertEqual([sum(r['split']==s for r in m['rules']) for s in ('train','fresh_rules','structural')], [12,6,6])
        self.assertEqual([len(tasks_from_manifest(m,split=s)) for s in ('fresh_rules','structural','novel_opponent')], [18,18,12])
        self.assertGreater(len(tasks_from_manifest(m,split='combination')),0)
        for condition in CONDITIONS:
            selected=tasks_from_manifest(m,condition=condition)
            self.assertTrue(selected)
            self.assertTrue(all(t.split=='train' for t in selected))
            expected_rule_count=12 if condition.startswith('high_rules') else 4
            self.assertEqual(len({t.rule_id for t in selected}),expected_rule_count)
            opponents={t.opponent_kind for t in selected}
            if condition.endswith('low_opponents'):
                self.assertEqual(opponents,{'copy_previous'})
            else:
                self.assertEqual(opponents,{'random','copy_previous','avoid_previous'})

    def test_tensor_normalization_and_global_permutation_dedup(self):
        import itertools
        from mamrl.logical_rules import rule_from_dict
        from mamrl.generated_tasks import common_values
        records=self.manifest['rules']
        table=[]
        for entry in records:
            rule=rule_from_dict(entry['rule'])
            vals=common_values(rule)
            self.assertAlmostEqual(min(vals),-1.)
            self.assertAlmostEqual(max(vals),1.)
            src=common_values(rule_from_dict(entry['source_rule']))
            factor=entry['normalization']['factor']; shift=entry['normalization']['shift']
            self.assertGreater(factor,0.)
            for v,raw,stored in zip(vals,src,entry['common_payoffs']):
                self.assertAlmostEqual(v,factor*raw+shift,places=10)
                self.assertAlmostEqual(v,stored,places=10)
            table.append(vals)
        points=list(itertools.product(range(3),repeat=4))
        def offset(a,b,c,d): return ((a*3+b)*3+c)*3+d
        for i,a in enumerate(table):
            for b in table[i+1:]:
                for p in itertools.permutations(range(3)):
                    self.assertTrue(any(abs(a[k]-b[offset(p[x],p[y],p[z],p[w])])>1e-9
                                        for k,(x,y,z,w) in enumerate(points)),
                                    'equivalent normalized games must not straddle splits')

    def test_permutation_distance_and_structural_distance_fixtures(self):
        from mamrl.generated_tasks import tensor_distance,structure_distance
        from mamrl.logical_rules import LogicalRule,Predicate,Payoff
        def indicator(action):
            cond=Predicate('eq',left='own',right=action)
            return LogicalRule(3,Payoff('if',condition=cond,
                                       children=(Payoff('const',value=1),Payoff('const',value=0))))
        a,b=indicator(0),indicator(1)
        self.assertAlmostEqual(tensor_distance(a,b),2/3)
        self.assertAlmostEqual(tensor_distance(a,b,canonical=True),0)
        self.assertAlmostEqual(structure_distance(a,a),0)
        self.assertGreaterEqual(structure_distance(a,b),0)
        self.assertLessEqual(structure_distance(a,b),1)
        self.assertAlmostEqual(structure_distance(a,b),structure_distance(b,a))

    def test_manifest_distance_separates_source_and_normalized_reward_scales(self):
        from mamrl.generated_tasks import distance_manifest
        from mamrl.logical_rules import rule_from_dict
        entries=self.manifest['rules']
        target=entries[12]
        raw=compile_rule(rule_from_dict(target['source_rule'])).values[:81]
        normalized=compile_rule(rule_from_dict(target['rule'])).values[:81]
        nearest_raw=float('inf')
        nearest_normalized=float('inf')
        for entry in entries[:4]:
            train_raw=compile_rule(rule_from_dict(entry['source_rule'])).values[:81]
            train_normalized=compile_rule(rule_from_dict(entry['rule'])).values[:81]
            nearest_raw=min(nearest_raw,sum(abs(a-b) for a,b in zip(raw,train_raw))/81)
            nearest_normalized=min(nearest_normalized,
                                   sum(abs(a-b) for a,b in zip(normalized,train_normalized))/81)
        distances=distance_manifest(self.manifest)['low_rules_low_opponents']['r012']
        self.assertAlmostEqual(distances['nearest_payoff_l1'],nearest_raw)
        self.assertAlmostEqual(distances['nearest_standardized_l1'],nearest_normalized)
        self.assertNotAlmostEqual(nearest_raw,nearest_normalized)

    def test_manifest_rejects_invalid_reward_and_heldout_training_reference(self):
        import copy
        from mamrl.generated_tasks import validate_manifest,CONDITIONS
        bad=copy.deepcopy(self.manifest)
        bad['rules'][0]['common_payoffs'][0] += .125
        with self.assertRaises((TypeError,ValueError)):
            validate_manifest(bad)
        bad=copy.deepcopy(self.manifest)
        heldout=next(t for t in bad['tasks'] if t['split']=='combination')
        bad['conditions'][CONDITIONS[0]].append(heldout['id'])
        with self.assertRaises((TypeError,ValueError)):
            validate_manifest(bad)

    def test_manifest_rejects_relabeling_heldout_pair_as_train(self):
        import copy
        from mamrl.generated_tasks import validate_manifest
        bad=copy.deepcopy(self.manifest)
        heldout=next(t for t in bad['tasks'] if t['split']=='combination')
        heldout['split']='train'
        # Add to the high/high condition as a corrupt but otherwise consistent manifest.
        bad['conditions']['high_rules_high_opponents'].append(heldout['id'])
        with self.assertRaises((TypeError,ValueError)):
            validate_manifest(bad)

    def test_reachable_action_information_has_future_value_for_a_rule_pair(self):
        """Exact finite-horizon Bayes witness, not a learned-policy success test.

        Two known rule hypotheses have equal prior. Initial previous actions
        (1,0) occur with probability 1/9; opponent copies learner's previous
        action. The first reward can reveal which rule was sampled. The erased
        control removes only that reward *as information*, keeping its score and
        allowing the learner to observe all subsequent rewards.
        """
        from functools import lru_cache
        from mamrl.logical_rules import rule_from_dict
        rules = [compile_rule(rule_from_dict(self.manifest['rules'][i]['rule'])) for i in (0,5)]
        self.assertEqual([self.manifest['rules'][i]['id'] for i in (0,5)],['r000','r005'])

        def outcomes(hyp,x,y,a):
            groups={}
            for i in hyp:
                reward=round(rules[i][0,x,y,a,x],9)
                groups.setdefault(reward,[]).append(i)
            return groups

        @lru_cache(None)
        def value(hyp,x,y,remaining):
            if remaining==0:
                return 0.
            return max(qvalue(hyp,x,y,remaining,a) for a in range(3))

        def qvalue(hyp,x,y,remaining,a):
            return sum(len(group)/len(hyp) * (reward+value(tuple(group),a,x,remaining-1))
                       for reward,group in outcomes(hyp,x,y,a).items())

        hyp=(0,1)
        immediate=[sum(rules[k][0,1,0,a,1] for k in hyp)/2 for a in range(3)]
        for actual,expected in zip(immediate,(1/6,1/2,2/3)):
            self.assertAlmostEqual(actual,expected)
        four_step=[qvalue(hyp,1,0,4,a) for a in range(3)]
        self.assertAlmostEqual(four_step[1],3.5,places=7)
        self.assertGreater(four_step[1],max(four_step[0],four_step[2]))
        self.assertLess(immediate[1],immediate[2])
        erased = immediate[1] + value(hyp,1,1,3)
        self.assertAlmostEqual(erased,3.0,places=7)
        self.assertAlmostEqual(four_step[1]-erased,.5,places=7)
        # Compare the research-side implementation with the independent DP;
        # it must make the same reward-erasure control and report its scope.
        from mamrl.generated_information import information_witness
        witness = information_witness(self.manifest)
        self.assertEqual(witness['rule_ids'],['r000','r005'])
        self.assertEqual(witness['initial_previous_joint_action'],[1,0])
        self.assertAlmostEqual(witness['initial_state_probability_under_uniform_reset'],1/9)
        self.assertEqual(witness['optimal_first_actions'],[1])
        self.assertEqual(witness['myopically_suboptimal_informative_optimal_actions'],[1])
        self.assertTrue(witness['demonstrates_positive_information_value'])
        for actual,expected in zip(witness['optimal_total_values_by_action'],four_step):
            self.assertAlmostEqual(actual,expected,places=7)
        self.assertAlmostEqual(witness['first_reward_blinded_total_values_by_action'][1],erased,places=7)


@unittest.skipUnless(torch is not None, 'optional torch training dependency')
class InformedBestResponseContract(unittest.TestCase):
    def test_expected_optimal_reward_in_hand_computable_coordination_game(self):
        from mamrl.generated_experiment import best_response_value
        from mamrl.generated_tasks import GeneratedTask
        from mamrl.logical_rules import LogicalRule,Predicate,Payoff
        # Both players obtain 1 exactly when current actions match.
        rule=LogicalRule(3,Payoff('if',condition=Predicate('eq',left='own',right='other'),
                                  children=(Payoff('const',value=1),Payoff('const',value=0))))
        def task(kind):
            return GeneratedTask('hand/'+kind,'hand',rule,kind,'test','hand')
        for kind,expected in (('random',1/3),('copy_previous',1.),('avoid_previous',.5)):
            with self.subTest(opponent=kind):
                self.assertAlmostEqual(best_response_value(task(kind),max_steps=3),expected)
        # Win-stay: a uniform first move matches with prob 1/3. On round 2,
        # conditional success is 1 after a match, 1/2 after a mismatch.
        self.assertAlmostEqual(best_response_value(task('win_stay'),max_steps=1),1/3)
        self.assertAlmostEqual(best_response_value(task('win_stay'),max_steps=2),.5)

    def test_finite_prior_uses_reward_and_opponent_likelihood(self):
        from mamrl.generated_experiment import PosteriorLearner
        from mamrl.generated_tasks import GeneratedTask
        from mamrl.logical_rules import LogicalRule,Predicate,Payoff
        from mamrl.environment import Observation
        rule=LogicalRule(3,Payoff('if',condition=Predicate('eq',left='own',right='other'),
                                  children=(Payoff('const',value=1),Payoff('const',value=0))))
        hypotheses=[GeneratedTask('r/'+k,'r',rule,k,'test','hand') for k in ('random','copy_previous')]
        learner=PosteriorLearner(hypotheses=hypotheses,family='generated',n=3,seed=0)
        before=Observation(1,2,0,False)
        after=Observation(1,1,1,False)
        learner.observe(before,1,1,after)
        # P(opponent chose 1)=1/3 under random, 1 under copy, equal prior.
        self.assertAlmostEqual(learner.weights[0],.25)
        self.assertAlmostEqual(learner.weights[1],.75)

    def test_runner_matches_training_collector_and_preserves_information_boundary(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.benchmarks import NeuralLearner
        from mamrl.generated_experiment import run_task
        from mamrl.generated_tasks import GeneratedTask
        from mamrl.logical_rules import LogicalRule,Predicate,Payoff
        from mamrl.training import collect_trial
        rule=LogicalRule(3,Payoff('if',condition=Predicate('eq',left='own',right='other'),
                                  children=(Payoff('const',value=1),Payoff('const',value=-1))))
        model=ActionSetActorCritic(hidden_size=4,seed=11)
        calls=[]
        def factory(**kwargs):
            calls.append(kwargs.copy())
            return NeuralLearner(model=model,**kwargs)
        for kind in ('random','copy_previous','avoid_previous','win_stay'):
            task=GeneratedTask('hand/'+kind,'hand',rule,kind,'test','hand')
            for reset in (False,True):
                actual=run_task(task,factory,seed=543,episodes_per_trial=2,max_steps=4,reset_memory=reset)
                oracle=collect_trial(model,rule,episodes_per_trial=2,max_steps=4,
                                     seed=543,reset_hidden=reset,opponent_kind=kind)
                self.assertEqual(actual['environment_seeds'],list(oracle.environment_seeds))
                self.assertEqual(actual['opponent_seeds'],list(oracle.opponent_seeds))
                self.assertEqual(actual['actions'],list(oracle.actions))
                self.assertEqual(actual['rewards'],list(oracle.rewards))
                self.assertEqual(actual['opponent_actions'],[s.actions[1] for e in oracle.episodes for s in e.transitions])
                self.assertEqual(actual['observations'],[dataclasses.asdict(oracle.events[i]) for i in oracle.action_event_indices])
        self.assertTrue(all(set(call)=={'family','n','seed'} and call['family']=='generated' for call in calls))

    def test_task_seed_does_not_depend_on_task_traversal_order(self):
        from mamrl.generated_experiment import evaluate_tasks
        from mamrl.generated_tasks import GeneratedTask
        from mamrl.benchmarks import RandomLearner
        from mamrl.logical_rules import LogicalRule,Predicate,Payoff
        rule=LogicalRule(3,Payoff('if',condition=Predicate('eq',left='own',right='other'),
                                  children=(Payoff('const',value=1),Payoff('const',value=0))))
        tasks=[GeneratedTask('hand/'+k,'hand',rule,k,'test','hand') for k in ('random','copy_previous')]
        kwargs=dict(reps=2,seed=42,episodes_per_trial=2,max_steps=3)
        a=evaluate_tasks(RandomLearner,tasks,**kwargs)
        b=evaluate_tasks(RandomLearner,list(reversed(tasks)),**kwargs)
        self.assertEqual({r['task_id']:r for r in a['tasks']},{r['task_id']:r for r in b['tasks']})

    def test_cli_records_shared_heldout_and_checkpoint_reproduces(self):
        from mamrl.benchmarks import NeuralLearner
        from mamrl.generated_experiment import evaluate_tasks
        from mamrl.generated_tasks import CONDITIONS,SPLITS,tasks_from_manifest,validate_manifest
        from mamrl.recurrent import load_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'pilot'
            cmd=[sys.executable,'-B','-m','mamrl.generated_experiment','--output-dir',str(out),
                 '--seeds','7','--updates','1','--trials-per-update','1','--max-steps','2',
                 '--hidden-size','4','--reps','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr[-5000:])
            manifest=json.loads((out/'manifest.json').read_text())
            record=json.loads((out/'experiment.json').read_text())
            protocol=json.loads((out/'protocol.json').read_text())
            validate_manifest(manifest)
            self.assertEqual(len(record['runs']),4)
            self.assertEqual({r['condition'] for r in record['runs']},set(CONDITIONS))
            self.assertEqual(protocol['decision_budget_per_condition'],4)
            for name, digest in protocol['source_sha256'].items():
                self.assertEqual(hashlib.sha256((out/'sources'/name).read_bytes()).hexdigest(),digest)
            for split in SPLITS:
                expected=[t.task_id for t in tasks_from_manifest(manifest,split=split)]
                for row in record['runs']:
                    actual=row['evaluations']['trained'][split]['tasks']
                    self.assertEqual([t['task_id'] for t in actual],expected)
                # Every model receives the exact same external trial seeds.
                seeds=[[t['trials'][0]['seed'] for t in row['evaluations']['trained'][split]['tasks']]
                       for row in record['runs']]
                self.assertTrue(all(s==seeds[0] for s in seeds))
                # Same initialization, task seeds and environment produce equal initial trajectories.
                initial=[row['evaluations']['initial'][split] for row in record['runs']]
                self.assertTrue(all(x==initial[0] for x in initial))
            first=record['runs'][0]
            run_dir=out/f"{first['condition']}_seed7"
            self.assertEqual(json.loads((run_dir/'evaluation.json').read_text()),first)
            model=load_checkpoint(out/first['checkpoint'])
            def factory(**kwargs):
                return NeuralLearner(model=model,**kwargs)
            tasks=tasks_from_manifest(manifest,split='combination')
            reproduced=evaluate_tasks(factory,tasks,reps=1,episodes_per_trial=2,max_steps=2)
            self.assertEqual(reproduced,first['evaluations']['trained']['combination'])
            self.assertIn('manifest.json',(out/'report.html').read_text())
            before=(out/'protocol.json').read_bytes()
            repeat=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180)
            self.assertNotEqual(repeat.returncode,0)
            self.assertEqual((out/'protocol.json').read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
