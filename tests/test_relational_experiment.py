"""Independent checks of PD labels, broader opponents and transfer protocol."""
from collections import Counter
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from mamrl.environment import Observation

try:
    import torch
except ImportError:
    torch = None


class CompiledAuditContract(unittest.TestCase):
    def test_exact_pd_requires_all_quadrants_and_unilateral_dominance(self):
        from mamrl.relational_catalog import audit_relational_rule
        from mamrl.relational_rules import social_rule

        for n in (3,4,5):
            pd=audit_relational_rule(social_rule(n=n,T=5,R=3,P=1,S=0))['pd']
            self.assertTrue(pd['exact_embedding'])
            self.assertTrue(pd['quadrants_reachable'])
            self.assertTrue(pd['is_prisoners_dilemma'])
            self.assertTrue(pd['defection_strictly_dominates'])
            self.assertTrue(pd['two_R_gt_T_plus_S'])
            self.assertEqual(pd['payoffs'],{'T':5.0,'R':3.0,'P':1.0,'S':0.0})

        # 2R>T+S is a separately reported optional property, not part of PD.
        pd=audit_relational_rule(social_rule(n=3,T=7,R=3,P=1,S=0))['pd']
        self.assertTrue(pd['is_prisoners_dilemma'])
        self.assertFalse(pd['two_R_gt_T_plus_S'])
        non_pd=audit_relational_rule(social_rule(n=3,T=2,R=3,P=1,S=0))['pd']
        self.assertTrue(non_pd['exact_embedding'])
        self.assertFalse(non_pd['is_prisoners_dilemma'])

    def test_hidden_action_bonus_invalidates_pd_class_equivalence(self):
        from mamrl.relational_catalog import audit_relational_rule
        from mamrl.relational_rules import Relation, Reward, social_rule

        pd=social_rule(n=3,T=5,R=3,P=1,S=0)
        hidden=Reward('if',condition=Relation('eq',left='own',right=0),
                      children=(Reward('const',value=2),Reward('const',value=0)))
        modified=replace(pd,root=Reward('add',children=(pd.root,hidden)))
        audit=audit_relational_rule(modified)['pd']
        self.assertFalse(audit['exact_embedding'])
        self.assertFalse(audit['is_prisoners_dilemma'])

    def test_current_match_does_not_make_repeat_change_pd(self):
        from mamrl.relational_catalog import audit_relational_rule
        from mamrl.relational_rules import Relation, Reward, RelationalRule, cyclic_outcomes

        same=Relation('eq',left='own',right='other')
        rule=RelationalRule(3,cyclic_outcomes(3),
                            Reward('if',condition=same,children=(Reward('const',value=5),Reward('const',value=0))))
        audit=audit_relational_rule(rule)['pd']
        self.assertFalse(audit['exact_embedding'])
        self.assertFalse(audit['is_prisoners_dilemma'])

    def test_independent_payoffs_can_depend_on_both_players_without_strategic_dependence(self):
        from mamrl.relational_catalog import audit_relational_rule
        from mamrl.relational_rules import Relation, Reward, RelationalRule, cyclic_outcomes

        def term(variable):
            return Reward('if',condition=Relation('eq',left=variable,right=0),
                          children=(Reward('const',value=1),Reward('const',value=0)))
        rule=RelationalRule(3,cyclic_outcomes(3),Reward('add',children=(term('own'),term('other'))))
        audit=audit_relational_rule(rule)
        self.assertTrue(audit['axis_dependence'][2])
        self.assertTrue(audit['axis_dependence'][3])
        # A unique dominant physical action exists at every state: payoff
        # dependence on the opponent does not imply best-response dependence.
        for state in audit['best_responses']:
            response_sets=state['by_opponent_action']
            self.assertTrue(all(set(values)=={0} for values in response_sets))


class RelationalManifestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mamrl.relational_catalog import build_relational_manifest
        cls.global_rng_before=random.getstate()
        cls.manifest=build_relational_manifest(n=3,seed=20261114)
        cls.global_rng_after=random.getstate()

    def test_splits_distinct_reproducible_and_pd_preselected(self):
        from mamrl.relational_catalog import build_relational_manifest,validate_relational_manifest

        manifest=self.manifest
        self.assertEqual(self.global_rng_before,self.global_rng_after)
        self.assertEqual(build_relational_manifest(n=3,seed=20261114),manifest)
        self.assertEqual(manifest['version'],2)
        self.assertEqual(manifest['catalog'],'relational')
        self.assertEqual(Counter(r['split'] for r in manifest['rules']),
                         {'train':12,'fresh_rules':8,'structural':8})
        self.assertEqual(Counter(r['family'] for r in manifest['rules'] if r['split']=='train'),
                         {'social':3,'contest':3,'reciprocity':3,'hidden':3})
        self.assertTrue(any(r['audit']['pd']['is_prisoners_dilemma'] for r in manifest['rules'] if r['split']=='train'))
        fingerprints=[r['canonical_fingerprint'] for r in manifest['rules']]
        self.assertEqual(len(fingerprints),len(set(fingerprints)))
        self.assertIsNotNone(validate_relational_manifest(manifest))

    def test_validation_rejects_modified_metadata_and_compiled_payoffs(self):
        from mamrl.relational_catalog import validate_relational_manifest

        bad=copy.deepcopy(self.manifest)
        bad['rules'][0]['audit']['pd']['is_prisoners_dilemma']=False
        with self.assertRaises((TypeError,ValueError)):
            validate_relational_manifest(bad)
        bad=copy.deepcopy(self.manifest)
        bad['rules'][0]['rule']['shift']+=.25
        with self.assertRaises((TypeError,ValueError)):
            validate_relational_manifest(bad)
        bad=copy.deepcopy(self.manifest)
        bad['rules'][-1]['split']='train'
        with self.assertRaises((TypeError,ValueError)):
            validate_relational_manifest(bad)
        # A recorded generation seed is provenance, not arbitrary decoration.
        bad=copy.deepcopy(self.manifest)
        bad['rules'][1]['seed']+=1
        with self.assertRaises((TypeError,ValueError)):
            validate_relational_manifest(bad)

    def test_serialized_rules_compile_to_normalized_nonconstant_tensors(self):
        from mamrl.relational_rules import relational_from_dict
        from mamrl.rules import compile_rule

        for entry in self.manifest['rules']:
            rule=relational_from_dict(entry['rule'])
            tensor=compile_rule(rule)
            source=compile_rule(relational_from_dict(entry['source_rule']))
            factor=entry['normalization']['factor']
            shift=entry['normalization']['shift']
            self.assertGreater(factor,0)
            for actual,original in zip(tensor.values,source.values):
                self.assertAlmostEqual(actual,factor*original+shift)
            values=tensor.values[:rule.n**4]
            self.assertAlmostEqual(min(values),-1)
            self.assertAlmostEqual(max(values),1)
            self.assertTrue(entry['description'])
            for p,q,a,b in ((0,1,2,0),(2,0,1,1)):
                self.assertEqual(tensor[1,p,q,a,b],tensor[0,q,p,b,a])


class ExpectedSarsaContract(unittest.TestCase):
    def test_expected_sarsa_target_differs_from_max_q(self):
        from mamrl.adaptive_agents import OnlineQ, ExpectedSARSA

        current=Observation(0,0,0,False)
        following=Observation(1,2,1,False)
        agents=[OnlineQ(3,17,alpha=1,gamma=.95,epsilon=.2),
                ExpectedSARSA(3,17,alpha=1,gamma=.95,epsilon=.2)]
        for agent in agents:
            agent.update(following,0,2.0,following,terminal=True)
            agent.update(following,1,1.0,following,terminal=True)
            # epsilon=.2: probabilities are (13/15,1/15,1/15),
            # so E[Q']=1.8 while max Q'=2.
            agent.update(current,0,1.0,following,terminal=False)
        self.assertAlmostEqual(agents[0].values(current)[0],2.9)
        self.assertAlmostEqual(agents[1].values(current)[0],2.71)

    def test_clone_keeps_algorithm_rng_and_values_and_is_independent(self):
        from mamrl.adaptive_agents import ExpectedSARSA

        obs=Observation(1,2,1,False)
        agent=ExpectedSARSA(3,72,alpha=.5,gamma=.95,epsilon=.2)
        agent.update(obs,1,1,obs,terminal=True)
        cloned=agent.clone()
        self.assertIs(type(cloned),type(agent))
        self.assertEqual(cloned.snapshot(),agent.snapshot())
        self.assertEqual([cloned.act(obs) for _ in range(30)],[agent.act(obs) for _ in range(30)])
        cloned.update(obs,0,7,obs,terminal=True)
        self.assertNotEqual(cloned.values(obs),agent.values(obs))
        self.assertIn('algorithm',agent.snapshot())

    def test_paired_assessments_start_from_identical_sarsa_opponent(self):
        from mamrl.meta_exploration_rollout import run_meta_trial
        from mamrl.meta_exploration import RandomExplorer
        from mamrl.relational_rules import social_rule

        trial=run_meta_trial(social_rule(),RandomExplorer,seed=51,steps=12,opponent_profile='sarsa')
        self.assertEqual(trial['opponent_before_assessment'],trial['erased_opponent_before_assessment'])
        self.assertEqual(trial['opponent_final']['algorithm'],trial['erased_opponent_final']['algorithm'])
        self.assertEqual(trial['assessment'][0]['observations'],trial['erased_assessment'][0]['observations'])
        self.assertEqual(trial['opponent_final']['updates'],24)
        self.assertGreater(trial['opponent_final']['parameter_changes'],0)


class BalancedActionContract(unittest.TestCase):
    def test_repeat_change_reference_is_semantically_balanced(self):
        from mamrl.meta_exploration import BalancedExplorer

        for n in (3,4,5):
            for own in range(n):
                obs=Observation(own,(own+1)%n,0,False)
                baseline=BalancedExplorer(n=n,seed=17,student=None)
                counts=Counter(baseline.act(obs) for _ in range(6000))
                self.assertAlmostEqual(counts[own]/6000,.5,delta=.035)
                for action in range(n):
                    if action!=own:
                        self.assertAlmostEqual(counts[action]/6000,.5/(n-1),delta=.035)


class AssessmentMetricContract(unittest.TestCase):
    def test_prefix_auc_is_mean_reward_with_deduplicated_short_horizon(self):
        from mamrl.meta_exploration import summarize_meta

        def row(task, rewards):
            return {'task_id':task,'meta_reward':0,'assessment_mean':sum(rewards)/len(rewards),
                    'erased_mean':0,'exploration_mean':0,
                    'assessment':[{'rewards':[r,0]} for r in rewards],
                    'exploration':[{'rewards':[0,0]} for _ in rewards],
                    'opponent_final':{'parameter_changes':0,'policy_l1_change':0},
                    'opponent_nonuniform_action_events':0}
        summary=summarize_meta([row('a',[1,0,0,1]),row('b',[0,1,1,0])])
        self.assertEqual(summary['assessment_prefix_auc'],{'1':.5,'3':.5,'4':.5})
        short=summarize_meta([row('a',[1,0])])
        self.assertEqual(short['assessment_prefix_auc'],{'1':1,'2':.5})


@unittest.skipUnless(torch is not None,'optional torch dependency')
class RelationalEndToEndContract(unittest.TestCase):
    def test_tiny_cli_source_provenance_and_checkpoint_reevaluation(self):
        from mamrl.meta_exploration import evaluate_meta,meta_tasks,neural_factory
        from mamrl.recurrent import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'experiment'
            cmd=[sys.executable,'-B','-m','mamrl.meta_exploration','--output-dir',str(out),
                 '--catalog','relational','--opponent-suite','mixed_learning','--seeds','7',
                 '--updates','1','--trials-per-update','1','--steps','2','--hidden-size','4','--reps','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            result=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr[-5000:])
            protocol=json.loads((out/'protocol.json').read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            from mamrl.experiment_artifacts import load_meta_experiment
            record=load_meta_experiment(out/'experiment.json')
            self.assertEqual(protocol['catalog'],'relational')
            self.assertEqual(protocol['opponent_suite'],'mixed_learning')
            for filename,digest in protocol['source_sha256'].items():
                self.assertEqual(hashlib.sha256((out/'sources'/filename).read_bytes()).hexdigest(),digest)
            row=record['runs'][0]
            self.assertEqual(json.loads((out/'seed7'/'evaluation.json').read_text()),row)
            model=load_checkpoint(out/row['checkpoint'])
            tasks=meta_tasks(manifest,'fresh_rules',opponent_suite='mixed_learning')
            reproduced=evaluate_meta(neural_factory(model),tasks,steps=2,reps=1)
            self.assertEqual(reproduced,row['evaluation']['trained']['fresh_rules'])
            self.assertIn('balanced',record['baselines'])
            before=(out/'protocol.json').read_bytes()
            failed=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=180)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(before,(out/'protocol.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
