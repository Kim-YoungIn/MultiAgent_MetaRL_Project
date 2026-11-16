"""Independent, hand-computed contracts for probe games and history replay."""
import copy
import unittest
from mamrl.probe_templates import build_probe_rule, build_probe_manifest, validate_probe_manifest, exact_information_audit
from mamrl.rules import compile_rule


class ProbeTemplateContract(unittest.TestCase):
    def test_fixed_opponent_information_witness(self):
        a=exact_information_audit(horizon=2)
        self.assertAlmostEqual(a['no_information_total'],2)
        self.assertAlmostEqual(a['optimal_total'],2.9)
        self.assertAlmostEqual(a['oracle_total'],5.8)
        self.assertAlmostEqual(a['information_value'],.9)
        self.assertEqual(a['optimal_initial_actions'],[2])
        self.assertAlmostEqual(exact_information_audit(horizon=1)['information_value'],0)
        self.assertAlmostEqual(exact_information_audit(safe_reward=4,horizon=2)['information_value'],0)

    def test_hand_payoffs_and_exchange_symmetry(self):
        for theta in (-1,1):
            tensor=compile_rule(build_probe_rule(theta=theta))
            for p in range(3):
                for q in range(3):
                    self.assertEqual(tensor[0,p,q,0,0],1)
                    self.assertAlmostEqual(tensor[0,p,q,1,0],theta*3-.2)
                    self.assertAlmostEqual(tensor[0,p,q,2,0],-theta*3)
                    for a in range(3):
                        for b in range(3):
                            self.assertEqual(tensor[0,p,q,a,b],tensor[1,q,p,b,a])

    def test_role_renaming_does_not_create_a_new_structural_game(self):
        from mamrl.rule_metrics import canonical_values, common_values
        a=build_probe_rule(theta=1,roles=(0,1,2))
        b=build_probe_rule(theta=1,roles=(2,0,1))
        opposite=build_probe_rule(theta=-1,roles=(0,1,2))
        self.assertEqual(canonical_values(common_values(a),3),canonical_values(common_values(b),3))
        self.assertNotEqual(canonical_values(common_values(a),3),canonical_values(common_values(opposite),3))

    def test_temporal_elements_seen_alone_then_combined(self):
        manifest=build_probe_manifest(seed=123,train_rules=4,fresh_rules=2,structural_rules=2)
        train=[r['parameters'] for r in manifest['rules'] if r['split']=='train']
        held=[r['parameters'] for r in manifest['rules'] if r['split']=='structural']
        self.assertEqual({(p['repeat_bonus'],p['copy_bonus']) for p in train},{(.15,0.0),(0.0,.15)})
        self.assertEqual({(p['repeat_bonus'],p['copy_bonus']) for p in held},{(.15,.15)})
        # Check additive semantics with hand-computed conditions: previous own=1,
        # previous other=2; own 1 is repeat, own 2 is copy, neither is both.
        baseline=compile_rule(build_probe_rule())
        repeat=compile_rule(build_probe_rule(repeat_bonus=.15))
        copy=compile_rule(build_probe_rule(copy_bonus=.15))
        both=compile_rule(build_probe_rule(repeat_bonus=.15,copy_bonus=.15))
        for a,bonus_repeat,bonus_copy in [(0,0,0),(1,.15,0),(2,0,.15)]:
            base=baseline[0,1,2,a,0]
            self.assertAlmostEqual(repeat[0,1,2,a,0]-base,bonus_repeat)
            self.assertAlmostEqual(copy[0,1,2,a,0]-base,bonus_copy)
            self.assertAlmostEqual(both[0,1,2,a,0]-base,bonus_repeat+bonus_copy)
        # When both previous actions match the present own action, bonuses add.
        self.assertAlmostEqual(both[0,1,1,1,0]-baseline[0,1,1,1,0],.3)

    def test_manifest_pairs_normalization_and_seed_provenance(self):
        m=build_probe_manifest(seed=123,train_rules=4,fresh_rules=4,structural_rules=4)
        self.assertEqual(m,build_probe_manifest(seed=123,train_rules=4,fresh_rules=4,structural_rules=4))
        self.assertEqual([sum(e['split']==s for e in m['rules']) for s in ('train','fresh_rules','structural')],[4,4,4])
        fingerprints=[e['canonical_fingerprint'] for e in m['rules']]
        self.assertEqual(len(fingerprints),len(set(fingerprints)))
        from mamrl.relational_rules import relational_from_dict
        for i in range(0,12,2):
            left,right=m['rules'][i:i+2]
            self.assertEqual(left['seed'],right['seed'])
            self.assertNotEqual(left['canonical_fingerprint'],right['canonical_fingerprint'])
            a,b=compile_rule(relational_from_dict(left['rule'])),compile_rule(relational_from_dict(right['rule']))
            safe=left['parameters']['roles'][0]
            # Structural bonuses do not depend on theta, so safe reward still agrees.
            for p in range(3):
                for q in range(3):
                    for other in range(3):
                        self.assertAlmostEqual(a[0,p,q,safe,other],b[0,p,q,safe,other])
        broken=copy.deepcopy(m);broken['seed']+=1
        with self.assertRaises(ValueError):validate_probe_manifest(broken)


class HistoryContract(unittest.TestCase):
    def test_replay_bootstraps_from_assessment_reset_not_terminal_observation(self):
        from mamrl.history_study import replay_history
        from mamrl.environment import Observation
        reset=Observation(2,1,0,True)
        middle=Observation(0,0,1,False)
        terminal=Observation(1,1,0,False)
        def data(o):
            return {'previous_own_action':o.previous_own_action,'previous_opponent_action':o.previous_opponent_action,
                    'previous_own_reward':o.previous_own_reward,'episode_start_flag':o.episode_start_flag}
        events=[{'observations':[data(reset)],'actions':[0],'rewards':[2],
                 'learning_next_observations':[data(middle)],'terminal':False},
                {'observations':[data(middle)],'actions':[1],'rewards':[0],
                 'next_observations':[data(terminal)],'learning_next_observations':[data(reset)],'terminal':False}]
        student=replay_history(events,n=3,seed=7)
        self.assertEqual(student.updates,2)
        self.assertAlmostEqual(student.values(reset)[0],1)
        self.assertAlmostEqual(student.values(middle)[1],.475)

    def test_full_replay_and_conditional_opponent_are_shared(self):
        from mamrl.history_study import collect_history_context,replay_history,score_history
        from mamrl.meta_exploration import RandomExplorer,RepeatExplorer
        rule=build_probe_rule()
        random_trial,reference=collect_history_context(rule,RandomExplorer,seed=111,steps=4,assessment_steps=6)
        trial,capture=collect_history_context(rule,RepeatExplorer,seed=111,steps=4,assessment_steps=6)
        replay=replay_history(trial['exploration'],n=3,seed=capture['seeds']['student'])
        self.assertEqual(replay.snapshot(),capture['student'].snapshot())
        scores=[score_history(trial,capture,reference,representation=r) for r in ('full','actions','binned')]
        self.assertEqual(len({s['history_sha256'] for s in scores}),1)
        self.assertTrue(all(s['student_updates_from_history']==4 for s in scores))
        self.assertTrue(all(s['opponent_before']==reference['opponent'].snapshot() for s in scores))
        self.assertTrue(all(len(s['assessment_curve'])==6 for s in scores))
        self.assertEqual(random_trial['assessment'][0]['observations'],trial['assessment'][0]['observations'])

    def test_theta_identifiability_requires_a_diagnostic_action(self):
        from mamrl.history_study import paired_identifiability
        from mamrl.relational_rules import relational_from_dict
        manifest=build_probe_manifest(seed=123,train_rules=2,fresh_rules=2,structural_rules=2)
        pair=manifest['rules'][:2]
        tensor=compile_rule(relational_from_dict(pair[0]['rule']))
        safe,costly,free=pair[0]['parameters']['roles']
        def event(action):
            return {'observations':[{'previous_own_action':safe,'previous_opponent_action':safe}],
                    'actions':[action,safe],'rewards':[tensor[0,safe,safe,action,safe]]}
        identity=paired_identifiability([event(safe),event(free)],pair)
        self.assertEqual(identity['candidate_counts'],[2,2,1])
        self.assertEqual(identity['first_unique_step'],2)
        self.assertEqual(identity['final_entropy_bits'],0)

    def test_threshold_censor_and_sustained_windows(self):
        from mamrl.history_study import threshold_time
        self.assertEqual(threshold_time([0,1,1,1],threshold=.6,window=2,sustain=2),
                         {'step':4,'reached':True,'right_censored':False,'horizon':4})
        self.assertEqual(threshold_time([0,1,0,1],threshold=.6,window=2,sustain=2),
                         {'step':None,'reached':False,'right_censored':True,'horizon':4})


class ProbeCLIContract(unittest.TestCase):
    def test_tiny_training_checkpoint_and_history_roundtrip(self):
        import json,os,subprocess,sys,tempfile
        from pathlib import Path
        try:
            import torch
        except ImportError:
            self.skipTest('optional PyTorch')
        from mamrl.experiment_artifacts import load_meta_experiment
        from mamrl.meta_exploration import evaluate_meta,meta_tasks,neural_factory
        from mamrl.recurrent import load_checkpoint
        from mamrl.history_study import run_history_study
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'training';study=Path(tmp)/'history'
            command=[sys.executable,'-B','-m','mamrl.meta_exploration','--catalog','probe_template',
                     '--train-rules','4','--fresh-rules','2','--structural-rules','2',
                     '--opponent-suite','mixed_learning','--steps','2','--assessment-steps','5',
                     '--seeds','7','--updates','1','--trials-per-update','2','--batch-size','2',
                     '--hidden-size','8','--reps','1','--device','cpu','--output-dir',str(out)]
            result=subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                  capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr[-6000:])
            manifest=json.loads((out/'manifest.json').read_text());validate_probe_manifest(manifest)
            record=load_meta_experiment(out/'experiment.json');self.assertEqual(record['protocol']['catalog'],'probe_template')
            self.assertEqual(record['protocol']['inner_training_gameplay_steps_per_seed'],14)
            self.assertEqual(record['protocol']['inner_training_control_steps_per_seed'],10)
            self.assertIn('systematic',record['baselines'])
            model=load_checkpoint(out/record['runs'][0]['checkpoint'])
            reproduced=evaluate_meta(neural_factory(model),meta_tasks(manifest,'fresh_rules',opponent_suite='mixed_learning'),
                                     steps=2,assessment_steps=5,reps=1,batch_size=2)
            self.assertEqual(reproduced,record['runs'][0]['evaluation']['trained']['fresh_rules'])
            info=json.loads((out/'information_audit.json').read_text())
            self.assertAlmostEqual(info['positive']['information_value'],.9)
            history=run_history_study(manifest,output_dir=study,checkpoint=out/record['runs'][0]['checkpoint'],
                                      steps=2,assessment_steps=5,reps=1)
            self.assertTrue((study/'report.html').exists())
            self.assertEqual(history['protocol']['K'],2);self.assertEqual(history['protocol']['H'],5)
            self.assertIn('learned',history['summary'])
            self.assertTrue(all(r['student_updates_from_history']==2 for r in history['rows']))


class SharedManifestDiversityContract(unittest.TestCase):
    def test_nested_training_limits_keep_identical_evaluation_and_cost(self):
        import json,os,subprocess,sys,tempfile
        from pathlib import Path
        try:
            import torch
        except ImportError:
            self.skipTest('optional PyTorch')
        from mamrl.experiment_artifacts import load_meta_experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=build_probe_manifest(seed=222,train_rules=4,fresh_rules=2,structural_rules=2)
            manifest_path=root/'shared.json';manifest_path.write_text(json.dumps(manifest))
            common=[sys.executable,'-B','-m','mamrl.meta_exploration','--catalog','probe_template',
                    '--manifest',str(manifest_path),'--opponent-suite','mixed_learning',
                    '--seeds','7','--updates','1','--trials-per-update','4','--steps','2',
                    '--assessment-steps','5','--hidden-size','8','--batch-size','2','--reps','1','--device','cpu']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'); results={}
            for limit in (2,4):
                path=root/f'limit{limit}'
                proc=subprocess.run([*common,'--train-rule-limit',str(limit),'--output-dir',str(path)],
                                    capture_output=True,text=True,env=env,timeout=180)
                self.assertEqual(proc.returncode,0,proc.stderr[-6000:])
                self.assertEqual(json.loads((path/'manifest.json').read_text()),manifest)
                record=load_meta_experiment(path/'experiment.json'); results[limit]=record
                ids=record['protocol']['training_rule_ids']
                self.assertEqual(ids,[r['id'] for r in manifest['rules'] if r['split']=='train'][:limit])
                self.assertEqual(record['protocol']['train_rule_limit'],limit)
                self.assertTrue({t['rule_id'] for h in record['runs'][0]['history'] for t in h['trials']} <= set(ids))
                for split, evaluation in record['runs'][0]['evaluation']['trained'].items():
                    for row in evaluation['trials']:
                        self.assertEqual(row['seen_rule'],row['rule_id'] in ids)
                        self.assertEqual(row['environment_steps_with_control'],12)
            for split in ('train_rule_reference','fresh_rules','structural','new_learning_profile'):
                for stage in ('initial','trained'):
                    def identity(record):
                        return [(t['task_id'],t['rep'],t['seed']) for t in record['runs'][0]['evaluation'][stage][split]['trials']]
                    self.assertEqual(identity(results[2]),identity(results[4]))
            for limit in (3,6):
                proc=subprocess.run([*common,'--train-rule-limit',str(limit),'--output-dir',str(root/f'invalid{limit}')],
                                    capture_output=True,text=True,env=env,timeout=180)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn('limit',proc.stderr.lower())
