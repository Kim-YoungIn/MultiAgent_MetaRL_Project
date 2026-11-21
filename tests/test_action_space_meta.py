"""Action-count integration checks for the meta-exploration execution path."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'optional torch dependency')
class MetaActionSpaceContract(unittest.TestCase):
    def test_meta_training_and_portable_evaluation_at_n4_n5(self):
        from mamrl.meta_exploration import MetaConfig, train_meta, meta_tasks, evaluate_meta, neural_factory
        from mamrl.relational_rules import social_rule, relational_to_dict
        from mamrl.recurrent import save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            for n in (4, 5):
                # A hand-specified logical rule supplies three evaluation splits;
                # it tests action count rather than catalog diversity.
                rule=relational_to_dict(social_rule(n=n))
                manifest={'n':n,'rules':[{'id':f'r{i}', 'split':split, 'rule':rule}
                           for i,split in enumerate(('train','fresh_rules','structural'))]}
                config=MetaConfig(seed=7,updates=1,trials_per_update=2,steps=3,
                                  assessment_steps=4,hidden_size=4,batch_size=2)
                trained=train_meta(config,manifest,opponent_suite='mixed_learning')
                self.assertEqual(len(trained['history']),1)
                weight=Path(tmp)/f'n{n}.pt'
                save_checkpoint(trained['model'],weight)
                loaded=load_checkpoint(weight)
                for key,value in trained['model'].state_dict().items():
                    self.assertTrue(torch.equal(value.cpu(),loaded.state_dict()[key].cpu()))
                tasks=meta_tasks(manifest,'fresh_rules',opponent_suite='mixed_learning')
                assessment=evaluate_meta(neural_factory(loaded),tasks,steps=3,assessment_steps=4,reps=1,batch_size=2)
                self.assertEqual(len(assessment['trials']),2)
                for trial in assessment['trials']:
                    self.assertEqual(trial['n'],n)
                    self.assertEqual(len(trial['exploration']),3)
                    self.assertEqual(len(trial['assessment']),4)
                    for event in trial['exploration']+trial['assessment']:
                        self.assertTrue(all(0<=a<n for a in event['actions']))
                    self.assertGreater(trial['opponent_final']['updates'],0)

    def test_invalid_action_counts_rejected_for_selected_catalog(self):
        from mamrl.meta_exploration import run_meta_experiment
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for n in (2,6,True):
                with self.subTest(n=n), self.assertRaises((TypeError,ValueError)):
                    run_meta_experiment(p/f'bad{n}',n=n,catalog='relational')
            for catalog in ('logical','probe_template'):
                with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                    run_meta_experiment(p/catalog,n=4,catalog=catalog)

    def test_manifest_action_count_must_match_request(self):
        from mamrl.meta_exploration import run_meta_experiment
        from mamrl.diverse_catalog import build_diverse_manifest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); manifest=p/'manifest.json'
            manifest.write_text(json.dumps(build_diverse_manifest(n=4,train_rules=4,fresh_rules=2,structural_rules=2)))
            with self.assertRaisesRegex(ValueError,'manifest n'):
                run_meta_experiment(p/'bad',catalog='diverse_relational',manifest_path=manifest,n=3)

    def test_n4_cli_builds_catalog_and_records_action_count(self):
        from mamrl.experiment_artifacts import load_meta_experiment
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            cmd=[sys.executable,'-B','-m','mamrl.meta_exploration','--catalog','diverse_relational',
                 '--n','4','--diversity','focused','--train-rules','4','--fresh-rules','2',
                 '--structural-rules','2','--opponent-suite','mixed_learning','--output-dir',str(output),
                 '--seeds','7','--updates','1','--trials-per-update','2','--steps','3',
                 '--assessment-steps','4','--hidden-size','4','--batch-size','2','--reps','1']
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=120,
                                  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            self.assertEqual(result.returncode,0,result.stderr[-4000:])
            protocol=json.loads((output/'protocol.json').read_text())
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual((protocol['n'],manifest['n']),(4,4))
            self.assertEqual(protocol['diversity'],'focused')
            self.assertTrue((output/'seed7'/'policy.pt').is_file())
            record=load_meta_experiment(output/'experiment.json')
            self.assertIn('change',record['baselines'])
            for trial in record['runs'][0]['evaluation']['trained']['fresh_rules']['trials']:
                self.assertEqual(trial['n'],4)
                for event in trial['exploration']+trial['assessment']:
                    self.assertTrue(all(0<=action<4 for action in event['actions']))
