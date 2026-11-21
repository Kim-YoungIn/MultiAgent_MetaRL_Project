"""Independent D-030 contracts; these are engineering, not performance, tests."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mamrl.diverse_catalog import build_diverse_manifest, validate_diverse_manifest

try:
    import torch
except ImportError:
    torch = None


class DiverseCatalogContract(unittest.TestCase):
    def test_held_out_reservation_prefix_regeneration_and_canonical_uniqueness(self):
        for n in (3,4,5):
            with self.subTest(n=n):
                common = dict(n=n, seed=18, fresh_rules=2, structural_rules=2)
                small = build_diverse_manifest(**common, train_rules=2, diversity='broad')
                broad = build_diverse_manifest(**common, train_rules=4, diversity='broad')
                focused = build_diverse_manifest(**common, train_rules=4, diversity='focused')
                self.assertEqual(broad, build_diverse_manifest(**common, train_rules=4, diversity='broad'))
                for catalog in (small,broad,focused): validate_diverse_manifest(catalog)
                def split(catalog, name): return [r for r in catalog['rules'] if r['split']==name]
                for name in ('fresh_rules','structural'):
                    self.assertEqual(split(small,name),split(broad,name))
                    self.assertEqual(split(focused,name),split(broad,name))
                self.assertEqual(split(small,'train'),split(broad,'train')[:2])
                self.assertEqual({r['family'] for r in split(focused,'train')}, {'contest'})
                self.assertEqual({r['family'] for r in split(broad,'train')},
                                 {'social','contest','reciprocity','hidden'})
                # Independent pairwise permutation check on compiled payoff tensors.
                rows=broad['rules']; points=list(itertools.product(range(n), repeat=4))
                def key(row): return tuple(round(v,10) for v in row['common_payoffs'])
                keys=[key(r) for r in rows]
                for i in range(len(rows)):
                    a=keys[i]
                    self.assertEqual(len(a), n**4)
                    self.assertAlmostEqual(min(a),-1);self.assertAlmostEqual(max(a),1)
                    for j in range(i):
                        b=keys[j]
                        for p in itertools.permutations(range(n)):
                            remapped=tuple(b[((p[x]*n+p[y])*n+p[z])*n+p[w]] for x,y,z,w in points)
                            self.assertNotEqual(a,remapped)

    def test_tampering_and_invalid_sizes_rejected(self):
        catalog=build_diverse_manifest(n=3, seed=44, train_rules=3, fresh_rules=2, structural_rules=1)
        for mutate in ('payoff','seed','audit','duplicate'):
            data=copy.deepcopy(catalog)
            if mutate=='payoff': data['rules'][0]['common_payoffs'][0]+=1
            elif mutate=='seed': data['rules'][0]['seed']+=1
            elif mutate=='audit': data['catalog_audit']['train']['attempts']+=1
            else: data['rules'][1]=copy.deepcopy(data['rules'][0])
            with self.subTest(mutate=mutate), self.assertRaises((ValueError,KeyError,TypeError)):
                validate_diverse_manifest(data)
        for kwargs in ({'n':2}, {'n':6}, {'train_rules':0}, {'diversity':'wrong'}):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                build_diverse_manifest(**kwargs)


@unittest.skipUnless(torch is not None, 'optional torch dependency')
class VariableNTrainingContract(unittest.TestCase):
    def test_cli_train_save_and_reload_evaluation_at_each_n(self):
        from mamrl.experiment_artifacts import load_meta_experiment
        from mamrl.meta_exploration import evaluate_meta, meta_tasks, neural_factory
        from mamrl.recurrent import load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            for n in (3,4,5):
                with self.subTest(n=n):
                    out=Path(tmp)/str(n)
                    cmd=[sys.executable,'-B','-m','mamrl.meta_exploration','--output-dir',str(out),
                         '--catalog','diverse_relational','--n',str(n),'--train-rules','2',
                         '--fresh-rules','2','--structural-rules','2','--seeds','7',
                         '--updates','1','--trials-per-update','1','--steps','2',
                         '--assessment-steps','2','--hidden-size','4','--reps','1']
                    done=subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                        capture_output=True,text=True,timeout=120)
                    self.assertEqual(done.returncode,0,done.stderr[-4000:])
                    protocol=json.loads((out/'protocol.json').read_text())
                    self.assertEqual(protocol['n'],n)
                    manifest=json.loads((out/'manifest.json').read_text())
                    record=load_meta_experiment(out/'experiment.json'); run=record['runs'][0]
                    model=load_checkpoint(out/run['checkpoint'])
                    actual=evaluate_meta(neural_factory(model),meta_tasks(manifest,'fresh_rules'),
                                         steps=2,assessment_steps=2,reps=1)
                    self.assertEqual(actual,run['evaluation']['trained']['fresh_rules'])

    def test_fixed_checkpoint_transfer_paired_cost_and_immutability(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.action_transfer import run_action_transfer
        from mamrl.recurrent import save_checkpoint
        manifests={n:build_diverse_manifest(n=n,seed=71,train_rules=1,fresh_rules=1,structural_rules=1)
                   for n in (3,4,5)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'policy.pt'
            save_checkpoint(ActionSetActorCritic(hidden_size=4,seed=7),checkpoint)
            before=checkpoint.read_bytes(); output=root/'result'
            result=run_action_transfer(checkpoint,manifests,output,steps=2,assessment_steps=2,reps=1,seed=12)
            self.assertEqual(before,checkpoint.read_bytes())
            self.assertEqual(before,(output/'policy.pt').read_bytes())
            self.assertEqual(result['protocol']['checkpoint_sha256'],hashlib.sha256(before).hexdigest())
            self.assertTrue(result['protocol']['weights_unchanged'])
            self.assertEqual(json.loads((output/'study.json').read_text()),result)
            total=0
            for n in (3,4,5):
                self.assertEqual(json.loads((output/f'manifest_n{n}.json').read_text()),manifests[n])
                for split, group in result['evaluation'][str(n)].items():
                    self.assertIn(split,('fresh_rules','structural'))
                    a=group['trained']['rows']; b=group['random']['rows']
                    self.assertGreater(len(a),0)
                    fields=('task_id','rule_id','rep','seed','opponent_profile')
                    self.assertEqual([tuple(x[k] for k in fields) for x in a],
                                     [tuple(x[k] for k in fields) for x in b])
                    self.assertEqual(group['paired_trained_minus_random'],
                                     [x['assessment_mean']-y['assessment_mean'] for x,y in zip(a,b)])
                    for row in a+b:
                        self.assertFalse({'exploration','assessment','events'} & row.keys())
                        self.assertGreaterEqual(row['action_coverage'],1/n)
                        self.assertLessEqual(row['action_coverage'],1)
                    total+=(len(a)+len(b))*(2+2*2)
            self.assertEqual(result['protocol']['simulation_steps'],total)
            with self.assertRaisesRegex(ValueError,'action count'):
                run_action_transfer(checkpoint,{3:manifests[4]},root/'invalid',steps=2,assessment_steps=2)

    def test_manifest_n_mismatch_rejected(self):
        from mamrl.meta_exploration import run_meta_experiment
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); manifest=build_diverse_manifest(n=4,train_rules=1,fresh_rules=1,structural_rules=1)
            path=p/'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'manifest n'):
                run_meta_experiment(p/'out',catalog='diverse_relational',n=3,manifest_path=path,
                                    seeds=(7,),updates=1,trials_per_update=1,steps=2,hidden_size=4,reps=1)

if __name__ == '__main__': unittest.main()
