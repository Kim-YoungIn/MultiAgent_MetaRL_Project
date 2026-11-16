"""Independent CPU/CUDA batching and update-boundary resume contracts."""
import copy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

@unittest.skipUnless(torch is not None, 'optional PyTorch')
class AccelerationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mamrl.generated_tasks import build_manifest
        cls.manifest = build_manifest()
        torch.set_num_threads(1)

    def test_batch_preserves_each_trial_and_replay_gradient(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.meta_exploration import collect_exploration, replay_exploration, decode_rule
        from mamrl.meta_batch import collect_batch, replay_batch
        model = ActionSetActorCritic(hidden_size=8, seed=7)
        rules = [decode_rule(r['rule']) for r in self.manifest['rules'][:2]]
        specs = [(rules[0], 345, 'steady'), (rules[1], 678, 'rapid')]
        batch = collect_batch(model, specs, steps=4)
        scalar = [collect_exploration(model, r, seed=s, steps=4, opponent_profile=p) for r,s,p in specs]
        for left,right in zip(batch,scalar):
            self.assertEqual(left.actions,right.actions)
            self.assertEqual(left.trial,right.trial)
            self.assertEqual(left.trial['opponent_before_assessment'],left.trial['erased_opponent_before_assessment'])
            self.assertEqual(left.trial['environment_steps_with_control'],12)
            self.assertEqual(left.trial['assessment'][0]['observations'],left.trial['erased_assessment'][0]['observations'])
        outputs=replay_batch(model,batch)
        expected=[torch.cat([replay_exploration(model,r)[i] for r in batch]) for i in range(3)]
        for a,b in zip(outputs,expected): self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))
        g1=torch.autograd.grad(sum(v.sum() for v in outputs),model.gru.weight_ih,retain_graph=True)[0]
        g2=torch.autograd.grad(sum(v.sum() for v in expected),model.gru.weight_ih)[0]
        self.assertTrue(torch.allclose(g1,g2,atol=1e-5,rtol=1e-5))

    def test_resume_matches_uninterrupted_and_rejects_changed_contract(self):
        from mamrl.meta_exploration import MetaConfig,train_meta
        config=MetaConfig(seed=91,updates=3,trials_per_update=3,steps=3,hidden_size=8,batch_size=2)
        full=train_meta(config,self.manifest)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'training.pt'
            paused=train_meta(config,self.manifest,checkpoint_path=checkpoint,stop_after_update=1)
            self.assertFalse(paused['completed'])
            resumed=train_meta(config,self.manifest,checkpoint_path=checkpoint,resume=True)
            self.assertTrue(resumed['completed'])
            self.assertEqual(full['history'],resumed['history'])
            for key,value in full['model'].state_dict().items():
                self.assertTrue(torch.equal(value,resumed['model'].state_dict()[key]),key)
            changed=copy.deepcopy(self.manifest);changed['seed']+=1
            with self.assertRaises(ValueError):
                train_meta(config,changed,checkpoint_path=checkpoint,resume=True)
            with self.assertRaises(ValueError):
                train_meta(MetaConfig(seed=91,updates=3,trials_per_update=3,steps=3,hidden_size=8,batch_size=1),self.manifest,checkpoint_path=checkpoint,resume=True)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training_and_resume(self):
        from mamrl.meta_exploration import MetaConfig,train_meta
        cfg=MetaConfig(seed=3,updates=2,trials_per_update=2,steps=2,hidden_size=8,batch_size=2,device='cuda')
        full=train_meta(cfg,self.manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            self.assertFalse(train_meta(cfg,self.manifest,checkpoint_path=path,stop_after_update=1)['completed'])
            result=train_meta(cfg,self.manifest,checkpoint_path=path,resume=True)
            self.assertTrue(result['completed'])
            self.assertEqual(result['model'].device.type,'cuda')
            self.assertEqual(full['history'],result['history'])
            for key,value in full['model'].state_dict().items():
                self.assertTrue(torch.equal(value,result['model'].state_dict()[key]),key)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_cpu_batched_replay_outputs_and_gradients_match(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.meta_exploration import decode_rule
        from mamrl.meta_batch import collect_batch,replay_batch
        cpu=ActionSetActorCritic(hidden_size=8,seed=12)
        gpu=copy.deepcopy(cpu).to('cuda')
        rules=[decode_rule(r['rule']) for r in self.manifest['rules'][:2]]
        trajectories=collect_batch(cpu,[(rules[0],55,'steady'),(rules[1],66,'rapid')],steps=4)
        a=replay_batch(cpu,trajectories);b=replay_batch(gpu,trajectories)
        for x,y in zip(a,b):self.assertTrue(torch.allclose(x,y.cpu(),atol=2e-6,rtol=2e-5))
        ac=torch.autograd.grad(sum(x.sum() for x in a),cpu.gru.weight_ih)[0]
        ag=torch.autograd.grad(sum(x.sum() for x in b),gpu.gru.weight_ih)[0]
        self.assertTrue(torch.allclose(ac,ag.cpu(),atol=2e-5,rtol=2e-4))

@unittest.skipUnless(torch is not None, 'optional PyTorch')
class RunnerResumeContract(unittest.TestCase):
    def test_colab_notebook_cells_compile(self):
        import json
        notebook=json.loads(Path('notebooks/MAMRL_GPU.ipynb').read_text())
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),f'notebook cell {i}','exec')

    def test_cli_backup_restore_resume_matches_uninterrupted(self):
        import os,subprocess,sys,shutil,json
        from mamrl.experiment_artifacts import load_meta_experiment
        from mamrl.recurrent import load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);orig=root/'original';backup=root/'backup';restored=root/'restored';full=root/'full'
            def invoke(out,*extra):
                cmd=[sys.executable,'-B','-m','mamrl.meta_exploration','--output-dir',str(out),
                     '--seeds','7','--updates','2','--trials-per-update','2','--steps','2','--hidden-size','8',
                     '--reps','1','--device','auto','--batch-size','2','--assessment-steps','5',*extra]
                return subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                      capture_output=True,text=True,timeout=180)
            paused=invoke(orig,'--stop-after-update','1','--backup-dir',str(backup))
            self.assertEqual(paused.returncode,0,paused.stderr)
            self.assertTrue((backup/'seed7'/'training_state.pt').is_file())
            self.assertFalse((backup/'experiment.json').exists())
            reject=invoke(root/'another','--backup-dir',str(backup))
            self.assertNotEqual(reject.returncode,0)
            shutil.copytree(backup,restored)
            changed=invoke(restored,'--resume','--batch-size','1')
            self.assertNotEqual(changed.returncode,0)
            completed=invoke(restored,'--resume')
            self.assertEqual(completed.returncode,0,completed.stderr)
            uninterrupted=invoke(full)
            self.assertEqual(uninterrupted.returncode,0,uninterrupted.stderr)
            a=load_meta_experiment(restored/'experiment.json');b=load_meta_experiment(full/'experiment.json')
            self.assertEqual(a['runs'][0]['history'],b['runs'][0]['history'])
            self.assertEqual(a['runs'][0]['evaluation'],b['runs'][0]['evaluation'])
            ma=load_checkpoint(restored/'seed7'/'policy.pt');mb=load_checkpoint(full/'seed7'/'policy.pt')
            for k,v in ma.state_dict().items():self.assertTrue(torch.equal(v,mb.state_dict()[k]),k)
            self.assertEqual(a['protocol']['device'],'cuda' if torch.cuda.is_available() else 'cpu')
            self.assertEqual(a['protocol']['assessment_steps'],5)
            for splits in a['runs'][0]['evaluation'].values():
                for evaluation in splits.values():
                    for trial in evaluation['trials']:
                        self.assertEqual(trial['phase_steps'],{'exploration':2,'assessment':5,'erased_assessment':5})
                        self.assertEqual(trial['environment_steps_with_control'],12)
            self.assertIn('탐색 2회 + 실제 평가 5회 + 경험 제거 대조 5회',(restored/'report.html').read_text())
