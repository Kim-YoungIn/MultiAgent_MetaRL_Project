"""Contract checks for downstream objectives and history/opponent crossing."""
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'optional torch dependency')
class ObjectiveReferenceContract(unittest.TestCase):
    def test_targets_are_downstream_only_and_default_is_retained_gain(self):
        from mamrl.meta_exploration import ExplorationRollout, MetaConfig, meta_targets
        # Assessment 3, erased 5: assessment and gain deliberately have opposite signs.
        r = ExplorationRollout((None, None), (0, 1), (0., 0.), (0., 0.), -2.,
                               {'assessment_mean': 3., 'erased_mean': 5., 'exploration_mean': 1000.})
        self.assertEqual(MetaConfig().objective, 'retained_gain')
        self.assertEqual(meta_targets([r]).tolist(), [-2., -2.])
        self.assertEqual(meta_targets([r], 'assessment_return').tolist(), [3., 3.])
        changed = dataclasses.replace(r, trial={**r.trial, 'exploration_mean': -1000.})
        self.assertTrue(torch.equal(meta_targets([r], 'assessment_return'), meta_targets([changed], 'assessment_return')))
        with self.assertRaises(ValueError): MetaConfig(objective='invalid')
        with self.assertRaises(ValueError): meta_targets([r], 'invalid')

    def test_training_logs_target_and_resume_rejects_changed_objective(self):
        from mamrl.meta_exploration import MetaConfig, train_meta
        from mamrl.probe_templates import build_probe_manifest
        manifest = build_probe_manifest(train_rules=2, fresh_rules=2, structural_rules=2)
        config = MetaConfig(updates=2, trials_per_update=1, steps=2, assessment_steps=2,
                            hidden_size=4, epochs=1, objective='assessment_return')
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'state.pt'
            result = train_meta(config, manifest, checkpoint_path=checkpoint, stop_after_update=1)
            row = result['history'][0]
            self.assertAlmostEqual(row['mean_teacher_target'], row['mean_assessment'], places=6)
            with self.assertRaises(ValueError):
                train_meta(dataclasses.replace(config, objective='retained_gain'), manifest,
                           checkpoint_path=checkpoint, resume=True)
            resumed = train_meta(config, manifest, checkpoint_path=checkpoint, resume=True)
            self.assertEqual(len(resumed['history']), 2)
            baseline = train_meta(dataclasses.replace(config, updates=1, objective='retained_gain'), manifest)
            row = baseline['history'][0]
            self.assertAlmostEqual(row['mean_teacher_target'], row['mean_meta_reward'], places=6)

    def test_cli_assessment_objective_record_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            cmd = [sys.executable, '-B', '-m', 'mamrl.meta_exploration', '--output-dir', str(output),
                   '--objective', 'assessment_return', '--catalog', 'probe_template',
                   '--train-rules', '2', '--fresh-rules', '2', '--structural-rules', '2',
                   '--seeds', '7', '--updates', '1', '--trials-per-update', '1', '--steps', '2',
                   '--assessment-steps', '2', '--hidden-size', '4', '--reps', '1']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            protocol = json.loads((output/'protocol.json').read_text())
            self.assertEqual(protocol['objective'], 'assessment_return')
            self.assertIn('후속 평균 보상', (output/'report.html').read_text())

    def test_paired_objective_study_budget_and_resume(self):
        from mamrl.objective_study import run_objective_study
        from mamrl.probe_templates import build_probe_manifest
        from mamrl.recurrent import load_checkpoint
        manifest = build_probe_manifest(train_rules=2, fresh_rules=2, structural_rules=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'study'
            args = dict(output_dir=root, seeds=(7,), updates=2, checkpoints=(1,2),
                        trials_per_update=1, steps=2, assessment_steps=2, hidden_size=4,
                        batch_size=1, reps=1)
            record = run_objective_study(manifest, **args)
            self.assertEqual(record['protocol']['training_simulation_steps'], 2*1*2*1*(2+2*2))
            paths = [root/name/'seed7' for name in ('retained_gain','assessment_return')]
            models = [load_checkpoint(path/'initial.pt') for path in paths]
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, models[1].state_dict()[key]), key)
            histories = [json.loads((path/'history.json').read_text()) for path in paths]
            self.assertEqual([len(x) for x in histories], [2,2])
            for left, right in zip(*histories):
                fields = ('rule_id','opponent_profile','seed')
                self.assertEqual([tuple(t[k] for k in fields) for t in left['trials']],
                                 [tuple(t[k] for k in fields) for t in right['trials']])
            for update in (0,1,2):
                evaluations = [json.loads((path/f'update{update}.json').read_text())['evaluation'] for path in paths]
                self.assertEqual(evaluations[0].keys(), evaluations[1].keys())
                for split in evaluations[0]:
                    left, right = [evaluation[split]['rows'] for evaluation in evaluations]
                    fields = ('task_id','rule_id','rep','seed','opponent_profile')
                    self.assertEqual([tuple(row[k] for k in fields) for row in left],
                                     [tuple(row[k] for k in fields) for row in right])
                    for row in left + right:
                        self.assertFalse({'exploration','assessment','erased_assessment','events'} & row.keys())
                if update == 0:
                    self.assertEqual(evaluations[0], evaluations[1])
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            resumed = run_objective_study(manifest, resume=True, **args)
            after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(record, resumed)
            self.assertEqual(before, after)

    def test_cross_reference_cost_frozen_and_provenance(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.recurrent import save_checkpoint
        from mamrl.probe_templates import build_probe_manifest
        from mamrl.reference_study import run_reference_study
        manifest = build_probe_manifest(train_rules=2, fresh_rules=2, structural_rules=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); initial = root/'initial.pt'; learned = root/'learned.pt'
            save_checkpoint(ActionSetActorCritic(hidden_size=4, seed=3), initial)
            save_checkpoint(ActionSetActorCritic(hidden_size=4, seed=4), learned)
            result = run_reference_study(manifest, initial, learned, root/'out', steps=2,
                                         assessment_steps=2, reps=1, seed=9)
            protocol = result['protocol']; rows = result['rows']
            self.assertEqual(len(rows), protocol['tasks'] * 18)
            self.assertEqual(protocol['simulation_steps'], protocol['tasks'] * (3*(2+2*2)+18*2))
            self.assertEqual(protocol['checkpoint_sha256']['initial'], hashlib.sha256(initial.read_bytes()).hexdigest())
            self.assertEqual(json.loads((root/'out'/'study.json').read_text()), result)
            for task in {r['task_id'] for r in rows}:
                group = [r for r in rows if r['task_id'] == task]
                self.assertEqual(len({(r['history'],r['reference'],r['mode']) for r in group}), 18)
                for r in group:
                    self.assertEqual(r['opponent_after']['updates'] - r['opponent_before']['updates'], 2)
                    self.assertEqual(r['student_after']['updates'] - r['student_before']['updates'],
                                     0 if r['mode'] == 'frozen' else 2)
                    if r['mode'] == 'frozen':
                        self.assertEqual(r['student_before']['entries'], r['student_after']['entries'])
                        self.assertNotEqual(r['student_before']['rng_digest'], r['student_after']['rng_digest'])
                    if r['mode'] == 'online' and r['history'] == r['reference']:
                        self.assertAlmostEqual(r['assessment_mean'],r['natural_assessment_mean'])
                for history in ('random','initial','learned'):
                    sub = [r for r in group if r['history'] == history]
                    self.assertEqual(len({json.dumps(r['student_before'],sort_keys=True) for r in sub}), 1)
                    self.assertEqual(len({r['history_sha256'] for r in sub}), 1)
                for reference in ('random','initial','learned'):
                    sub = [r for r in group if r['reference'] == reference]
                    self.assertEqual(len({json.dumps(r['opponent_before'],sort_keys=True) for r in sub}), 1)

if __name__ == '__main__':
    unittest.main()
