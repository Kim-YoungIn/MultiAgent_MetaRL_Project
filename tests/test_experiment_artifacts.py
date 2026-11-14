"""Independent artifact contracts for compact and legacy meta-experiment runs."""

import copy
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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fixture():
    # Distinct sentinels expose any lost detail or accidental copying into the index.
    baseline = {'random': {'fresh_rules': {'summary': {'assessment_mean': 0.25},
                                          'trials': [{'task_id': 'sentinel-baseline', 'reward': 0.25}]}}}
    row = {'seed': 7, 'config': {'seed': 7}, 'history': [{'step': 1}],
           'checkpoint': 'seed7/policy.pt', 'initial_checkpoint': 'seed7/initial.pt',
           'evaluation': {'trained': {'fresh_rules': {'summary': {'assessment_mean': 0.75},
                                                        'trials': [{'task_id': 'sentinel-trained', 'reward': 0.75}]}},
                          'initial': {'fresh_rules': {'summary': {'assessment_mean': 0.5},
                                                        'trials': [{'task_id': 'sentinel-initial', 'reward': 0.5}]}}}}
    return {'version': 2, 'protocol': {'version': 2, 'seeds': [7]}, 'baselines': baseline,
            'runs': [row], 'comparisons': {'fresh_rules': {'difference': 0.5}}, 'rule_catalog': []}


class ArtifactContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = fixture()
        write_json(self.root / 'baselines.json', self.record['baselines'])
        write_json(self.root / 'seed7' / 'evaluation.json', self.record['runs'][0])

    def save(self):
        from mamrl.experiment_artifacts import save_meta_experiment
        save_meta_experiment(self.root / 'experiment.json', self.record)
        return json.loads((self.root / 'experiment.json').read_text(encoding='utf-8'))

    def load(self):
        from mamrl.experiment_artifacts import load_meta_experiment
        return load_meta_experiment(self.root / 'experiment.json')

    def test_legacy_embedded_record_is_loaded_unchanged(self):
        write_json(self.root / 'experiment.json', self.record)
        self.assertEqual(self.load(), self.record)

    def test_compact_index_roundtrips_all_details_once(self):
        index = self.save()
        self.assertEqual(index['version'], 3)
        text = (self.root / 'experiment.json').read_text(encoding='utf-8')
        for detail in ('sentinel-baseline', 'sentinel-trained', 'sentinel-initial'):
            self.assertNotIn(detail, text)
        loaded = self.load()
        self.assertEqual(loaded['baselines'], self.record['baselines'])
        self.assertEqual(loaded['runs'], self.record['runs'])
        self.assertEqual(loaded['comparisons'], self.record['comparisons'])
        self.assertEqual(loaded['protocol'], self.record['protocol'])
        self.assertEqual(loaded['rule_catalog'], self.record['rule_catalog'])

    def test_missing_or_modified_reference_is_rejected(self):
        self.save()
        baseline_path = self.root / 'baselines.json'
        original = baseline_path.read_bytes()
        baseline_path.unlink()
        with self.assertRaises((ValueError, OSError)):
            self.load()
        baseline_path.write_bytes(original)
        baseline = json.loads(original)
        baseline['random']['fresh_rules']['trials'][0]['reward'] = 999
        write_json(baseline_path, baseline)
        with self.assertRaises((ValueError, OSError)):
            self.load()

    def test_index_summary_tampering_is_rejected(self):
        index = self.save()
        index['baselines']['random']['fresh_rules']['summary']['assessment_mean'] = -3
        write_json(self.root / 'experiment.json', index)
        with self.assertRaises((ValueError, OSError)):
            self.load()

    def test_escaping_or_absolute_reference_is_rejected(self):
        original = self.save()
        index_path = self.root / 'experiment.json'
        for bad_path in ('../outside.json', '/tmp/outside.json'):
            altered = copy.deepcopy(original)
            altered['references']['baselines']['path'] = bad_path
            write_json(index_path, altered)
            with self.subTest(bad_path=bad_path):
                with self.assertRaises((ValueError, OSError)):
                    self.load()

    def test_symlink_reference_cannot_escape_run_directory(self):
        original = self.save()
        # Point to a real file containing exact valid bytes outside this run.
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / 'baseline.json'
            outside.write_bytes((self.root / 'baselines.json').read_bytes())
            (self.root / 'linked.json').symlink_to(outside)
            original['references']['baselines']['path'] = 'linked.json'
            write_json(self.root / 'experiment.json', original)
            with self.assertRaises((ValueError, OSError)):
                self.load()

    def test_save_rejects_detail_mismatch_and_does_not_overwrite(self):
        from mamrl.experiment_artifacts import save_meta_experiment
        path = self.root / 'experiment.json'
        wrong = copy.deepcopy(self.record)
        wrong['runs'][0]['seed'] = 8
        with self.assertRaises((ValueError, OSError)):
            save_meta_experiment(path, wrong)
        self.assertFalse(path.exists())
        self.save()
        previous = path.read_bytes()
        with self.assertRaises((ValueError, OSError)):
            save_meta_experiment(path, self.record)
        self.assertEqual(path.read_bytes(), previous)

    def test_seed_reference_modification_is_rejected(self):
        self.save()
        row = copy.deepcopy(self.record['runs'][0])
        row['evaluation']['trained']['fresh_rules']['trials'][0]['reward'] = -17
        write_json(self.root / 'seed7' / 'evaluation.json', row)
        with self.assertRaises((ValueError, OSError)):
            self.load()


@unittest.skipUnless(torch is not None, 'optional torch dependency')
class TinyMetaArtifactContract(unittest.TestCase):
    def test_compact_cli_load_preserves_checkpoint_evaluation_and_report(self):
        from mamrl.experiment_artifacts import load_meta_experiment
        from mamrl.meta_exploration import evaluate_meta, meta_tasks, neural_factory, render_meta_report
        from mamrl.recurrent import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            command = [sys.executable, '-B', '-m', 'mamrl.meta_exploration', '--output-dir', str(root),
                       '--seeds', '7', '--updates', '1', '--trials-per-update', '1',
                       '--steps', '2', '--hidden-size', '4', '--reps', '1']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr[-5000:])
            index = json.loads((root / 'experiment.json').read_text())
            self.assertEqual(index['version'], 3)
            for splits in index['baselines'].values():
                for evaluation in splits.values():
                    self.assertNotIn('trials', evaluation)
            for run in index['runs']:
                for splits in run['evaluation'].values():
                    for evaluation in splits.values():
                        self.assertNotIn('trials', evaluation)
            loaded = load_meta_experiment(root / 'experiment.json')
            self.assertEqual(loaded['baselines'], json.loads((root / 'baselines.json').read_text()))
            row = json.loads((root / 'seed7' / 'evaluation.json').read_text())
            self.assertEqual(loaded['runs'], [row])
            model = load_checkpoint(root / row['checkpoint'])
            manifest = json.loads((root / 'manifest.json').read_text())
            reproduced = evaluate_meta(neural_factory(model), meta_tasks(manifest, 'fresh_rules'), steps=2, reps=1)
            self.assertEqual(reproduced, row['evaluation']['trained']['fresh_rules'])
            self.assertEqual(render_meta_report(loaded), (root / 'report.html').read_text())
