import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.project_control import __main__ as control


class ProjectControlTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = {
            "id": "example",
            "stage": "completed",
            "scope": "Example",
            "protocol": "protocol.md",
            "report": "report.md",
            "evidence": "evidence.json",
            "raw": "runs/example",
            "reproduce": "protocol.md",
            "verification": "Recorded in evidence.",
        }
        for name in ("protocol.md", "report.md", "evidence.json"):
            (self.root / name).touch()
        self.registry = self.root / "registry.json"

    def run_check(self):
        self.registry.write_text(json.dumps({"schema_version": 1, "studies": [self.study]}))
        output = io.StringIO()
        with patch.object(control, "ROOT", self.root), contextlib.redirect_stdout(output):
            result = control.main(["--registry", str(self.registry), "--check"])
        return result, output.getvalue()

    def test_raw_absence_is_visible_but_not_a_missing_versioned_file(self):
        result, output = self.run_check()
        self.assertEqual(result, 0)
        self.assertIn("raw absent", output)
        self.assertIn("Full artifact verification", output)

    def test_missing_completed_report_fails_check(self):
        (self.root / "report.md").unlink()
        result, output = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn("Missing versioned paths: report", output)

    def test_in_progress_paths_can_be_pending(self):
        self.study["stage"] = "in_progress"
        (self.root / "report.md").unlink()
        self.assertEqual(self.run_check()[0], 0)

    def test_registry_rejects_paths_outside_project(self):
        self.study["raw"] = "../elsewhere"
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self.run_check()

    def test_local_binding_detects_corruption_and_absence(self):
        raw = self.root / 'runs/example'
        raw.mkdir(parents=True)
        versioned = self.root / 'sealed.json'
        versioned.write_text('{"sealed": true}')
        local = raw / 'manifest.json'
        local.write_bytes(versioned.read_bytes())
        self.study['bindings'] = [{'local': 'runs/example/manifest.json', 'evidence': 'sealed.json'}]
        result, output = self.run_check()
        self.assertEqual(result, 0)
        self.assertIn('1 binding(s) match', output)
        local.write_text('corrupted')
        self.assertEqual(self.run_check()[0], 1)
        local.unlink()
        self.assertEqual(self.run_check()[0], 1)
        raw.rmdir()
        result, output = self.run_check()
        self.assertEqual(result, 0)
        self.assertIn('raw absent', output)

    def test_evidence_record_digest_binding(self):
        raw = self.root / 'runs/example'
        raw.mkdir(parents=True)
        local = raw / 'manifest.json'
        local.write_text('example')
        evidence = self.root / 'sealed.json'
        evidence.write_text(json.dumps({'sha256': hashlib.sha256(local.read_bytes()).hexdigest()}))
        self.study['bindings'] = [{'local': 'runs/example/manifest.json', 'evidence': 'sealed.json', 'field': 'sha256'}]
        self.assertEqual(self.run_check()[0], 0)
        evidence.write_text(json.dumps({'sha256': 'wrong'}))
        self.assertEqual(self.run_check()[0], 1)


if __name__ == "__main__":
    unittest.main()
