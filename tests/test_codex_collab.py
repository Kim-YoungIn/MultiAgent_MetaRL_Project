"""Unit tests for Lead (Antigravity) <-> Codex Collaboration Setup (.codex/ + handoff/ + scripts/)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCodexCollaborationEnv(unittest.TestCase):
    def test_codex_toml_configs_and_agents_are_valid(self) -> None:
        handoff = _load_script("codex_handoff", "scripts/codex_handoff.py")
        errors = handoff.validate_codex_configs(ROOT)
        self.assertEqual(errors, [], f"Codex config validation errors: {errors}")

        agents_dir = ROOT / ".codex" / "agents"
        roles = {p.stem for p in agents_dir.glob("*.toml")}
        self.assertTrue({"research_critic", "results_analyst", "validator", "experiment_worker"}.issubset(roles))
        self.assertNotIn("paper_reviewer", roles)

        t3 = handoff.parse_task_brief(handoff.find_task_file("TASK-003", ROOT))
        self.assertEqual(t3["role"], "results_analyst")
        self.assertEqual(t3["allowed_paths"], ["handoff/reports/TASK-003_legacy_bridge_and_insights.md"])

    def test_starter_task_briefs_parse_and_scope_checker_enforces_boundaries(self) -> None:
        handoff = _load_script("codex_handoff", "scripts/codex_handoff.py")
        t1 = handoff.parse_task_brief(handoff.find_task_file("TASK-001", ROOT))
        self.assertEqual(t1["role"], "research_critic")
        self.assertEqual(t1["allowed_paths"], ["handoff/reviews/20261127_exp1_2a_2b_audit.md"])

        prompt = handoff.build_codex_prompt(t1, ROOT)
        self.assertIn("AGENTS.md", prompt)
        self.assertIn("handoff/BOARD.md", prompt)
        self.assertIn("TASK-001", prompt)

        in_s, out_s = handoff.check_scope_against_paths(
            ["handoff/reviews/20261127_exp1_2a_2b_audit.md", "docs/SPEC.md", "src/mamrl/study_v3.py"],
            t1["allowed_paths"],
        )
        self.assertEqual(in_s, ["handoff/reviews/20261127_exp1_2a_2b_audit.md"])
        self.assertEqual(out_s, ["docs/SPEC.md", "src/mamrl/study_v3.py"])

    def test_create_task_brief_and_pack_context(self) -> None:
        handoff = _load_script("codex_handoff", "scripts/codex_handoff.py")
        packer = _load_script("pack_context", "scripts/pack_context.py")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(ROOT / ".codex", tmp_root / ".codex")
            brief = handoff.create_task_brief(
                task_id="TASK-999",
                role="experiment_worker",
                title="Test scoped worker task",
                allowed_paths=["src/mamrl/agents/demo.py", "handoff/reports/TASK-999_report.md"],
                root=tmp_root,
            )
            info = handoff.parse_task_brief(brief)
            self.assertEqual(info["task_id"], "TASK-999")
            self.assertEqual(info["role"], "experiment_worker")
            self.assertIn("src/mamrl/agents/demo.py", info["allowed_paths"])
            self.assertIn("handoff/reports/TASK-999_report.md", info["allowed_paths"])

        pack = packer.build_pack()
        self.assertIn("handoff/BOARD.md", pack)
        self.assertIn("results/core_v3_sealed/tables/methods.csv", pack)
        self.assertIn("results/exp2a_online_opp/tables/methods.csv", pack)
        self.assertIn("results/exp2b_symmetric_ma/tables/matchups.csv", pack)


if __name__ == "__main__":
    unittest.main()
