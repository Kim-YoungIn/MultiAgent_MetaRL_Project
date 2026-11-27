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
        self.assertIn("results/exp3a_reliability_sweep/tables/r3_obs_reliability.csv", pack)
        self.assertIn("results/exp3b_equivariant_semantic/tables/matchups.csv", pack)
        self.assertIn("results/exp4_multibarrier_aux_discovery/tables/methods.csv", pack)

    def test_claude_collaborator_entry_and_prompts(self) -> None:
        handoff = _load_script("codex_handoff", "scripts/codex_handoff.py")
        packer = _load_script("pack_context", "scripts/pack_context.py")
        self.assertTrue((ROOT / "CLAUDE.md").exists())
        self.assertTrue((ROOT / "GEMINI.md").exists())
        self.assertIn("AGENTS.md", (ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
        self.assertEqual(set(handoff.CLAUDE_ROLES.keys()), {"claude_architect", "claude_analyst"})
        self.assertNotIn("claude_writer", handoff.CLAUDE_ROLES)

        slim = packer.build_pack(slim=True)
        self.assertIn("PROGRESS.md", slim)
        self.assertIn("INTERIM_SYNTHESIS_EXP1_TO_4.md", slim)
        self.assertLess(len(slim.encode("utf-8")), 25000)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            shutil.copytree(ROOT / ".codex", tmp_root / ".codex")
            brief = handoff.create_task_brief(
                task_id="TASK-998",
                role="claude_analyst",
                title="Test claude interim review",
                allowed_paths=["handoff/reports/TASK-998_claude_review.md"],
                root=tmp_root,
            )
            info = handoff.parse_task_brief(brief)
            self.assertEqual(info["role"], "claude_analyst")
            prompt = handoff.build_codex_prompt(info, tmp_root, output_via_flag=True)
            self.assertIn("Antigravity-Claude", prompt)
            self.assertIn("CLAUDE.md", prompt)
            self.assertIn("context_pack_slim.md", prompt)
            self.assertIn("TASK-998", prompt)

            agy_cmd = handoff.build_agy_command("/usr/bin/agy", info, prompt, model="claude-sonnet-4-6")
            self.assertEqual(agy_cmd[0], "/usr/bin/agy")
            self.assertIn("--sandbox", agy_cmd)
            self.assertIn("--disable-slash-commands", agy_cmd)
            self.assertIn("plan", agy_cmd)
            self.assertIn("claude-sonnet-4-6", agy_cmd)
            self.assertEqual(agy_cmd[-2], "-p")


if __name__ == "__main__":
    unittest.main()
