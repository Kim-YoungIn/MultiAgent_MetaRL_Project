#!/usr/bin/env python3
"""Generate a compact single-file context bundle for Codex / GPT review & collaboration.

Usage:
    .venv/bin/python scripts/pack_context.py
    .venv/bin/python scripts/pack_context.py --output handoff/context_pack.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

SLIM_DOCS = [
    "PROGRESS.md",
    "handoff/BOARD.md",
    "handoff/reports/INTERIM_SYNTHESIS_EXP1_TO_4.md",
]

KEY_DOCS = [
    "AGENTS.md",
    "PROGRESS.md",
    "handoff/BOARD.md",
    "docs/SPEC.md",
    "docs/PRIOR_FINDINGS.md",
    "docs/COLLABORATION.md",
]

KEY_RESULT_TABLES = [
    "results/gate0_exact_dp.json",
    "results/core_v3_sealed/tables/methods.csv",
    "results/exp2a_online_opp/tables/methods.csv",
    "results/exp2b_symmetric_ma/tables/matchups.csv",
    "results/exp3a_reliability_sweep/tables/r3_obs_reliability.csv",
    "results/exp3a_reliability_sweep/tables/coop_trap_dynamics.csv",
    "results/exp3b_equivariant_semantic/tables/matchups.csv",
    "results/exp4_multibarrier_aux_discovery/tables/methods.csv",
]


def _git_info() -> str:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return f"branch `{branch}` @ `{commit}`"
    except Exception:
        return "git info unavailable"


def _repo_tree() -> str:
    skip = {".git", ".venv", "runs", "__pycache__", ".pytest_cache"}
    lines: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in skip for part in rel.parts):
            continue
        if rel == Path("handoff/context_pack.md"):
            continue
        marker = "/" if path.is_dir() else ""
        lines.append(f"{rel}{marker}")
    return "\n".join(lines)


def build_pack(slim: bool = False) -> str:
    if slim:
        sections = [
            "# MAMRL Slim Context Pack (Token-Optimized Routine Dispatch)",
            f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"- Repo state: {_git_info()}",
            "- Full pack: `handoff/context_pack.md` · Compact CSVs: `results/*/tables/*.csv` (skip `runs/` & full `summary.json`).",
            "",
        ]
        for rel in SLIM_DOCS:
            p = ROOT / rel
            if p.exists():
                sections.append(f"---\n\n## File: `{rel}`\n")
                sections.append(p.read_text(encoding="utf-8").strip())
                sections.append("")
        return "\n".join(sections)

    sections = [
        "# MAMRL Context Pack (Antigravity-Gemini Lead | Antigravity-Claude & Codex CLI Collaborators)",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Repo state: {_git_info()}",
        "- Legacy archive tag: `archive/v0.9-legacy-studies`",
        "",
        "## Repository Tree (Tracked Active Core)",
        "```text",
        _repo_tree(),
        "```",
        "",
    ]

    for rel in KEY_DOCS:
        p = ROOT / rel
        if p.exists():
            sections.append(f"---\n\n## File: `{rel}`\n")
            sections.append(p.read_text(encoding="utf-8").strip())
            sections.append("")

    present_tables = [ROOT / rel for rel in KEY_RESULT_TABLES if (ROOT / rel).exists()]
    if present_tables:
        sections.append("---\n\n## Sealed 5-Seed Result Tables (Exp 1 / 2A / 2B / 3A / 3B / 4)\n")
        for tp in present_tables:
            rel = tp.relative_to(ROOT)
            lang = "json" if tp.suffix == ".json" else "csv"
            text = tp.read_text(encoding="utf-8").strip()
            if len(text) > 5500:
                text = text[:5500] + "\n... [truncated; read full file in workspace]"
            sections.append(f"### `{rel}`\n```{lang}\n{text}\n```\n")

    sections.extend(
        [
            "---",
            "",
            "## Collaboration & Review Protocol",
            "- **Lead Agent:** `Antigravity-Gemini`. Collaborators (`Antigravity-Claude`, `Codex CLI`) must check `handoff/BOARD.md` and their assigned `handoff/tasks/TASK-*.md` brief before acting.",
            "- **Audit Focus (Interim Checkpoint):** Exp 1 (`R0/R1/R3`), Exp 2A (online Q-learning / unseen Expected SARSA), Exp 2B (symmetric 4-action `comp_rps` & `coop_trap`), Exp 3A ($d \\in \\{0, 0.3, 0.6\\}$ reliability stress sweep), Exp 3B (exact $S_4$ `EquivariantDRQN` + 4 semantic families + unseen 5th family + D-031), and Exp 4 (3-layer multi-barrier `T=64` + bilevel auxiliary task discovery) are all sealed across 5 seeds.",
            "- Return concrete, unbiased critiques or verified deliverables to the exact assigned path under `handoff/reviews/` or `handoff/reports/`.",
            "",
        ]
    )

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack active project context into one markdown file.")
    parser.add_argument("--output", default="", help="Output markdown path (defaults to handoff/context_pack[_slim].md).")
    parser.add_argument("--slim", action="store_true", help="Write a compact ~12 KB context pack for token-efficient dispatches.")
    args = parser.parse_args()

    out_rel = args.output or ("handoff/context_pack_slim.md" if args.slim else "handoff/context_pack.md")
    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    content = build_pack(slim=args.slim)
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out} ({len(content.encode('utf-8')) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
