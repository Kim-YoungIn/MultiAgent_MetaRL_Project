#!/usr/bin/env python3
"""Generate a compact single-file context bundle for GPT / external review agents.

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

KEY_DOCS = [
    "AGENTS.md",
    "PROGRESS.md",
    "docs/SPEC.md",
    "docs/PRIOR_FINDINGS.md",
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


def build_pack() -> str:
    sections = [
        "# MAMRL Context Pack (For External Review / GPT Collaboration)",
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

    # Include compact JSON summaries under results/ if present
    json_results = sorted((ROOT / "results").rglob("*.json")) if (ROOT / "results").exists() else []
    if json_results:
        sections.append("---\n\n## Latest Tracked Result Summaries\n")
        for jp in json_results[:5]:
            rel = jp.relative_to(ROOT)
            text = jp.read_text(encoding="utf-8")
            if len(text) > 4000:
                text = text[:4000] + "\n... [truncated]"
            sections.append(f"### `{rel}`\n```json\n{text}\n```\n")

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack active project context into one markdown file.")
    parser.add_argument("--output", default="handoff/context_pack.md", help="Output markdown path.")
    args = parser.parse_args()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    content = build_pack()
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out} ({len(content.encode('utf-8')) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
