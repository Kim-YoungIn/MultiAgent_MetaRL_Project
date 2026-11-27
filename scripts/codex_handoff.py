#!/usr/bin/env python3
"""Lead (Antigravity) <-> Codex Collaboration CLI Helper.

Manages scoped task briefs, TOML validation for .codex/, copy-paste / CLI prompts,
and automated git-status scope verification so Codex agents never collide with Lead files.

Usage:
    .venv/bin/python scripts/codex_handoff.py validate-codex
    .venv/bin/python scripts/codex_handoff.py status
    .venv/bin/python scripts/codex_handoff.py prompt --id TASK-001
    .venv/bin/python scripts/codex_handoff.py check-scope --id TASK-001
    .venv/bin/python scripts/codex_handoff.py new-task --id TASK-003 --role experiment_worker \
        --title "Auxiliary task discovery" --allow "src/mamrl/agents/aux_discovery.py,handoff/reports/TASK-003_report.md"
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
CODEX_DIR = ROOT / ".codex"
TASKS_DIR = ROOT / "handoff" / "tasks"
REPORTS_DIR = ROOT / "handoff" / "reports"
REVIEWS_DIR = ROOT / "handoff" / "reviews"

REQUIRED_AGENT_KEYS = ("name", "description", "model", "model_reasoning_effort", "developer_instructions")


def validate_codex_configs(root: Path = ROOT) -> list[str]:
    """Validate .codex/config.toml and all .codex/agents/*.toml; return list of error strings (empty = OK)."""
    errors: list[str] = []
    cfg_path = root / ".codex" / "config.toml"
    if not cfg_path.exists():
        return [f"Missing {cfg_path}"]
    try:
        cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Invalid TOML in {cfg_path}: {exc}"]

    if "model" not in cfg:
        errors.append(f"{cfg_path}: missing top-level 'model'")
    if "agents" not in cfg or not isinstance(cfg["agents"], dict):
        errors.append(f"{cfg_path}: missing [agents] table")

    agents_dir = root / ".codex" / "agents"
    if not agents_dir.exists():
        errors.append(f"Missing {agents_dir}")
        return errors

    agent_files = sorted(agents_dir.glob("*.toml"))
    if not agent_files:
        errors.append(f"No agent definitions found in {agents_dir}")
    for af in agent_files:
        try:
            data = tomllib.loads(af.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid TOML in {af}: {exc}")
            continue
        for key in REQUIRED_AGENT_KEYS:
            if key not in data or not str(data[key]).strip():
                errors.append(f"{af.name}: missing or empty required key '{key}'")
        if data.get("name") and data["name"] != af.stem:
            errors.append(f"{af.name}: name '{data['name']}' does not match filename stem '{af.stem}'")
    return errors


def find_task_file(task_id: str, root: Path = ROOT) -> Path:
    tasks_dir = root / "handoff" / "tasks"
    matches = sorted(tasks_dir.glob(f"{task_id}_*.md")) + sorted(tasks_dir.glob(f"{task_id}.md"))
    if not matches:
        raise FileNotFoundError(f"No task brief matching '{task_id}' in {tasks_dir}")
    return matches[0]


def parse_task_brief(task_path: Path) -> dict:
    """Extract metadata and Allowed Write Paths from a standardized TASK-*.md brief."""
    text = task_path.read_text(encoding="utf-8")
    role_m = re.search(r"\*\*Assigned Role:\*\*\s*`([^`]+)`", text)
    status_m = re.search(r"\*\*Status:\*\*\s*`([^`]+)`", text)
    id_m = re.search(r"\*\*Task ID:\*\*\s*`([^`]+)`", text)
    title_m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)

    allowed: list[str] = []
    in_allow = False
    for line in text.splitlines():
        if line.strip().startswith("## Allowed Write Paths"):
            in_allow = True
            continue
        if in_allow:
            if line.strip().startswith("## "):
                break
            m = re.match(r"^\s*-\s+`([^`]+)`", line)
            if m:
                allowed.append(m.group(1).strip())

    return {
        "path": task_path,
        "task_id": id_m.group(1) if id_m else task_path.stem.split("_")[0],
        "title": title_m.group(1).strip() if title_m else task_path.stem,
        "role": role_m.group(1) if role_m else "unknown",
        "status": status_m.group(1) if status_m else "UNKNOWN",
        "allowed_paths": allowed,
    }


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:40] or "task"


def create_task_brief(
    *,
    task_id: str,
    role: str,
    title: str,
    allowed_paths: list[str],
    objective: str = "",
    root: Path = ROOT,
) -> Path:
    agents_dir = root / ".codex" / "agents"
    valid_roles = {p.stem for p in agents_dir.glob("*.toml")}
    if valid_roles and role not in valid_roles:
        raise ValueError(f"Unknown role '{role}'. Available roles: {sorted(valid_roles)}")

    tasks_dir = root / "handoff" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    out_path = tasks_dir / f"{task_id}_{slug}.md"
    if out_path.exists():
        raise FileExistsError(f"Task brief already exists: {out_path}")

    allow_block = "\n".join(f"- `{p}`" for p in allowed_paths)
    report_path = next(
        (p for p in allowed_paths if p.startswith(("handoff/reports/", "handoff/reviews/"))),
        f"handoff/reports/{task_id}_{slug}_report.md",
    )
    if report_path not in allowed_paths:
        allow_block = (allow_block + f"\n- `{report_path}`").strip()

    content = f"""# {task_id}: {title}

- **Task ID:** `{task_id}`
- **Assigned Role:** `{role}` (`.codex/agents/{role}.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
{objective or title}

## Required Reading
- `AGENTS.md`, `PROGRESS.md`, `docs/SPEC.md`, `handoff/BOARD.md`

## Allowed Write Paths
{allow_block}

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, sealed `results/`, and any unassigned `src/`/`configs/`/`tests/` paths).

## Verification & Expected Report Output
1. Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` if any code or test was edited.
2. Write a structured report (changed files, verification outputs, key findings, open proposals) to `{report_path}`.
3. Do NOT run `git commit`; leave adoption and commit to the Lead Agent.
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


def detect_openai_codex_binary() -> tuple[str | None, str]:
    """Locate a genuine OpenAI Codex CLI binary (rejecting unrelated PyPI 'codex' comic server)."""
    candidates: list[str] = []
    for name in ("codex", "codex.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for cand in candidates:
        p = Path(cand)
        # Guard against Python console_script from PyPI 'codex' (comic-book server)
        try:
            head = p.read_bytes()[:256]
            if b"from codex.run import main" in head or b"comicbox" in head:
                continue
        except OSError:
            pass
        try:
            res = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
            out = (res.stdout + "\n" + res.stderr).strip()
            if res.returncode == 0 and "codex" in out.lower():
                return cand, out
        except Exception:
            continue
    return None, "OpenAI Codex CLI (@openai/codex) not found on PATH (note: .venv/bin/codex is PyPI comic server if installed via pip)."


def build_codex_prompt(task_info: dict, root: Path = ROOT) -> str:
    rel_brief = task_info["path"].relative_to(root)
    allowed_str = ", ".join(task_info["allowed_paths"]) or "(none - read-only)"
    return (
        f"You are collaborating on this repository under the Lead Agent (Antigravity) using role `{task_info['role']}`.\n"
        f"1. Read `AGENTS.md`, `handoff/BOARD.md`, `.codex/agents/{task_info['role']}.toml`, and `{rel_brief}` first.\n"
        f"2. Execute `{task_info['task_id']}` strictly within Allowed Write Paths: [{allowed_str}].\n"
        f"3. TOKEN EFFICIENCY: Read only compact CSV tables/files listed in `{rel_brief}`; skip `runs/` and full `summary.json` unless strictly needed; keep your report concise (<= 120 lines).\n"
        f"4. Do NOT modify Lead-owned governance files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`) "
        f"or sealed `results/` directories, and do NOT run `git commit`.\n"
        f"5. Deliver your verified report to the assigned path in `{rel_brief}`."
    )


def check_scope_against_paths(changed_paths: list[str], allowed_paths: list[str]) -> tuple[list[str], list[str]]:
    """Classify changed_paths into (in_scope, out_of_scope) against allowed_paths (exact or glob/dir prefix)."""
    in_scope: list[str] = []
    out_of_scope: list[str] = []
    for cp in changed_paths:
        ok = False
        for ap in allowed_paths:
            ap_norm = ap.rstrip("/")
            if cp == ap_norm or cp.startswith(ap_norm + "/") or fnmatch(cp, ap):
                ok = True
                break
        if ok:
            in_scope.append(cp)
        else:
            out_of_scope.append(cp)
    return in_scope, out_of_scope


def git_changed_files(root: Path = ROOT) -> list[str]:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    files: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path_part = line[3:].strip()
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1].strip()
        files.append(path_part)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Lead (Antigravity) <-> Codex Collaboration CLI Helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate-codex", help="Validate .codex/config.toml and .codex/agents/*.toml.")
    sub.add_parser("status", help="Show active Codex roles, tasks, reports, and reviews.")

    p_new = sub.add_parser("new-task", help="Create a new scoped task brief under handoff/tasks/.")
    p_new.add_argument("--id", required=True, help="Task ID, e.g. TASK-003")
    p_new.add_argument("--role", required=True, help="Codex agent role name (stem in .codex/agents/)")
    p_new.add_argument("--title", required=True, help="Short imperative task title")
    p_new.add_argument("--allow", required=True, help="Comma-separated Allowed Write Paths")
    p_new.add_argument("--objective", default="", help="Optional detailed objective")

    p_prompt = sub.add_parser("prompt", help="Print a ready-to-use Codex prompt for a task brief.")
    p_prompt.add_argument("--id", required=True, help="Task ID, e.g. TASK-001")

    p_run = sub.add_parser("run", help="Execute a task brief directly via OpenAI Codex CLI (codex exec).")
    p_run.add_argument("--id", required=True, help="Task ID, e.g. TASK-001")
    p_run.add_argument("--dry-run", action="store_true", help="Print the exact codex exec command without running it.")

    p_scope = sub.add_parser("check-scope", help="Check current git working tree changes against a task's Allowed Write Paths.")
    p_scope.add_argument("--id", required=True, help="Task ID, e.g. TASK-001")

    args = parser.parse_args()

    if args.cmd == "validate-codex":
        errs = validate_codex_configs()
        if errs:
            for e in errs:
                print(f"[ERROR] {e}", file=sys.stderr)
            return 1
        roles = [p.stem for p in sorted((CODEX_DIR / "agents").glob("*.toml"))]
        print(f"[OK] .codex/config.toml + {len(roles)} agent roles valid: {', '.join(roles)}")
        return 0

    if args.cmd == "status":
        roles = [p.stem for p in sorted((CODEX_DIR / "agents").glob("*.toml"))]
        print(f"Codex roles ({len(roles)}): {', '.join(roles)}")
        print("\nTasks (handoff/tasks/):")
        for tf in sorted(TASKS_DIR.glob("TASK-*.md")):
            info = parse_task_brief(tf)
            print(f"  - {info['task_id']} [{info['status']}] role={info['role']} | allow={info['allowed_paths']}")
        reports = [p.name for p in sorted(REPORTS_DIR.glob("*.md"))] if REPORTS_DIR.exists() else []
        reviews = [p.name for p in sorted(REVIEWS_DIR.glob("*.md"))] if REVIEWS_DIR.exists() else []
        print(f"\nReports ({len(reports)}): {', '.join(reports) or '(none yet)'}")
        print(f"Reviews ({len(reviews)}): {', '.join(reviews) or '(none yet)'}")
        return 0

    if args.cmd == "new-task":
        allowed = [p.strip() for p in args.allow.split(",") if p.strip()]
        path = create_task_brief(
            task_id=args.id,
            role=args.role,
            title=args.title,
            allowed_paths=allowed,
            objective=args.objective,
        )
        info = parse_task_brief(path)
        print(f"Created {path.relative_to(ROOT)}\n")
        print("--- Copy-Paste Prompt for Codex ---")
        print(build_codex_prompt(info))
        return 0

    if args.cmd == "prompt":
        info = parse_task_brief(find_task_file(args.id))
        print(build_codex_prompt(info))
        return 0

    if args.cmd == "run":
        info = parse_task_brief(find_task_file(args.id))
        prompt = build_codex_prompt(info)
        bin_path, detail = detect_openai_codex_binary()
        if bin_path is None:
            print(f"[BLOCKED] {detail}", file=sys.stderr)
            print("Install OpenAI Codex CLI via: npm install -g @openai/codex", file=sys.stderr)
            print("\n--- Fallback Copy-Paste Prompt ---")
            print(prompt)
            return 2
        cmd = [bin_path, "exec", "--cd", str(ROOT), prompt]
        if args.dry_run:
            print(f"[DRY-RUN] binary={bin_path} ({detail})")
            print("Command:", cmd[:4], "<prompt>")
            return 0
        print(f"[RUN] {bin_path} ({detail}) -> {info['task_id']} ({info['role']})", flush=True)
        return subprocess.call(cmd, cwd=ROOT)

    if args.cmd == "check-scope":
        info = parse_task_brief(find_task_file(args.id))
        changed = git_changed_files()
        in_s, out_s = check_scope_against_paths(changed, info["allowed_paths"])
        print(f"Task {info['task_id']} allowed: {info['allowed_paths']}")
        print(f"In-scope changes ({len(in_s)}): {in_s}")
        print(f"Out-of-scope changes ({len(out_s)}): {out_s}")
        return 0 if not out_s else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
