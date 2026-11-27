"""Show the research index and check whether its referenced paths exist."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("studies.json")
PATH_FIELDS = ("protocol", "report", "evidence", "raw", "reproduce")
STAGES = {"planned", "in_progress", "partial", "completed", "historical"}


def load_registry(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("studies"), list):
        raise ValueError("unsupported registry format")
    seen = set()
    for study in data["studies"]:
        if not isinstance(study, dict) or not isinstance(study.get("id"), str):
            raise ValueError("each study must have a string id")
        if study["id"] in seen:
            raise ValueError(f"duplicate study id: {study['id']}")
        seen.add(study["id"])
        if study.get("stage") not in STAGES:
            raise ValueError(f"invalid stage for {study['id']}")
        for field in (*PATH_FIELDS, "scope", "verification"):
            if not isinstance(study.get(field), str) or not study[field]:
                raise ValueError(f"missing {field} for {study['id']}")
        for field in PATH_FIELDS:
            path = Path(study[field])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {field} path for {study['id']}")
        for binding in study.get("bindings", []):
            if not isinstance(binding, dict) or not all(isinstance(binding.get(k), str) for k in ("local", "evidence")):
                raise ValueError(f"invalid integrity binding for {study['id']}")
            for key in ("local", "evidence"):
                path = Path(binding[key])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe binding path for {study['id']}")
            if not Path(binding['local']).is_relative_to(Path(study['raw'])):
                raise ValueError(f"binding must refer to the study's raw directory: {study['id']}")
            if 'field' in binding and not isinstance(binding['field'], str):
                raise ValueError(f"invalid binding field for {study['id']}")
    return data["studies"]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_binding(binding):
    local = ROOT / binding['local']
    evidence = ROOT / binding['evidence']
    try:
        expected = (json.loads(evidence.read_text())[binding['field']]
                    if 'field' in binding else sha256(evidence))
        return sha256(local) == expected
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check versioned paths and available local integrity bindings")
    parser.add_argument("--study", help="show one study id")
    parser.add_argument("--registry", type=Path, default=REGISTRY, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        studies = load_registry(args.registry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.study:
        studies = [study for study in studies if study["id"] == args.study]
        if not studies:
            parser.error(f"unknown study: {args.study}")

    errors = 0
    for study in studies:
        missing = []
        for field in PATH_FIELDS:
            exists = (ROOT / study[field]).exists()
            if field != "raw" and not exists and study["stage"] not in {"planned", "in_progress"}:
                errors += 1
                missing.append(field)
        raw_path = ROOT / study['raw']
        raw_exists = raw_path.is_dir()
        bindings = study.get('bindings', [])
        failed = []
        if args.check:
            if raw_path.exists() and not raw_exists:
                failed.append(study['raw'] + ' (expected a directory)')
                errors += 1
            for binding in bindings:
                if not (ROOT / binding['evidence']).is_file():
                    failed.append(binding['evidence'] + ' (evidence absent)')
                    errors += 1
        if args.check and raw_exists:
            for binding in bindings:
                if (ROOT / binding['evidence']).is_file() and not check_binding(binding):
                    failed.append(binding['local'])
                    errors += 1
        if failed:
            integrity = 'integrity FAILED: ' + ', '.join(failed)
        elif not raw_exists:
            integrity = 'raw absent'
        elif not args.check:
            integrity = 'raw present'
        elif bindings:
            integrity = f'{len(bindings)} binding(s) match'
        else:
            integrity = 'raw present; no binding'
        print(f"{study['id']} [{study['stage']}] — {study['scope']} — {integrity}")
        if missing:
            print('  Missing versioned paths: ' + ', '.join(missing))
        if args.study:
            for field in PATH_FIELDS:
                print(f"  {field}: {study[field]} ({'present' if (ROOT / study[field]).exists() else 'absent'})")
            print(f"  verification: {study['verification']}")
            if "status_command" in study:
                print(f"  detailed status: {study['status_command']}")
    print("Bindings check specified small files only. Full artifact verification is documented by each study.")
    return 1 if args.check and errors else 0


if __name__ == "__main__":
    sys.exit(main())
