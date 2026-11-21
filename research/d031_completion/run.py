"""Resume all D-031 cells from a sealed historical source and run snapshot."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_source(study, source):
    manifest = json.loads((study / 'study.json').read_text())
    for name, digest in manifest['sources'].items():
        candidates = [PROJECT / name, study / 'panels' / 'sources' / Path(name).name]
        matched = next((p for p in candidates if p.is_file() and sha(p) == digest), None)
        if matched is None:
            raise ValueError(f'cannot restore sealed source {name}')
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matched, destination)
    expected = set(manifest['sources'])
    actual = {str(p.relative_to(source)) for group in ('experiment', '05_toy_models/mamrl')
              for p in (source/group).glob('*.py')}
    if actual != {p for p in expected if p.endswith('.py')}:
        raise ValueError('isolated historical package contains unexpected sources')
    return manifest


def cell_job(args):
    from experiment.runner import run
    root, catalog, replica = args
    run(root, root/'panels', False, 'all', catalog, replica, 'cpu')
    return catalog, replica


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('original', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--workers', type=int, default=8)
    args = cli.parse_args()
    original, root = args.original.resolve(), args.output.resolve()
    if original == root or original in root.parents:
        cli.error('use a separate continuation directory')
    if args.workers < 1:
        cli.error('workers must be positive')
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(original, root, args.workers)


def execute(original, root, workers):
    work = root/'study'
    if not work.exists():
        import tempfile
        with tempfile.TemporaryDirectory(dir=root, prefix='.incoming-') as temp:
            copied = Path(temp)/'study'
            shutil.copytree(original, copied, ignore=shutil.ignore_patterns('.lock', 'locks'))
            copied.rename(work)
    for name in ('study.json', 'panel_sha256.json'):
        if sha(original/name) != sha(work/name):
            raise ValueError(f'continuation identity mismatch: {name}')
    if json.loads((work/'panel_sha256.json').read_text()) != sha(work/'panels'/'panel.json'):
        raise ValueError('continuation panel differs from seal')
    source = root/'source'
    manifest = restore_source(original, source)
    sys.path[:0] = [str(source),str(source/'05_toy_models')]
    import torch
    dev, device = manifest['development'], manifest['device']
    if manifest['python'] != platform.python_version() or manifest['torch'] != str(torch.__version__):
        raise ValueError('historical runtime mismatch')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('historical study requires an accessible CUDA device')
    if device != 'cpu' and not dev:
        raise ValueError('parallel continuation of a full run requires the historical CPU configuration')
    torch.set_num_threads(1)
    from experiment.runner import seal, exclusive, make_report, run
    with exclusive(work):
        seal(work, dev, device)
    if dev:
        run(work, work/'panels', True, 'all', None, None, device)
        with exclusive(work/'locks'/'report', wait=True):
            make_report(work, True)
        print(work/'summary.md', flush=True)
        return
    print(f'sealed inputs verified for {work}; {workers} workers', flush=True)
    jobs = [(work, c, r) for r in range(1,6) for c in range(1,6)]
    # Each shard carries out seven conditions. Prioritize all primary cells
    # so the primary result can be computed while extension cells continue.
    ctx = multiprocessing.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as executor:
        list(executor.map(primary_job, jobs))
        print('all primary cells evaluated', flush=True)
        with exclusive(work/'locks'/'report',wait=True):
            make_report(work, False)
        list(executor.map(cell_job, jobs))
    with exclusive(work/'locks'/'report',wait=True):
        make_report(work, False)
    print(work/'summary.md', flush=True)


def primary_job(args):
    from experiment.runner import run
    root, catalog, replica = args
    run(root,root/'panels',False,'primary',catalog,replica,'cpu')
    return catalog, replica


if __name__ == '__main__':
    main()
