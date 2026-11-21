"""Train, evaluate and resume the predeclared D-031 experiment."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import datetime
from tempfile import TemporaryDirectory

import torch

from mamrl.meta_exploration import neural_factory
from .artifacts import atomic_path, evaluate_to_disk, sha_file, verify_evaluation, write_json, write_rows
from .diagnostics import diagnose
from .evaluation import BASELINES, evaluate
from .math_audit import audit_panel, strategy_losses
from .panels import prepare, evaluation_entries, training_entries
from .protocol import budget, plan
from .training import train, load_model


@contextlib.contextmanager
def exclusive(root, *, wait=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f'an experiment process already owns {root}') from error
        yield


def seal(root, dev, device):
    repo = Path(__file__).resolve().parents[1]
    files = [*sorted((repo / 'experiment').glob('*.py')),
             *sorted((repo / '05_toy_models/mamrl').glob('*.py')),
             repo / '04_design/Adaptation_Experiment.md']
    version = {'protocol': 'D-031', 'development': dev, 'device': device,
               'python': platform.python_version(), 'torch': str(torch.__version__),
               'sources': {str(p.relative_to(repo)): sha_file(p) for p in files}}
    path = root / 'study.json'
    if path.exists():
        if json.loads(path.read_text()) != version:
            raise ValueError('source, protocol, device or runtime differs from study.json; use a new output directory')
    elif any(p.name != '.lock' for p in root.iterdir()):
        raise ValueError('use a new empty output directory')
    else:
        write_json(path, version)
    panels = root / 'panels'
    if not panels.exists():
        with TemporaryDirectory(dir=root) as stage:
            child = Path(stage) / 'panels'
            prepare(child, dev=dev)
            os.replace(child, panels)
    panel_sha = sha_file(panels / 'panel.json')
    anchor = root / 'panel_sha256.json'
    if anchor.exists() and json.loads(anchor.read_text()) != panel_sha:
        raise ValueError('sealed panel file has changed')
    write_json(anchor, panel_sha)
    return panels


def audit(root, panels):
    for n in (3, 4, 5):
        path = root / f'audit_n{n}.json'
        if not path.exists():
            print(f'Rule audit n={n}', flush=True)
            write_json(path, audit_panel(evaluation_entries(panels, n)))


def _train_models(directory, entries, cell, limits, device):
    models = {}
    for method in cell['methods']:
        path = directory / 'training' / method
        ledger = path / 'ledger.json'
        if ledger.exists():
            info = json.loads(ledger.read_text())
            if (sha_file(path / 'training_state.pt') != info['checkpoint_sha256']
                    or sha_file(path / 'model_config.pt') != info['model_config_sha256']):
                raise ValueError(f'checkpoint mismatch at {path}')
            models[method] = load_model(path, device=device)
            continue
        keys = ('updates', 'trials', 'k', 'h')
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        model = train(path, entries, method=method, seed=cell['seed'], profiles=cell['profiles'],
                      device=device, **{k: limits[k] for k in keys})
        state = torch.load(path / 'training_state.pt', weights_only=True, map_location='cpu')
        if state['completed_updates'] != limits['updates']:
            raise RuntimeError(f'incomplete training at {path}')
        trials = limits['updates'] * limits['trials']
        write_json(ledger, {'cell': cell, 'method': method, 'device': device,
                            'updates': state['history'], 'real_steps': trials*(limits['k'] + limits['h']),
                            'erasure_steps': trials*limits['h'],
                            'checkpoint_sha256': sha_file(path / 'training_state.pt'),
                            'model_config_sha256': sha_file(path / 'model_config.pt')})
        models[method] = model
    return models


def _run_cell(root, panels, cell, dev, device, transfer):
    limits = budget(dev)
    label = cell['condition']
    directory = root / 'cells' / label / f"c{cell['catalog']}" / f"j{cell['replica']}"
    print(f"{label} catalog={cell['catalog']} replica={cell['replica']}", flush=True)
    train_n = cell['train_n']
    entries = training_entries(panels, train_n, cell['diversity'], cell['catalog'], cell['count'])
    models = _train_models(directory, entries, cell, limits, device)
    targets = (3, 4, 5) if transfer and label in ('primary', 'n5') else (train_n,)
    for n in targets:
        out = directory / f'n{n}'
        tasks = evaluation_entries(panels, n)
        policies = {key: neural_factory(value) for key, value in models.items()
                    if key in ('M', 'P') or n == train_n}
        if n == train_n:
            initial = load_model(directory / 'training/M', initial=True, device=device)
            policies.update(initial=neural_factory(initial), **BASELINES)
        for method, factory in policies.items():
            fixed = label == 'primary' and n == 3 and method in ('M', 'P', 'O', 'random')
            rows = evaluate(tasks, factory, fixed_budget=fixed, reps=limits['reps'],
                            ks=limits['ks'], h=limits['h'])
            if method not in BASELINES:
                evaluate_to_disk(out, method, rows, limits['reps'])
                continue
            # These controls have no fitted state. Every model faces the same paired trials.
            cache = root / 'controls' / f'n{n}' / f'{method}_fixed{int(fixed)}'
            with exclusive(cache, wait=True):
                evaluate_to_disk(cache, method, rows, limits['reps'])
            if (out / (method + '.receipt.json')).exists():
                verify_evaluation(out, method)
                continue
            for suffix in ('.trials.jsonl.gz', '.json.gz', '.receipt.json'):
                filename = method + suffix
                with atomic_path(out / filename) as tmp:
                    shutil.copyfile(cache / filename, tmp)
        if n == 3 and label in ('primary', 'focused') and not (out / 'losses.json').exists():
            cache = root / 'distances' / f"{label}_c{cell['catalog']}.json"
            if not cache.exists():
                write_json(cache, strategy_losses(entries, tasks))
            write_json(out / 'losses.json', json.loads(cache.read_text()))
        if label == 'primary' and n == 3 and cell['replica'] == 1:
            path = out / 'diagnostics.jsonl.gz'
            if not path.exists():
                write_rows(path, diagnose(tasks, models, dev=dev))


def run(root, panels, dev, block, catalog, replica, device):
    cells = [cell for cell in plan(dev) if block == 'all' or block == cell['block']
             or (block == 'actions' and cell['condition'] == 'primary')]
    cells = [cell for cell in cells if (catalog is None or catalog == cell['catalog'])
             and (replica is None or replica == cell['replica'])]
    if not cells:
        raise ValueError('selection matches no cells in the study')
    with exclusive(root / 'locks' / 'audit', wait=True):
        audit(root, panels)
    for cell in cells:
        directory = root / 'cells' / cell['condition'] / f"c{cell['catalog']}" / f"j{cell['replica']}"
        with exclusive(directory, wait=True):
            try:
                _run_cell(root, panels, cell, dev, device, block in ('actions', 'all'))
            except (Exception, KeyboardInterrupt) as error:
                from .artifacts import encode
                category = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'execution_error'
                if 'non-finite' in str(error) or 'nonfinite' in str(error):
                    category = 'numerical_failure'
                with (directory / 'attempts.jsonl').open('a') as log:
                    log.write(encode({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      'category': category, 'type': type(error).__name__,
                                      'message': str(error)}) + '\n')
                raise


def make_report(root, dev):
    from .statistics import report
    result = report(root, plan(dev), dev=dev)
    write_json(root / 'report.json', result)
    from .summary import write_summary
    write_summary(root, result, dev)
    return result
