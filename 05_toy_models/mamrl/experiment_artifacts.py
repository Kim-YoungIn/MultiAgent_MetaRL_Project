"""Store trial data once and load both embedded and referenced experiment records.

The v3 aggregate contains summaries and hashed local references; the hydrated API
retains the original record schema for analysis and HTML rendering.
"""

import hashlib
import json
from pathlib import Path


ARTIFACT_FORMAT = 'mamrl.meta_experiment.references.v1'


def _summary(evaluation):
    return {key: value for key, value in evaluation.items() if key != 'trials'}


def _baseline_summary(baselines):
    return {name: {split: _summary(value) for split, value in splits.items()}
            for name, splits in baselines.items()}


def _run_summary(run):
    return {**run, 'evaluation': {
        stage: {split: _summary(value) for split, value in splits.items()}
        for stage, splits in run['evaluation'].items()}}


def _local_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('artifact path must be a nonempty relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes experiment directory')
    return path


def _reference(root, relative, expected):
    path = _local_path(root, relative)
    data = path.read_bytes()
    if json.loads(data) != expected:
        raise ValueError(f'artifact content differs from record: {relative}')
    return {'path': relative, 'sha256': hashlib.sha256(data).hexdigest()}


def _read_reference(root, reference):
    path = _local_path(root, reference['path'])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError(f'artifact hash mismatch: {reference["path"]}')
    return json.loads(data)


def save_meta_experiment(path, record):
    """Write an exclusive v3 aggregate after checking existing detailed artifacts."""
    path = Path(path)
    if record.get('version') not in (1, 2):
        raise ValueError('expected a hydrated v1/v2 experiment record')
    references = {
        'baselines': _reference(path.parent, 'baselines.json', record['baselines']),
        'runs': [_reference(path.parent, f'seed{run["seed"]}/evaluation.json', run)
                 for run in record['runs']]}
    aggregate = {**record, 'version': 3, 'record_version': record['version'],
                 'artifact_format': ARTIFACT_FORMAT,
                 'baselines': _baseline_summary(record['baselines']),
                 'runs': [_run_summary(run) for run in record['runs']],
                 'references': references}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_meta_experiment(path):
    """Hydrate a v3 aggregate, or read a historical embedded record unchanged."""
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('version') in (1, 2):
        return record
    if (record.get('version') != 3 or record.get('artifact_format') != ARTIFACT_FORMAT
            or record.get('record_version') not in (1, 2)):
        raise ValueError('unsupported experiment artifact format')
    references = record['references']
    baselines = _read_reference(path.parent, references['baselines'])
    runs = [_read_reference(path.parent, ref) for ref in references['runs']]
    if (_baseline_summary(baselines) != record['baselines']
            or [_run_summary(run) for run in runs] != record['runs']):
        raise ValueError('aggregate summaries differ from detailed artifacts')
    result = {key: value for key, value in record.items()
              if key not in ('artifact_format', 'record_version', 'references')}
    result.update(version=record['record_version'], baselines=baselines, runs=runs)
    return result
