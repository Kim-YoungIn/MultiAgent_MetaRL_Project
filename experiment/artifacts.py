"""Atomic run artifacts and paired trial reduction."""
import contextlib
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@contextlib.contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(encode(value) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with atomic_path(path) as tmp, gzip.open(tmp, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(encode(row) + '\n')


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _curve(records, name):
    fields = ('return', 'first', 'early3', 'G', 'game', 'q_reuse', 'q_revisit',
              'explore_mean', 'first_explore_reward', 'opponent_nonuniform_fraction')
    result = []
    for points in zip(*(record[name] for record in records), strict=True):
        averaged = {'k': points[0]['k'], 'h': points[0]['h']}
        if any((p['k'], p['h']) != (averaged['k'], averaged['h']) for p in points):
            raise ValueError('inconsistent evaluation budgets')
        for field in fields:
            averaged[field] = mean(p[field] for p in points) if points[0][field] is not None else None
        result.append(averaged)
    return result


def evaluate_to_disk(directory, method, rows, reps):
    raw = directory / (method + '.trials.jsonl.gz')
    summary = directory / (method + '.json.gz')
    receipt = directory / (method + '.receipt.json')
    if receipt.exists():
        verify_evaluation(directory, method)
        return
    reduced = []
    with atomic_path(raw) as tmp, gzip.open(tmp, 'wt', encoding='utf-8') as log:
        for _, group in itertools.groupby(rows, lambda r: (r['rule'], r['profile'])):
            records = list(group)
            if [r['rep'] for r in records] != list(range(reps)):
                raise ValueError('incomplete trial repetitions')
            row = {k: records[0][k] for k in ('rule', 'family', 'split', 'profile')}
            zero = []
            for r in records:
                log.write(encode(r) + '\n')
                zero.append([r['seed'], r['curve'][0]])
            row.update(auc=mean(r['auc'] for r in records), curve=_curve(records, 'curve'),
                       zero_hash=hashlib.sha256(encode(zero).encode()).hexdigest())
            if 'fixed_budget' in records[0]:
                row['fixed_budget'] = _curve(records, 'fixed_budget')
            reduced.append(row)
    with atomic_path(summary) as tmp, gzip.open(tmp, 'wt', encoding='utf-8') as stream:
        json.dump(reduced, stream, allow_nan=False)
    write_json(receipt, {p.name: sha_file(p) for p in (raw, summary)})


def verify_evaluation(directory, method):
    receipt = directory / (method + '.receipt.json')
    expected = json.loads(receipt.read_text())
    for filename in (method + '.trials.jsonl.gz', method + '.json.gz'):
        if sha_file(directory / filename) != expected[filename]:
            raise ValueError(f'evaluation checksum mismatch for {directory / filename}')
