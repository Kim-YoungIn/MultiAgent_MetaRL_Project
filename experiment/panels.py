"""Deterministic and integrity-checked D-031 task panels."""
import hashlib
import json
import random
import tempfile
from functools import lru_cache
from pathlib import Path

from mamrl.diverse_catalog import BASE_FAMILIES, _entry, build_diverse_manifest
from .generator import candidate, catalog_rows


def digest(value):
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    result = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        result.update(chunk.encode())
    return result.hexdigest()


def _past_fingerprints(n):
    seen = set()
    sources = {}
    unreadable = []
    missing = []
    project = Path(__file__).resolve().parent.parent
    runs = project / 'runs'
    paths = set(runs.rglob('manifest*.json')) | set(runs.rglob('catalog*.json'))
    for path in sorted(paths):
        try:
            data_bytes = path.read_bytes()
            data = json.loads(data_bytes)
            if not isinstance(data, dict):
                raise ValueError('manifest root is not an object')
            if data.get('n') == n:
                sources[str(path.relative_to(project))] = hashlib.sha256(data_bytes).hexdigest()
                for index, row in enumerate(data.get('rules', [])):
                    if 'canonical_fingerprint' in row:
                        seen.add(row['canonical_fingerprint'])
                    else:
                        missing.append(f'{path.relative_to(project)}:rules[{index}]')
        except (OSError, ValueError, TypeError):
            unreadable.append(str(path))
    return seen, sources, unreadable, missing


def prepare(root, *, dev=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    result = {'version': 2, 'generator': 'diverse_relational_d031_v2', 'development': dev, 'training': {}, 'evaluation': {}, 'past_sources': {}, 'unreadable_manifests': [], 'missing_fingerprints': [], 'source_sha256': {}}
    dev_exclusions = {}
    if not dev:
        with tempfile.TemporaryDirectory(prefix='d031-development-') as temp:
            dev_root = Path(temp) / 'panel'
            prepare(dev_root, dev=True)
            dev_exclusions = {n: {r['canonical_fingerprint'] for r in evaluation_entries(dev_root, n)}
                              for n in (3, 4, 5)}
    source_dir = root / 'sources'
    source_dir.mkdir()
    project = Path(__file__).resolve().parent.parent
    source_paths = [*sorted((project / '05_toy_models' / 'mamrl').glob('*.py')),
                    project / 'experiment' / 'generator.py']
    for path in source_paths:
        name = 'd031_generator.py' if path.name == 'generator.py' else path.name
        content = path.read_bytes()
        result['source_sha256'][name] = hashlib.sha256(content).hexdigest()
        (source_dir / name).write_bytes(content)
    for n in (3, 4, 5):
        seen, sources, unreadable, missing = _past_fingerprints(n)
        seen.update(dev_exclusions.get(n, ()))
        result['missing_fingerprints'].extend(missing)
        result['past_sources'].update(sources)
        result['unreadable_manifests'] = sorted(set(result['unreadable_manifests'] + unreadable))
        result['training'][str(n)] = {}
        if not dev:
            for level in ('focused', 'broad'):
                development = catalog_rows(n=n, seed=21001, train_count=8,
                    fresh_count=4, structural_count=2, diversity=level)
                seen.update(row['canonical_fingerprint'] for row in development)
        # Exclude previously exposed D-031 panels whenever their artifacts are available.
        historical = {}
        for name in ('d031_development_final', 'd031_development_20261121'):
            old = project / 'runs' / name / 'panels' / 'panel.json'
            if not old.exists():
                result.setdefault('unavailable_past_d031_panels', []).append(str(old.relative_to(project)))
                continue
            try:
                old_panel = json.loads(old.read_text())
                old_rows = old_panel.get('evaluation', {}).get(str(n), [])
                for catalog in old_panel.get('training', {}).get(str(n), {}).values():
                    for source_rows in catalog.values():
                        old_rows += source_rows
                seen.update(row['canonical_fingerprint'] for row in old_rows)
                historical[str(old.relative_to(project))] = hashlib.sha256(old.read_bytes()).hexdigest()
            except (OSError, ValueError, KeyError, TypeError):
                result['unreadable_manifests'].append(str(old))
        result.setdefault('past_d031_panels', {}).update(historical)
        if n == 3 and not dev:
            for old_seed, old_count in ((31001, 128), (21001, 32)):
                old_manifest = build_diverse_manifest(
                    n=3, seed=old_seed, diversity='broad', train_rules=old_count,
                    fresh_rules=16, structural_rules=16)
                seen.update(row['canonical_fingerprint'] for row in old_manifest['rules'])
        for c in (range(1, 2) if dev else range(1, 6)):
            result['training'][str(n)][str(c)] = {}
            for level in ('focused', 'broad'):
                manifest = catalog_rows(n=n, seed=21001 if dev else 31000+c,
                    train_count=8 if dev else 128, fresh_count=4 if dev else 16,
                    structural_count=2 if dev else 16, diversity=level)
                rows = [row for row in manifest if row['split'] == 'train']
                seen.update(row['canonical_fingerprint'] for row in manifest)
                result['training'][str(n)][str(c)][level] = rows
        rng = random.Random((21001 if dev else 51000) + n)
        rows = []
        counts = {'fresh_rules': 8 if dev else 64, 'structural': 2 if dev else 64}
        for split, count in counts.items():
            attempts = 0
            for index in range(count):
                family = 'exception' if split == 'structural' else BASE_FAMILIES[index % 4]
                for _ in range(20000):
                    attempts += 1
                    source_seed = rng.getrandbits(63)
                    item = candidate(n, source_seed, family)
                    if item is None or item[-1] in seen:
                        continue
                    seen.add(item[-1])
                    rows.append(_entry(split, index, source_seed, attempts, family, item))
                    break
                else:
                    raise RuntimeError(f'cannot generate distinct n={n} {split} family={family} slot={index}')
        result['evaluation'][str(n)] = rows
    result['sha256'] = digest(result)
    with (root / 'panel.json').open('w') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)


@lru_cache(maxsize=2)
def _parse_panel(filename, file_digest):
    with open(filename) as stream:
        panel = json.load(stream)
    expected = panel.pop('sha256')
    if digest(panel) != expected:
        raise ValueError('panel integrity mismatch')
    return panel


def _load(root):
    filename = Path(root) / 'panel.json'
    file_hash = hashlib.sha256()
    with filename.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            file_hash.update(chunk)
    panel = _parse_panel(str(filename.resolve()), file_hash.hexdigest())
    for name, expected_hash in panel['source_sha256'].items():
        source = Path(root) / 'sources' / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('panel source snapshot integrity mismatch')
    return panel


def training_entries(root, n, diversity, c, count):
    rows = _load(root)['training'][str(n)][str(c)][diversity]
    if count <= 0 or count > len(rows):
        raise ValueError('invalid training rule count')
    return rows[:count]


def evaluation_entries(root, n):
    return _load(root)['evaluation'][str(n)]
