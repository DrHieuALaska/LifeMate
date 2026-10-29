"""Stream train JSONL; validate the existing manifests rather than generating splits."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from lifemate.agent import AgentInput
from lifemate.tools.catalog import Catalog
from lifemate.utils.json import loads
from lifemate.utils.safety import safe_path


def read_ids(path):
    path = safe_path(path)
    if not path.is_file():
        raise ValueError('Manifest missing. Restore the existing EDA validation_split manifests; see scripts/explore_lifemate.py --help. Do not generate a new random split.')
    ids = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Manifest contains duplicate IDs')
    return ids


def build_agent_input(record, catalog):
    """Allowlist input fields; historical actions/results are legitimate pre-existing history."""
    source = record.get('context', {})
    context = {k: deepcopy(source[k]) for k in ('current_time', 'timezone') if k in source}
    context['location'] = {k: deepcopy(v) for k, v in source.get('location', {}).items() if k in ('lat', 'lng')}
    context['user_profile'] = {k: deepcopy(v) for k, v in source.get('user_profile', {}).items() if k in ('name', 'home_address', 'work_address', 'city')}
    context['installed_apps'] = {app: {'tools': list(spec.get('tools', []))} for app, spec in source.get('installed_apps', {}).items()}
    history = tuple({k: deepcopy(turn[k]) for k in ('role', 'audio', 'audio_text', 'text', 'tool_calls', 'tool_results') if k in turn} for turn in record.get('history', []))
    return AgentInput(record.get('audio_text', ''), history, context, catalog.schemas(context))


@dataclass
class EvaluationDataset:
    records: list[dict]  # Evaluator private. Never pass these objects to a predictor.
    catalog: Catalog
    dev_ids: list[str]
    training_tool_frequency: Counter
    manifest_checksum: str
    partition_validation: dict


def load_dev(data_root, manifest, expected_count=193, limit=None):
    root, manifest = safe_path(data_root), safe_path(manifest)
    if manifest.name != 'dev_ids.txt':
        raise ValueError('Routine development evaluation requires dev_ids.txt; holdout evaluation is not supported here')
    dev = read_ids(manifest)
    if len(dev) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} unique development IDs; found {len(dev)}')
    partitions = {'dev': set(dev)}
    for name in ('train', 'holdout'):
        path = manifest.parent / (name+'_ids.txt')
        if path.is_file():
            partitions[name] = set(read_ids(path))
    for left in partitions:
        for right in partitions:
            if left < right and partitions[left] & partitions[right]:
                raise ValueError('Train/dev/holdout manifests overlap')
    if limit is not None and not 1 <= limit <= len(dev):
        raise ValueError('--limit must be between 1 and the full manifest count')
    selected = dev[:limit] if limit else dev
    selected_set = set(selected)
    chosen, seen, training_frequency = {}, set(), Counter()
    source = safe_path(root/'train/train.jsonl')
    if not source.is_file():
        raise ValueError('Missing train/train.jsonl. Set --data-root to the confidential dataset directory.')
    with source.open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = loads(line)
                sid = record['id']
                if not isinstance(sid, str):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                raise ValueError(f'Invalid training record at line {line_no}') from None
            if sid in seen:
                raise ValueError('Duplicate training sample ID')
            seen.add(sid)
            if sid in selected_set:
                if not isinstance(record.get('label'), dict) or not call_shape(record['label'].get('tool_calls')):
                    raise ValueError('Selected training record has no valid gold tool-call list')
                chosen[sid] = record
            if sid in partitions.get('train', set()):
                training_frequency.update(c['name'] for c in record['label']['tool_calls'])
    if any(ids-seen for ids in partitions.values()):
        raise ValueError('Manifest contains IDs absent from train/train.jsonl')
    return EvaluationDataset([chosen[sid] for sid in selected], Catalog.from_path(root/'train/train_tools_catalog.json'),
                             dev, training_frequency, sha256(manifest.read_bytes()).hexdigest(),
                             {'available_partitions': sorted(partitions), 'disjoint': True,
                              'all_ids_known': True, 'full_dev_count': len(dev),
                              'evaluated_count': len(selected), 'is_subset': len(selected) != len(dev),
                              'training_frequency_available': 'train' in partitions})


@dataclass
class Predictions:
    calls: dict
    malformed_ids: set
    duplicate_ids: set
    file_errors: dict


def call_shape(calls):
    return isinstance(calls, list) and all(isinstance(c, dict) and isinstance(c.get('name'), str)
        and bool(c['name']) and isinstance(c.get('arguments'), dict) for c in calls)


def load_predictions(path, dev_ids):
    expected = set(dev_ids)
    calls, malformed, duplicates, seen, errors = {}, set(), set(), set(), Counter()
    with safe_path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = loads(line)
            except ValueError:
                errors['malformed_json_lines'] += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id']:
                errors['invalid_id_records'] += 1
                continue
            sid = record['id']
            if sid in seen:
                duplicates.add(sid)
                errors['duplicate_id_records'] += 1
            seen.add(sid)
            if sid not in expected:
                errors['extra_id_records'] += 1
                continue
            if not call_shape(record.get('tool_calls')):
                malformed.add(sid)
                errors['invalid_call_shape_records'] += 1
                continue
            calls[sid] = record['tool_calls']
    malformed |= duplicates & expected
    for sid in malformed:
        calls.pop(sid, None)
    return Predictions(calls, malformed, duplicates, dict(errors))
