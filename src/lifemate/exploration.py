"""Offline, allowlisted LifeMate exploration. Never instantiate ToolEnv or read state files."""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import itertools
import json
import math
import os
import platform
import random
import re
import sys
import unicodedata
import wave
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lifemate.utils.language import PATTERNS, REGEX, language_flags

SEED = 20261029
COUNTERS = collections.Counter
ACCESS = COUNTERS()
DENIED = []
FIGURES = []


def restricted(path):
    try:
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        return any(p.lower().endswith('_environment') or p.lower() == 'state.json'
                   for p in Path(path).parts)
    except TypeError:
        return False


def install_guard():
    def guard(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and args:
            if restricted(args[0]):
                DENIED.append(event)
                raise PermissionError('Restricted environment access blocked')
            if event == 'open' and isinstance(args[0], (str, bytes)):
                suffix = Path(os.fsdecode(args[0])).suffix.lower()
                ACCESS[suffix or '<none>'] += 1
    sys.addaudithook(guard)


def safe_path(path):
    if restricted(path):
        raise PermissionError('Restricted environment path')
    path = Path(path).resolve()
    if restricted(path):
        raise PermissionError('Restricted environment path')
    return path


def discover_root(start):
    start = safe_path(start)
    for p in (start, *start.parents):
        if ((p / 'README_DATA.md').is_file() or (p / 'README.md').is_file()) and (p / 'train' / 'train.jsonl').is_file():
            return p
    raise FileNotFoundError('No README_DATA.md/README.md + train/train.jsonl in root or ancestors')


def stats(values):
    a = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if not len(a):
        return {'n': 0}
    return {'n': len(a), 'min': float(a.min()), 'mean': float(a.mean()),
            'median': float(np.median(a)), 'std_population': float(a.std()),
            'p25': float(np.quantile(a, .25)), 'p75': float(np.quantile(a, .75)),
            'p90': float(np.quantile(a, .9)), 'p95': float(np.quantile(a, .95)),
            'p99': float(np.quantile(a, .99)), 'max': float(a.max())}


def pct(n, d):
    return round(100 * n / d, 3) if d else 0.0


def normalize(text):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold()))


# These are dictionaries of exploratory markers, never linguistic ground truth.


# Fixed generic vocabulary prevents exporting personal names or arbitrary transcript tokens.
SAFE_WORDS = set('bật tắt đặt xóa xoá gửi gọi nhắn tìm mở đóng thêm sửa đổi hủy huỷ xem kiểm tra lịch nhắc báo thức hôm nay mai sáng chiều tối giờ phút ngày tuần tháng năm rồi và sau đó tiếp theo đừng không thôi nhầm nó cái người lúc nãy tôi mình giúp cho với đi nhé ơi là có được cần muốn hãy ở trong trên đến về của một hai ba bốn năm sáu bảy tám chín mười phòng đèn quạt điều hòa tiền chuyển thanh toán hóa đơn thuốc uống đặt món xe liên hệ danh bạ tin nhắn thời tiết nhiệt độ cuộc họp công việc bật lên xuống lại'.split())
SAFE_PHRASES = ['ngày mai', 'tối nay', 'tuần sau', 'sau đó', 'tiếp theo', 'báo thức',
                'điều hòa', 'lịch họp', 'nhắc tôi', 'giúp tôi', 'lúc nãy', 'cái đó',
                'à nhầm', 'đừng', 'chuyển tiền', 'gửi tin nhắn', 'thời tiết']





def structural_shape(x, prefix='', depth=0):
    out = []
    out.append((prefix or '$', type(x).__name__))
    if depth < 4:
        if isinstance(x, dict):
            for k, v in x.items():
                out.extend(structural_shape(v, f'{prefix}.{k}' if prefix else k, depth+1))
        elif isinstance(x, list):
            for v in x:
                out.extend(structural_shape(v, prefix+'[]', depth+1))
    return out


def nested_constraints(schema, path='$'):
    rows = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k not in ('properties', 'items', 'description', 'type', 'required'):
                rows.append((path, k, v))
        for k, v in schema.get('properties', {}).items():
            rows.extend(nested_constraints(v, path+'.'+k))
        if 'items' in schema:
            rows.extend(nested_constraints(schema['items'], path+'[]'))
    return rows


def read_catalog(root, split):
    source = safe_path(root / split / f'{split}_tools_catalog.json')
    catalog = json.loads(source.read_text(encoding='utf-8-sig'))
    tools, inventory, app_rows, param_rows = {}, [], [], []
    for app in catalog.get('apps', []):
        app_rows.append({'app': app['id'], 'group': app.get('group'), 'tools': len(app.get('tools', []))})
        for tool in app.get('tools', []):
            name = tool['name']
            schema = tool.get('parameters', {})
            props = schema.get('properties', {})
            required = schema.get('required', [])
            constraints = nested_constraints(schema)
            tools[name] = {**tool, 'app_id': app['id'], 'group': app.get('group')}
            inventory.append({'app': app['id'], 'group': app.get('group'), 'tool': name,
                              'kind': tool.get('kind', 'unknown'), 'description': tool.get('description', ''),
                              'parameter_count': len(props), 'required_count': len(required),
                              'required': required, 'optional': sorted(set(props)-set(required)),
                              'parameter_types': {k: v.get('type') for k, v in props.items()},
                              'ref_count': sum(k == 'x-ref' for _, k, _ in constraints),
                              'fuzzy_count': sum(k == 'x-match' and v == 'fuzzy' for _, k, v in constraints),
                              'enum_count': sum(k == 'enum' for _, k, _ in constraints),
                              'one_of_count': sum(k == 'x-one-of' for _, k, _ in constraints),
                              'constraints': constraints})
            for key, spec in props.items():
                param_rows.append({'tool': name, 'parameter': key, 'required': key in required,
                                   'type': spec.get('type'), 'schema': spec})
    summary = {'schema_version': catalog.get('schema_version'), 'apps': len(app_rows),
               'tools': len(tools), 'kind_counts': dict(COUNTERS(t.get('kind', 'unknown') for t in tools.values())),
               'required_count': stats(r['required_count'] for r in inventory),
               'parameters': len(param_rows),
               **{f'tools_with_{k}': sum(r[k+'_count'] > 0 for r in inventory) for k in ('ref', 'fuzzy', 'enum', 'one_of')},
               **{f'{k}_occurrences': sum(r[k+'_count'] for r in inventory) for k in ('ref', 'fuzzy', 'enum', 'one_of')}}
    return tools, inventory, app_rows, param_rows, summary


def runtime_system_tools(root, split):
    """Read source syntax only, without importing the runtime or loading any environment."""
    source = safe_path(root / split / 'zaic_tools' / 'executor.py')
    if not source.is_file():
        return set()
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'allowed' for t in node.targets):
            # Only unconditional literal sets unioned into self.allowed are exceptions.
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.BitOr) and isinstance(node.value.right, ast.Set):
                names.update(x.value for x in node.value.right.elts if isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value.startswith('sys.'))
    return names


def audio_metadata(root, split, reference, cache):
    if not isinstance(reference, str) or not reference:
        return {'status': 'missing_reference'}
    try:
        p = safe_path(root / split / reference)
        if not p.is_relative_to((root / split / 'audio').resolve()) or p.suffix.lower() != '.wav':
            return {'status': 'unsafe_or_nonwav_reference'}
        key = str(p)
        if key in cache:
            return cache[key]
        with wave.open(str(p), 'rb') as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            # Detect truncated PCM payload without decoding audio or running ASR.
            actual = 0
            while chunk := wav.readframes(65536):
                actual += len(chunk)
            expected = frames * width * channels
            result = {'status': 'ok' if actual == expected else 'truncated',
                      'duration': frames/rate if rate else None, 'rate': rate,
                      'channels': channels, 'bits': 8*width, 'frames': frames,
                      'compression': wav.getcomptype()}
    except FileNotFoundError:
        result = {'status': 'missing_file'}
    except PermissionError:
        result = {'status': 'blocked_reference'}
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        result = {'status': 'corrupt_'+type(exc).__name__}
    cache[str(p) if 'p' in locals() else reference] = result
    return result


def scalar_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from ((key, x) for _, x in scalar_values(value))
    elif isinstance(obj, list):
        for value in obj:
            yield from scalar_values(value)
    elif isinstance(obj, (str, int)) and not isinstance(obj, bool):
        yield ('', obj)


def validate_arguments(arguments, schema):
    """Conservative static check, not a replacement for official runtime validation."""
    errors = []
    if not isinstance(arguments, dict):
        return ['arguments_not_object']
    props = schema.get('properties', {})
    for key in schema.get('required', []):
        if key not in arguments:
            errors.append('missing_required')
    types = {'string': str, 'integer': int, 'number': (int, float), 'boolean': bool,
             'array': list, 'object': dict, 'null': type(None)}
    for key, val in arguments.items():
        spec = props.get(key)
        if spec is None:
            if schema.get('additionalProperties') is False:
                errors.append('unexpected_argument')
            continue
        typ = spec.get('type')
        if isinstance(typ, str) and typ in types:
            valid = isinstance(val, types[typ]) and not (typ in ('integer', 'number') and isinstance(val, bool))
            if not valid:
                errors.append('wrong_type')
        if 'enum' in spec and val not in spec['enum']:
            errors.append('outside_enum')
        if isinstance(val, (float, int)) and not isinstance(val, bool):
            if 'minimum' in spec and val < spec['minimum']:
                errors.append('below_minimum')
            if 'maximum' in spec and val > spec['maximum']:
                errors.append('above_maximum')
    return errors


def analyze_split(root, split, tools, supervised=True, progress=True):
    rows, texts, raw_texts, ids = [], [], [], []
    system_tools = runtime_system_tools(root, split) & set(tools)
    effective_available = COUNTERS()
    explicit_unlisted = COUNTERS()
    quality, shape, context_fields, profiles, locations = COUNTERS(), COUNTERS(), COUNTERS(), COUNTERS(), COUNTERS()
    installed, available, history_patterns, zones, hours, month_counts = [COUNTERS() for _ in range(6)]
    selected, first, app_selected, kinds, transitions, kind_transitions, sequences = [COUNTERS() for _ in range(7)]
    results_keys, result_status, changed_apps, history_tool_names, arg_types = [COUNTERS() for _ in range(5)]
    issues, audio_cache, audio_occurrences, dependency, schema_errors = [], {}, [], COUNTERS(), COUNTERS()
    arg_names, safe_words, safe_phrases = COUNTERS(), COUNTERS(), COUNTERS()
    history_roles, history_text_types = COUNTERS(), COUNTERS()
    expected = {'id': str, 'audio': str, 'context': dict, 'history': list}
    if supervised:
        expected.update(audio_text=str, label=dict)
    jsonl_path = safe_path(root / split / f'{split}.jsonl')
    with jsonl_path.open(encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                quality['blank_lines'] += 1
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeError):
                quality['malformed_json_lines'] += 1
                issues.append({'sample_id': '', 'line': line_number, 'issue': 'malformed_json'})
                continue
            if not isinstance(item, dict):
                quality['nonobject_records'] += 1
                continue
            sid = item.get('id')
            sid = sid if isinstance(sid, str) else f'__invalid_line_{line_number}'
            ids.append(sid)
            for key, typ in expected.items():
                if key not in item:
                    quality['missing_'+key] += 1
                elif not isinstance(item[key], typ):
                    quality['wrong_type_'+key] += 1
                elif isinstance(item[key], str) and not item[key].strip():
                    quality['empty_'+key] += 1
            # Public-test audit inspects only explicitly permitted input structures.
            permitted = {k: item[k] for k in ('id', 'audio', 'context', 'history') if k in item}
            if supervised:
                permitted = {k: v for k, v in item.items() if k != 'label'}
            for path, typ in set(structural_shape(permitted)):
                # Free-form installed app keys are catalog IDs; profile keys only, never values.
                shape[(path, typ)] += 1
            context = item.get('context') if isinstance(item.get('context'), dict) else {}
            context_fields.update(context.keys())
            profile = context.get('user_profile')
            location = context.get('location')
            if isinstance(profile, dict):
                profiles.update(k for k, v in profile.items() if v is not None and v != '')
            if isinstance(location, dict):
                locations.update(k for k, v in location.items() if v is not None)
            zone = context.get('timezone')
            if isinstance(zone, str):
                zones[zone] += 1
            current_time = context.get('current_time')
            if isinstance(current_time, str):
                try:
                    timestamp = dt.datetime.fromisoformat(current_time)
                    hours[timestamp.hour] += 1
                    month_counts[timestamp.strftime('%Y-%m')] += 1
                except ValueError:
                    quality['unparseable_current_time'] += 1
            apps = context.get('installed_apps', {})
            if not isinstance(apps, dict):
                quality['invalid_installed_apps'] += 1
                apps = {}
            permitted_tools = set()
            for app, spec in apps.items():
                installed[app] += 1
                if app not in {t['app_id'] for t in tools.values()}:
                    quality['unknown_installed_app_occurrences'] += 1
                names = spec.get('tools', []) if isinstance(spec, dict) else []
                if not isinstance(names, list):
                    quality['invalid_available_tool_list'] += 1
                    continue
                for name in names:
                    if not isinstance(name, str):
                        quality['invalid_available_tool_name'] += 1
                        continue
                    permitted_tools.add(name)
                    if name not in tools:
                        quality['unknown_available_tool_occurrences'] += 1
                        issues.append({'sample_id': sid, 'line': line_number, 'issue': 'unknown_available_tool'})
                    elif tools[name]['app_id'] != app:
                        quality['available_tool_wrong_app'] += 1
            available.update(permitted_tools)
            effective_tools = permitted_tools | system_tools
            effective_available.update(effective_tools)
            history = item.get('history', [])
            history = history if isinstance(history, list) else []
            roles, history_calls = [], 0
            user_n = assistant_n = 0
            for turn in history:
                if not isinstance(turn, dict):
                    quality['malformed_history_turn'] += 1
                    continue
                role = turn.get('role', 'missing')
                roles.append(role if role in ('user', 'assistant', 'system', 'tool') else 'other')
                history_roles[roles[-1]] += 1
                if role == 'user':
                    user_n += 1
                    meta = audio_metadata(root, split, turn.get('audio'), audio_cache)
                    audio_occurrences.append({'role': 'history', **meta})
                    if supervised:
                        history_text_types[type(turn.get('audio_text')).__name__] += 1
                elif role == 'assistant':
                    assistant_n += 1
                    hc = turn.get('tool_calls', [])
                    if isinstance(hc, list):
                        history_calls += len(hc)
                        if supervised:
                            history_tool_names.update(c.get('name', 'missing') for c in hc if isinstance(c, dict))
            history_patterns[' → '.join(roles) or '(empty)'] += 1
            meta = audio_metadata(root, split, item.get('audio'), audio_cache)
            audio_occurrences.append({'role': 'current', **meta})
            row = {'sample_id': sid, 'duration_seconds': meta.get('duration'),
                   'history_turns': len(history), 'history_user_turns': user_n,
                   'history_assistant_turns': assistant_n, 'history_tool_calls': history_calls,
                   'installed_apps': len(apps), 'available_tools': len(permitted_tools),
                   'runtime_available_tools': len(effective_tools)}
            if supervised:
                text = item.get('audio_text', '')
                text = text if isinstance(text, str) else ''
                raw_texts.append(text)
                norm = normalize(text)
                texts.append(norm)
                flags = language_flags(text)
                row.update({'transcript_chars': len(text), 'transcript_tokens': len(text.split()), **flags})
                safe_words.update(t for t in norm.split() if t in SAFE_WORDS)
                safe_phrases.update(p for p in SAFE_PHRASES if re.search(r'(?<!\w)'+re.escape(p)+r'(?!\w)', norm))
                label = item.get('label') if isinstance(item.get('label'), dict) else {}
                for key, typ in [('tool_calls', list), ('tool_results', list), ('final_state', dict)]:
                    if key not in label:
                        quality['label_missing_'+key] += 1
                    elif not isinstance(label[key], typ):
                        quality['label_wrong_type_'+key] += 1
                calls = label.get('tool_calls', [])
                calls = calls if isinstance(calls, list) else []
                outputs = label.get('tool_results', [])
                outputs = outputs if isinstance(outputs, list) else []
                final = label.get('final_state', {})
                # Deliberately inspect final_state keys ONLY. Never recurse into app-state values.
                if isinstance(final, dict):
                    changed_apps.update(final.keys())
                    row['changed_app_count'] = len(final)
                if len(calls) != len(outputs):
                    quality['call_result_length_mismatch'] += 1
                    issues.append({'sample_id': sid, 'line': line_number, 'issue': 'call_result_length_mismatch'})
                names, call_kinds, app_set, prior_values = [], [], set(), set()
                required_n = supplied_n = ref_n = id_n = 0
                dependent_calls = ref_matches = id_matches = general_matches = 0
                error_calls = 0
                for idx, call in enumerate(calls):
                    if not isinstance(call, dict):
                        quality['malformed_gold_call'] += 1
                        continue
                    name = call.get('name')
                    name = name if isinstance(name, str) else '<missing>'
                    arguments = call.get('arguments', {})
                    names.append(name)
                    tool = tools.get(name, {})
                    kind = tool.get('kind', 'unknown')
                    call_kinds.append(kind)
                    app_set.add(tool.get('app_id', name.split('.')[0]))
                    if name not in tools:
                        quality['unknown_gold_tool_calls'] += 1
                    if name not in permitted_tools:
                        explicit_unlisted[name] += 1
                        issues.append({'sample_id': sid, 'line': line_number, 'issue': 'system_tool_absent_from_explicit_list' if name in system_tools else 'gold_tool_not_permitted'})
                    if name not in effective_tools:
                        quality['gold_tool_not_permitted_calls'] += 1
                    schema = tool.get('parameters', {})
                    required_n += len(schema.get('required', []))
                    errs = validate_arguments(arguments, schema)
                    if errs:
                        error_calls += 1
                        schema_errors.update(errs)
                        issues.append({'sample_id': sid, 'line': line_number, 'issue': 'static_argument_schema'})
                    has_match = False
                    if isinstance(arguments, dict):
                        supplied_n += len(arguments)
                        arg_names.update(arguments.keys())
                        arg_types.update(type(v).__name__ for v in arguments.values())
                        for key, value in arguments.items():
                            prop = schema.get('properties', {}).get(key, {})
                            is_ref = any(k == 'x-ref' for _, k, _ in nested_constraints(prop))
                            is_id = key == 'id' or key.endswith('_id') or key.endswith('_ids')
                            ref_n += is_ref
                            id_n += is_id
                            matching = {str(v) for _, v in scalar_values(value) if isinstance(v, str) and len(v) >= 3} & prior_values
                            if matching:
                                has_match = True
                                general_matches += 1
                                ref_matches += is_ref
                                id_matches += is_id
                    dependent_calls += has_match
                    if idx < len(outputs):
                        output = outputs[idx]
                        if isinstance(output, dict):
                            results_keys.update(output.keys())
                            status = 'ok_true' if output.get('ok') is True else 'ok_false' if output.get('ok') is False else 'ok_absent_or_nonboolean'
                            result_status[status] += 1
                            if 'error' in output:
                                result_status['has_error_key'] += 1
                            if 'done' in output:
                                result_status['done_true' if output['done'] is True else 'done_false' if output['done'] is False else 'done_other'] += 1
                            if output.get('waiting_for_user') is True:
                                result_status['waiting_for_user_true'] += 1
                            prior_values.update(str(v) for _, v in scalar_values(output) if isinstance(v, str) and len(v) >= 3)
                        else:
                            result_status['nonobject'] += 1
                selected.update(names)
                if names:
                    first[names[0]] += 1
                kinds.update(call_kinds)
                app_selected.update(tools.get(name, {}).get('app_id', name.split('.')[0]) for name in names)
                transitions.update(zip(names, names[1:]))
                kind_transitions.update(zip(call_kinds, call_kinds[1:]))
                sequences[tuple(names)] += 1
                read_before_write = any(a == 'read' and 'write' in call_kinds[i+1:] for i, a in enumerate(call_kinds))
                row.update({'tool_calls': len(calls), 'trajectory_apps': len(app_set),
                            'trajectory_business_apps': len(app_set-{'sys'}),
                            'required_arguments': required_n, 'supplied_arguments': supplied_n,
                            'reference_arguments': ref_n, 'id_arguments': id_n,
                            'result_match_calls': dependent_calls, 'ref_result_matches': ref_matches,
                            'id_result_matches': id_matches, 'general_result_matches': general_matches,
                            'read_before_write': read_before_write, 'ask_missing_info': 'sys.ask_missing_info' in names,
                            'repeated_tools': len(names) > len(set(names)), 'schema_error_calls': error_calls,
                            'tool_sequence': tuple(names)})
                # Transparent composite risk index, not empirical difficulty.
                score = (int(len(calls) >= 2) + int(len(calls) >= 4) + int(len(app_set) >= 2)
                         + int(bool(history)) + int(dependent_calls > 0) + int(ref_n > 0 or id_n > 0)
                         + int(flags['relative_time'] or flags['date'] or flags['clock_time'])
                         + int(row['transcript_tokens'] >= 40) + int((row['duration_seconds'] or 0) >= 15)
                         + int(read_before_write) + int(row['ask_missing_info']))
                row['difficulty_score'] = score
                row['difficulty'] = 'simple' if score <= 2 else 'moderate' if score <= 5 else 'complex'
                dependency.update({'samples_with_result_match': dependent_calls > 0,
                                   'samples_with_ref_match': ref_matches > 0,
                                   'samples_with_id_match': id_matches > 0,
                                   'matched_calls': dependent_calls})
            rows.append(row)
            if progress and len(rows) % 400 == 0:
                print(f'{split}: processed {len(rows)} records', flush=True)
    counts = COUNTERS(ids)
    duplicate_ids = {k: v for k, v in counts.items() if v > 1}
    for sid in duplicate_ids:
        issues.append({'sample_id': sid, 'line': '', 'issue': 'duplicate_id'})
    summary = {'samples': len(rows), 'unique_ids': len(counts), 'duplicate_id_values': len(duplicate_ids),
               'duplicate_id_excess_rows': sum(v-1 for v in duplicate_ids.values()), 'quality': dict(quality),
               'history_nonempty': sum(r['history_turns'] > 0 for r in rows),
               'history_empty': sum(r['history_turns'] == 0 for r in rows),
               'multiple_installed_apps': sum(r['installed_apps'] > 1 for r in rows),
               'history_assistant_tool_call_samples': sum(r['history_tool_calls'] > 0 for r in rows),
               'history_tool_call_occurrences': sum(r['history_tool_calls'] for r in rows),
               'context_field_presence': dict(context_fields), 'profile_field_presence_nonempty': dict(profiles),
               'location_field_presence_nonnull': dict(locations), 'timezones': dict(zones),
               'current_time_hours': dict(hours), 'current_time_months': dict(month_counts),
               'installed_app_frequency': dict(installed), 'available_tool_frequency': dict(available),
               'runtime_available_tool_frequency': dict(effective_available),
               'runtime_system_tool_exceptions': sorted(system_tools),
               'history_role_counts': dict(history_roles), 'history_patterns': dict(history_patterns),
               'numeric': {k: stats(r.get(k) for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)} if rows else {},
               'audio': audio_summary(audio_cache, audio_occurrences)}
    if supervised:
        raw_counts = COUNTERS(raw_texts)
        norm_counts = COUNTERS(texts)
        summary.update({'unique_raw_utterances': len(raw_counts), 'unique_normalized_utterances': len(norm_counts),
                        'exact_raw_duplicate_groups': sum(v > 1 for v in raw_counts.values()),
                        'exact_raw_duplicate_excess': sum(v-1 for v in raw_counts.values()),
                        'normalized_duplicate_groups': sum(v > 1 for v in norm_counts.values()),
                        'normalized_duplicate_excess': sum(v-1 for v in norm_counts.values()),
                        'normalized_duplicate_participating_samples': sum(v for v in norm_counts.values() if v > 1),
                        'language_marker_counts': {k: sum(r[k] for r in rows) for k in PATTERNS},
                        'gold_total_calls': sum(selected.values()), 'gold_tool_frequency': dict(selected),
                        'gold_app_frequency': dict(app_selected), 'gold_kind_frequency': dict(kinds),
                        'gold_first_tool_frequency': dict(first), 'changed_app_frequency': dict(changed_apps),
                        'zero_calls': sum(r['tool_calls'] == 0 for r in rows),
                        'single_call': sum(r['tool_calls'] == 1 for r in rows),
                        'multi_call': sum(r['tool_calls'] > 1 for r in rows),
                        'multi_app': sum(r['trajectory_apps'] > 1 for r in rows),
                        'multi_business_app': sum(r['trajectory_business_apps'] > 1 for r in rows),
                        'gold_calls_absent_from_explicit_list': dict(explicit_unlisted),
                        'read_before_write_samples': sum(r['read_before_write'] for r in rows),
                        'ask_missing_info_samples': sum(r['ask_missing_info'] for r in rows),
                        'repeated_tool_samples': sum(r['repeated_tools'] for r in rows),
                        'difficulty_counts': dict(COUNTERS(r['difficulty'] for r in rows)),
                        'dependency_heuristics': dict(dependency), 'result_status': dict(result_status),
                        'result_top_level_key_frequency': dict(results_keys), 'static_schema_errors': dict(schema_errors),
                        'static_schema_error_calls': sum(r['schema_error_calls'] for r in rows),
                        'argument_value_types': dict(arg_types), 'history_audio_text_types': dict(history_text_types),
                        'kind_transitions': [{'from': a, 'to': b, 'count': c} for (a, b), c in kind_transitions.most_common()],
                        'correlations_spearman': correlations(rows),
                        'reference_marker_by_history': reference_by_history(rows)})
    tables = {'field_types': [{'path': p, 'type': t, 'samples': n} for (p, t), n in sorted(shape.items())],
              'integrity_issues': issues,
              'history_patterns': [{'pattern': p, 'samples': n} for p, n in history_patterns.most_common()],
              'audio_metadata': audio_metadata_rows(audio_cache)}
    durations = COUNTERS((r['role'], int(r['duration'])) for r in audio_occurrences if r.get('duration') is not None)
    tables['audio_duration_bins'] = [{'role': role, 'lower_seconds_inclusive': lower,
                                     'upper_seconds_exclusive': lower+1, 'references': count}
                                    for (role, lower), count in sorted(durations.items())]
    if supervised:
        tables.update({'tool_transitions': [{'from': a, 'to': b, 'count': c} for (a, b), c in transitions.most_common()],
                       'kind_transitions': summary['kind_transitions'],
                       'tool_sequences': [{'sequence': list(s), 'samples': n} for s, n in sequences.most_common()],
                       'safe_common_words': [{'word': k, 'occurrences': n} for k, n in safe_words.most_common()],
                       'safe_common_phrases': [{'phrase': k, 'samples': n} for k, n in safe_phrases.most_common()],
                       'first_tool_frequency': [{'tool': k, 'samples': n} for k, n in first.most_common()],
                       'history_tool_frequency': [{'tool': k, 'calls': n} for k, n in history_tool_names.most_common()]})
    return rows, texts, summary, tables


def audio_metadata_rows(cache):
    counts = COUNTERS((r.get('status'), r.get('rate'), r.get('channels'), r.get('bits'), r.get('compression')) for r in cache.values())
    return [{'status': s, 'sample_rate_hz': r, 'channels': c, 'bit_depth': b, 'compression': p, 'unique_files': n}
            for (s, r, c, b, p), n in counts.items()]


def audio_summary(cache, occurrences):
    ok = [r for r in cache.values() if r['status'] == 'ok']
    return {'unique_referenced_files': len(cache), 'reference_occurrences': len(occurrences),
            'unique_status_counts': dict(COUNTERS(r['status'] for r in cache.values())),
            'reference_status_counts': dict(COUNTERS(r['status'] for r in occurrences)),
            'sample_rates': dict(COUNTERS(r['rate'] for r in ok)),
            'channels': dict(COUNTERS(r['channels'] for r in ok)),
            'bit_depths': dict(COUNTERS(r['bits'] for r in ok)),
            'unique_total_hours': sum(r['duration'] for r in ok)/3600,
            'unique_duration_seconds': stats(r['duration'] for r in ok),
            'current_duration_seconds': stats(r.get('duration') for r in occurrences if r['role'] == 'current'),
            'history_duration_seconds': stats(r.get('duration') for r in occurrences if r['role'] == 'history'),
            'current_total_hours': sum(r.get('duration', 0) or 0 for r in occurrences if r['role'] == 'current')/3600,
            'history_total_hours': sum(r.get('duration', 0) or 0 for r in occurrences if r['role'] == 'history')/3600}


def reference_by_history(rows):
    return {name: {'samples': len(sub), 'reference_marker_samples': sum(r['reference'] for r in sub),
                   'reference_marker_percent': pct(sum(r['reference'] for r in sub), len(sub))}
            for name, sub in [('with_history', [r for r in rows if r['history_turns']]),
                              ('without_history', [r for r in rows if not r['history_turns']])]}


def correlations(rows):
    keys = ['duration_seconds', 'transcript_tokens', 'history_turns', 'tool_calls', 'trajectory_apps',
            'required_arguments', 'difficulty_score']
    out = []
    for a, b in itertools.combinations(keys, 2):
        values = [(r.get(a), r.get(b)) for r in rows if r.get(a) is not None and r.get(b) is not None]
        if not values:
            continue
        x, y = np.asarray(values).T
        def ranks(v):
            order = np.argsort(v, kind='stable')
            ranks = np.empty(len(v), dtype=float)
            start = 0
            while start < len(v):
                end = start+1
                while end < len(v) and v[order[end]] == v[order[start]]:
                    end += 1
                ranks[order[start:end]] = (start+end-1)/2
                start = end
            return ranks
        rx, ry = ranks(x), ranks(y)
        rho = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else None
        out.append({'a': a, 'b': b, 'spearman_rho': rho, 'n': len(x)})
    return out


def duplicate_groups(texts, threshold=.8):
    """Exact normalized matches + character 5-shingle Jaccard >= .80, length ratio >= .8."""
    parent = list(range(len(texts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    first, unique, pairs = {}, [], []
    for i, text in enumerate(texts):
        if text in first:
            union(i, first[text])
        else:
            first[text] = i
            unique.append(i)
    shingles = {i: {texts[i][j:j+5] for j in range(len(texts[i])-4)} for i in unique if len(texts[i]) >= 20}
    candidates = list(shingles)
    for pos, i in enumerate(candidates):
        for j in candidates[pos+1:]:
            a, b = shingles[i], shingles[j]
            if min(len(texts[i]), len(texts[j])) / max(len(texts[i]), len(texts[j])) < .8:
                continue
            if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            intersection = len(a & b)
            similarity = intersection / (len(a)+len(b)-intersection)
            if similarity >= threshold:
                union(i, j)
                pairs.append((i, j, similarity))
    groups = collections.defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    return list(groups.values()), pairs


def make_splits(rows, groups, seed=SEED):
    """Group-constrained greedy allocation with tool/task coverage and seeded tie breaks."""
    n = len(rows)
    targets = [round(n*.75), round(n*.125)]
    targets.append(n-sum(targets))
    rng = random.Random(seed)
    features = []
    for r in rows:
        features.append(set(['tool:'+t for t in r['tool_sequence']] +
                            ['difficulty:'+r['difficulty'], 'history:'+str(bool(r['history_turns'])),
                             'calls:'+str(min(r['tool_calls'], 4))]))
    total = COUNTERS(f for fs in features for f in sorted(fs))
    tool_group_counts = COUNTERS(t for g in groups for t in set(t for i in g for t in rows[i]['tool_sequence']))
    order = [(g, rng.random()) for g in groups]
    order.sort(key=lambda x: (-len(x[0]), x[1]))
    allocations, sizes, assigned_features = [[], [], []], [0, 0, 0], [COUNTERS(), COUNTERS(), COUNTERS()]
    for group, _ in order:
        gf = COUNTERS(f for i in group for f in sorted(features[i]))
        rare = any(tool_group_counts[t] < 3 for i in group for t in rows[i]['tool_sequence'])
        scores = []
        for part in range(3):
            size_score = ((sizes[part]+len(group)-targets[part])/max(targets[part], 1))**2
            before = ((sizes[part]-targets[part])/max(targets[part], 1))**2
            coverage = sum((assigned_features[part][f]/max(total[f]*targets[part]/max(n, 1), 1)) * c for f, c in sorted(gf.items()))/max(sum(gf.values()), 1)
            score = size_score-before + .002*coverage
            if rare and part != 0:
                score += 10
            if sizes[part]+len(group) > targets[part]:
                score += 2
            scores.append(score)
        part = min(range(3), key=lambda p: (scores[p], p))
        allocations[part].extend(group)
        sizes[part] += len(group)
        assigned_features[part].update(gf)
    names = ['train', 'dev', 'holdout']
    membership = {i: names[p] for p, group in enumerate(allocations) for i in group}
    summaries = {}
    all_tools = set(t for r in rows for t in r['tool_sequence'])
    for p, indexes in enumerate(allocations):
        used = set(t for i in indexes for t in rows[i]['tool_sequence'])
        summaries[names[p]] = {'target_samples': targets[p], 'actual_samples': len(indexes),
                               'unique_gold_tools': len(used), 'missing_gold_tools': sorted(all_tools-used),
                               'difficulty': dict(COUNTERS(rows[i]['difficulty'] for i in indexes)),
                               'history_samples': sum(rows[i]['history_turns'] > 0 for i in indexes)}
    assert len(membership) == n
    assert all(len({membership[i] for i in g}) == 1 for g in groups)
    return allocations, membership, summaries


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else ['sample_id', 'issue']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def count_rows(counter, key='category', denom=None):
    c = COUNTERS(counter)
    return [{key: k, 'count': v, **({'percent': pct(v, denom)} if denom is not None else {})} for k, v in c.most_common()]


def plots(rows, s, inventory, out, tables):
    plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 180, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    color = '#287a9c'
    def save(fig, name, caption, totals):
        fig.tight_layout()
        fig.savefig(out / (name+'.png'), bbox_inches='tight')
        plt.close(fig)
        FIGURES.append({'file': name+'.png', 'caption': caption, 'observations': totals})
    def bars(name, counter, title, xlabel, ylabel='Count', horizontal=False, caption='', limit=None):
        items = COUNTERS(counter).most_common(limit)
        if items and all(isinstance(k, (float, int)) for k, _ in items):
            items.sort()
        fig, ax = plt.subplots(figsize=(10, max(4, len(items)*.27) if horizontal else 4.5))
        labels, values = [str(k) for k, _ in items], [v for _, v in items]
        if horizontal:
            ax.barh(labels[::-1], values[::-1], color=color)
        else:
            ax.bar(labels, values, color=color)
            if len(labels) > 12:
                ax.tick_params(axis='x', labelrotation=60)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis='x' if horizontal else 'y', alpha=.18)
        save(fig, name, caption, sum(values))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    current = [r['duration_seconds'] for r in rows if r['duration_seconds'] is not None]
    axes[0].hist(current, bins=35, color=color, edgecolor='white')
    axes[0].set(title='Current-request WAV duration', xlabel='Duration (seconds)', ylabel='Samples')
    axes[1].boxplot(current, orientation='horizontal', showmeans=True)
    axes[1].set(title='Current-request duration spread', xlabel='Duration (seconds)', yticks=[])
    save(fig, '01_audio_duration', 'Referenced training current requests; WAV frame count / sample rate. Box shows median and interquartile range; dots are Tukey outliers.', len(current))
    bars('02_history_length', COUNTERS(r['history_turns'] for r in rows), 'Conversation history length', 'Historical turns', 'Samples', caption='Counts of all historical user and assistant turns per training sample.')
    bars('03_gold_calls', COUNTERS(r['tool_calls'] for r in rows), 'Gold trajectory length', 'Gold tool calls', 'Samples', caption='Final-request gold calls only; historical calls are excluded.')
    bars('04_top_gold_tools', s['gold_tool_frequency'], 'Most frequent gold tools (top 20)', 'Gold calls', 'Tool', True, 'Counts of tool occurrences, including repeated calls within a sample.', 20)
    bars('05_tool_kinds', s['gold_kind_frequency'], 'Gold call kinds', 'Catalog kind', 'Gold calls', caption='Read/write/meta labels are taken from the tool catalog; all gold calls included.')
    bars('06_gold_apps', s['gold_app_frequency'], 'Gold tool usage by app', 'Gold calls', 'App', True, 'App of every gold call; a multi-app sample contributes to multiple apps.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ['installed_apps', 'available_tools'], ['Installed apps per sample', 'Permitted tools per sample']):
        ax.hist([r[key] for r in rows], bins=20, color=color, edgecolor='white')
        ax.set(title=title, xlabel='Count per sample', ylabel='Samples')
    save(fig, '07_availability', 'Counts from context.installed_apps; tools are the explicit permitted subset, not the entire catalog.', len(rows))
    bars('08_required_arguments', COUNTERS(r['required_count'] for r in inventory), 'Required arguments in catalog tools', 'Required parameters', 'Catalog tools', caption='Top-level JSON Schema required list; x-one-of alternatives are counted separately in schema tables.')
    bars('09_trajectory_classes', {'zero': s['zero_calls'], 'single': s['single_call'], 'multi-step': s['multi_call']}, 'Final-request trajectory classes', 'Trajectory class', 'Samples', caption='Zero = 0 calls, single = 1 call, multi-step = at least 2 calls.')
    bars('10_difficulty', s['difficulty_counts'], 'Heuristic engineering-risk categories', 'Composite category', 'Samples', caption='Fixed additive risk index: simple 0–2, moderate 3–5, complex 6–11. This is not measured model difficulty.')
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.hexbin([r['transcript_tokens'] for r in rows], [r['tool_calls'] for r in rows], gridsize=25, mincnt=1, cmap='viridis')
    fig.colorbar(c, ax=ax, label='Samples per hexagon')
    ax.set(title='Transcript length and gold action count', xlabel='Whitespace tokens in current transcript', ylabel='Gold tool calls')
    save(fig, '11_transcript_actions', 'Whitespace tokens are not Vietnamese lexical word counts. All training samples are included.', len(rows))
    bars('12_kind_transitions', {r['from']+' → '+r['to']: r['count'] for r in s['kind_transitions']}, 'Adjacent gold tool-kind transitions', 'Adjacent transitions', 'Transition', True, 'Each trajectory contributes L−1 adjacent transitions; read→write alone does not prove dependency.')
    bars('13_catalog_kinds', COUNTERS(r['kind'] for r in inventory), 'Tool kinds in complete catalog', 'Catalog kind', 'Catalog tools', caption='Complete training catalog, irrespective of per-sample availability or gold usage.')
    bars('14_language_markers', s['language_marker_counts'], 'Exploratory language marker prevalence', 'Training samples matching marker', 'Marker', True, 'Regex dictionaries overlap. These are indicators, with false positives and false negatives, not annotated intent classes.')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, values = [], []
    for key in sorted(set(r['tool_calls'] for r in rows)):
        v = [r['duration_seconds'] for r in rows if r['tool_calls'] == key and r['duration_seconds'] is not None]
        if v:
            labels.append(str(key)); values.append(v)
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set(title='Audio duration by trajectory length', xlabel='Gold tool calls', ylabel='Current-request duration (seconds)')
    save(fig, '15_duration_actions', 'Duration distributions by final-request action count; outlier points hidden, all measurements retained in statistics.', sum(map(len, values)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    audio_bins = tables['audio_duration_bins']
    for role, label in [('current', 'Current'), ('history', 'Historical user')]:
        subset = [r for r in audio_bins if r['role'] == role]
        if subset:
            max_bin = max(r['lower_seconds_inclusive'] for r in subset)
            counts = {r['lower_seconds_inclusive']: r['references'] for r in subset}
            axes[0].stairs([counts.get(i, 0) for i in range(max_bin+1)], np.arange(max_bin+2), label=label, linewidth=1.6)
    axes[0].set(title='Current and historical audio', xlabel='WAV duration (seconds)', ylabel='Referenced WAV occurrences')
    axes[0].legend()
    boxstats = []
    for key, label in [('current_duration_seconds', 'Current'), ('history_duration_seconds', 'History')]:
        d = s['audio'][key]
        if d['n']:
            boxstats.append({'label': label, 'q1': d['p25'], 'med': d['median'], 'q3': d['p75'], 'whislo': d['min'], 'whishi': d['max'], 'fliers': []})
    axes[1].bxp(boxstats, orientation='horizontal')
    axes[1].set(title='Duration summaries by audio role', xlabel='WAV duration (seconds)')
    save(fig, '16_audio_roles', 'One-second histogram bins, reference-weighted by role. Boxes show quartiles/median; whiskers span observed minimum–maximum (not Tukey fences).', sum(r['references'] for r in audio_bins))


def markdown_table(rows, columns, limit=None):
    if limit is not None:
        rows = rows[:limit]
    def fmt(v):
        if isinstance(v, float):
            return f'{v:,.3f}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |'] +
                     ['| '+' | '.join(fmt(r.get(c, '')) for c in columns)+' |' for r in rows])


def summary_table(values):
    return markdown_table([{'measure': k, **v} for k, v in values.items()],
                          ['measure', 'n', 'min', 'mean', 'median', 'std_population', 'p90', 'p95', 'p99', 'max'])


def build_report(root, out, s, catalog_s, rows, inventory, tables, split_s, public, version):
    n, calls = s['samples'], s['gold_total_calls']
    sections = []
    def add(text):
        sections.append(text.strip())
    def freq(counter, title, denominator, limit=15):
        add(title+'\n\n'+markdown_table(count_rows(counter, 'item', denominator), ['item', 'count', 'percent'], limit))
    add('# LifeMate Dataset Exploration\n\nOffline analysis of the local competition release. All measured findings below come from the executed pipeline. No ASR model, LLM, ToolEnv, history replay, or gold execution was run. Public-test observations are isolated as an unlabeled distribution audit.')
    add(f'''## 1. Executive Summary

- The training JSONL contains **{n:,} records**, **{s['unique_ids']:,} unique IDs**, and **{calls:,} gold calls** across **{len(s['gold_tool_frequency'])} observed tools**. The catalog contains **{catalog_s['apps']} apps / {catalog_s['tools']} tools**.
- **{s['multi_call']:,} samples ({pct(s['multi_call'], n):.2f}%)** require multiple calls; **{s['multi_app']:,} ({pct(s['multi_app'], n):.2f}%)** use multiple apps including sys, or **{s['multi_business_app']:,} ({pct(s['multi_business_app'], n):.2f}%)** excluding sys. Maximum trajectory length is **{s['numeric']['tool_calls']['max']:.0f}** calls.
- **{s['history_nonempty']:,} ({pct(s['history_nonempty'], n):.2f}%)** contain history; the final-request reference-marker heuristic matches **{s['language_marker_counts']['reference']:,} ({pct(s['language_marker_counts']['reference'], n):.2f}%)**.
- Referenced training audio totals **{s['audio']['unique_total_hours']:.3f} hours** over **{s['audio']['unique_referenced_files']:,} unique paths**. Current-request median / p95 duration is **{s['audio']['current_duration_seconds']['median']:.3f} / {s['audio']['current_duration_seconds']['p95']:.3f} seconds**.
- Gold trajectories contain **{s['dependency_heuristics']['samples_with_ref_match']:,} samples** with an x-ref argument exactly matching an earlier returned string. This is heuristic evidence for runtime entity resolution, not proof of causal reliance.
- **{sum(s['gold_calls_absent_from_explicit_list'].values())} gold calls** are absent from explicit installed lists, but static runtime-source inspection confirms the always-available system-tool exception. There are **{s['quality'].get('gold_tool_not_permitted_calls', 0)} genuine permission discrepancies** after applying it.
- The immediate engineering priority is a training-only evaluation harness, schema/availability enforcement, and a result-aware agent loop. Benchmark ASR on execution-sensitive entities and time expressions; this exploration provides no ASR performance evidence.''')
    add(f'''## 2. Dataset Description

The root was discovered from the requested directory or its ancestors by requiring README.md and train/train.jsonl. The authoritative source is [README](../README.md). Package metadata:

```text
{version.strip()}
```

The observed top-level layout matches the README: train and public_test each contain a JSONL, a tool catalog, audio, a split environment directory, and zaic_tools. The documented private_test is a future release and is absent. The public split also has sample_submission.jsonl. Environment directories were not used by the analysis.

Current audio_text is a string, not a list or token object. context contains current_time, timezone, location, user_profile and installed_apps. History is a list of role-tagged user/assistant turns. Gold calls are objects with name and arguments; results align by list index. final_state maps changed app IDs to full app states: only its app keys were inspected.

Only the final user request is scored. **History actions are already applied to the initial state and must not be replayed.** At inference, each sample permits a single independent ToolEnv instance; all calls, including rejected calls, count. Source environment state must be accessed through the official interface only. No environment access was needed here.

The README prohibits sharing/distributing the package. All processing stayed local; no dataset content was uploaded, committed, or sent to a model. Derived artifacts contain aggregate measurements, catalog schemas, and sample IDs/feature counts; they remain competition-confidential.''')
    quality_rows = count_rows(s['quality'], 'issue', n)
    add('## 3. Data Quality and Integrity\n\n'+markdown_table([
        {'check': 'Parsed training records', 'count': n, 'percent': 100.0},
        {'check': 'Unique IDs', 'count': s['unique_ids'], 'percent': pct(s['unique_ids'], n)},
        {'check': 'Duplicate ID excess rows', 'count': s['duplicate_id_excess_rows'], 'percent': pct(s['duplicate_id_excess_rows'], n)},
        {'check': 'Nonempty histories', 'count': s['history_nonempty'], 'percent': pct(s['history_nonempty'], n)},
        {'check': 'Empty histories', 'count': s['history_empty'], 'percent': pct(s['history_empty'], n)},
        {'check': 'Multiple installed apps', 'count': s['multiple_installed_apps'], 'percent': pct(s['multiple_installed_apps'], n)},
        {'check': 'Static schema error calls', 'count': s['static_schema_error_calls'], 'percent': pct(s['static_schema_error_calls'], calls)},
    ], ['check', 'count', 'percent'])+
        '\n\nNonzero integrity counters:\n\n'+(markdown_table(quality_rows, ['issue', 'count', 'percent']) if quality_rows else 'None detected by implemented checks.')+
        '\n\nPercentages above use samples, except static schema errors, which use calls. Issue counters containing “calls” or “occurrences” can exceed a sample count. Full anomaly/observation IDs are in tables/integrity_issues.csv; type paths and sample prevalence are in tables/field_types.csv. Empty history, zero tool calls, empty final_state, absent optional assistant calls/results, and optional catalog arguments are legitimate. Missing documented top-level or gold-label fields, wrong types, unresolved WAV references, unknown catalogs, and unpermitted gold calls are treated as anomalies. Static validation checks top-level required/type/enum/minimum/maximum and explicitly forbidden extra arguments; it does not certify x-one-of, nested schemas, cross-app references, fuzzy matching, or runtime behavior.')
    add('### Explicit integrity check totals\n\n'+markdown_table(tables['integrity_checks'], ['check', 'anomaly_count', 'denominator', 'percent']))
    add(f"### Explicit-list discrepancy resolved by runtime documentation\n\n{sum(s['gold_calls_absent_from_explicit_list'].values())} gold calls ({pct(sum(s['gold_calls_absent_from_explicit_list'].values()), calls):.3f}% of calls) are absent from the explicit context tool lists: `{json.dumps(s['gold_calls_absent_from_explicit_list'])}`. The main README describes only installed tools, but [zaic_tools README](../train/zaic_tools/README.md) explicitly documents sys.ask_missing_info as always available; [executor.py](../train/zaic_tools/executor.py) unions it into self.allowed and creates the system app. The pipeline reads and parses that source syntax without importing or running it. This is a documentation clarification and legitimate meta behavior, not mislabeled training data. Effective availability includes the observed runtime system exceptions `{s['runtime_system_tool_exceptions']}`. No environment metadata was read to verify actual per-sample _meta.tools; permission checks compare the JSONL allowlists plus these source-defined exceptions.")
    add('## 4. Audio and Language Characteristics\n\n### Audio measurements\n\n'+summary_table({
        'current WAV duration (seconds)': s['audio']['current_duration_seconds'],
        'historical WAV duration (seconds)': s['audio']['history_duration_seconds'],
        'all unique WAV duration (seconds)': s['audio']['unique_duration_seconds']})+
        f"\n\nCurrent audio contributes {s['audio']['current_total_hours']:.3f} hours and historical references {s['audio']['history_total_hours']:.3f} hours (reference-weighted); unique-path total is {s['audio']['unique_total_hours']:.3f} hours. Standard deviations use population normalization, and quantiles use NumPy's default linear interpolation.\n\n"+
        markdown_table(tables['audio_metadata'], ['status', 'sample_rate_hz', 'channels', 'bit_depth', 'compression', 'unique_files'])+
        '\n\nAudio reference statuses: `'+json.dumps(s['audio']['reference_status_counts'])+'`. PCM payload byte counts were checked against WAV frame metadata, without interpreting sample amplitudes. This catches missing/truncated/corrupt WAVs but does not measure clipping, noise, accents, speakers, intelligibility, or transcript correctness. An inventory of unreferenced audio files was deliberately not attempted.')
    add('### Transcript measurements\n\n'+summary_table({k: s['numeric'][k] for k in ('transcript_chars', 'transcript_tokens')})+
        f"\n\nThere are {s['unique_raw_utterances']:,} distinct raw strings and {s['unique_normalized_utterances']:,} distinct normalized strings. Raw exact duplicates: {s['exact_raw_duplicate_groups']} groups / {s['exact_raw_duplicate_excess']} excess rows. Normalized duplicates: {s['normalized_duplicate_groups']} groups / {s['normalized_duplicate_excess']} excess rows / {s['normalized_duplicate_participating_samples']} participating samples. Normalization uses NFC, Unicode casefold, punctuation-to-space, and whitespace collapse, preserving Vietnamese diacritics. Whitespace tokens approximate syllable/orthographic units, not segmented Vietnamese words.\n\n"+
        'To prevent personal text export, common-token output is restricted to a fixed generic Vietnamese vocabulary; phrase output uses a fixed generic phrase list. Thus it is a **privacy-filtered frequency view**, not a complete vocabulary ranking. No transcript examples, names, contact values, addresses, coordinates, or arbitrary transcript n-grams are exported.\n\n'+
        markdown_table(tables['safe_common_words'], ['word', 'occurrences'], 25)+'\n\n'+
        markdown_table(tables['safe_common_phrases'], ['phrase', 'samples'], 20))
    freq(s['language_marker_counts'], '### Language indicators (overlapping sample-level regex matches)', n, 30)
    add('The complete regex definitions are in tables/language_heuristics.csv and the script. Relative-time markers include tomorrow/tonight/next week equivalents; clock and date markers are separate. Quantity includes digit-unit expressions and common number words; this overcounts ambiguous Vietnamese words such as “năm”. Contact is a mention/intent proxy, not detected entities. Negation includes “không”, which also occurs in questions. Reference includes broad words such as “đó”, “lại”, and “tiếp”, so it cannot establish actual history dependence. English indicators combine loanwords, brands and app vocabulary; they do not measure true code-switching rates. Regexes may miss spoken variants and produce false positives. No regional-accent, gender, speaker-identity, WER, or noise claims can be made from this analysis.')
    add('## 5. Conversation and Context Characteristics\n\n'+summary_table({k: s['numeric'][k] for k in ('history_turns', 'history_user_turns', 'history_assistant_turns', 'history_tool_calls', 'installed_apps', 'available_tools')})+
        f"\n\nHistorical assistant tool calls occur in {s['history_assistant_tool_call_samples']} samples and total {s['history_tool_call_occurrences']} calls. Historical user audio_text types: `{json.dumps(s['history_audio_text_types'])}`. These histories were summarized, not executed.\n\n"+
        markdown_table(tables['history_patterns'], ['pattern', 'samples'], 12)+'\n\n'+
        markdown_table([{'history_group': k, **v} for k, v in s['reference_marker_by_history'].items()], ['history_group', 'samples', 'reference_marker_samples', 'reference_marker_percent']))
    freq(s['context_field_presence'], '### Context field availability', n, 30)
    freq(s['profile_field_presence_nonempty'], 'Profile fields present and nonempty (values never exported)', n, 30)
    freq(s['location_field_presence_nonnull'], 'Location fields present and nonnull (coordinates never exported)', n, 30)
    freq(s['timezones'], 'Timezones', n, 30)
    freq(s['current_time_hours'], 'Current-time hour distribution (hour recorded in timestamp)', n, 24)
    freq(s['current_time_months'], 'Current-time calendar-month distribution', n, 30)
    freq(s['installed_app_frequency'], '### Installed app frequency (top apps; full CSV available)', n, 20)
    add('current_time plus timezone anchors relative dates, weekdays and time-of-day interpretation. Profile/location availability supports home/work and geographic resolution, but values should remain private. The installed-app mapping plus the runtime system-tool exception forms the execution allowlist. Prior user audio and assistant tool calls/results may resolve entities and references. Reference-marker prevalence with and without history measures linguistic association only; a useful follow-up is manual annotation and history ablation on development data.')
    add('## 6. Tool Ecosystem\n\n'+markdown_table([{'measure': k, 'value': v} for k, v in catalog_s.items() if k not in ('required_count', 'kind_counts')], ['measure', 'value'])+
        '\n\n'+markdown_table(count_rows(catalog_s['kind_counts'], 'kind', catalog_s['tools']), ['kind', 'count', 'percent'])+
        '\n\nTop-level required-argument counts:\n\n'+summary_table({'required per catalog tool': catalog_s['required_count']})+
        '\n\nThe complete tool inventory includes app/group/kind/description, required and optional arguments, types, enums, x-ref, fuzzy matching, x-one-of and all additional schema constraints. Parameter-level schemas are in tables/catalog_parameters.csv. tools_with_* counts count tools containing a constraint; *_occurrences counts nested schema occurrences.')
    complex_tools = sorted(inventory, key=lambda r: (r['required_count']+r['ref_count']+r['one_of_count'], r['parameter_count']), reverse=True)
    add('### Schema-heavy tool examples\n\n'+markdown_table(complex_tools, ['tool', 'parameter_count', 'required_count', 'ref_count', 'enum_count', 'fuzzy_count', 'one_of_count'], 15)+
        '\n\nEnums and numeric bounds support deterministic rejection before execution. x-ref arguments require grounded IDs obtained through allowed tools rather than invented values. Fuzzy-match fields should preserve user intent while avoiding unnecessary exact-string penalties. Observed x-one-of is a list of alternative parameter-name groups (e.g., contact_id or phone_number); one complete group must be present. Separate required arguments from conditionally required alternatives. Tool availability is each sample’s installed-app list plus the source-defined system exception; the full catalog is a schema source, not universal permission. Unused catalog tools still need valid general handling, but their performance cannot be estimated from absent gold examples.')
    app_inventory = collections.defaultdict(lambda: {'tools': 0, 'required_total': 0, 'ref_constraints': 0})
    for r in inventory:
        app_inventory[r['app']]['tools'] += 1
        app_inventory[r['app']]['required_total'] += r['required_count']
        app_inventory[r['app']]['ref_constraints'] += r['ref_count']
    add('### Catalog tools per app\n\n'+markdown_table([{'app': a, **v} for a, v in sorted(app_inventory.items(), key=lambda x: -x[1]['tools'])], ['app', 'tools', 'required_total', 'ref_constraints']))
    add(f'''## 7. Gold Action Trajectory Analysis

There are **{calls:,} calls**: {s['zero_calls']} zero-call samples ({pct(s['zero_calls'], n):.2f}%), {s['single_call']} single-call samples ({pct(s['single_call'], n):.2f}%), and {s['multi_call']} multi-call samples ({pct(s['multi_call'], n):.2f}%). {s['multi_app']} samples use multiple apps including sys ({pct(s['multi_app'], n):.2f}%); excluding sys, {s['multi_business_app']} samples use multiple business apps ({pct(s['multi_business_app'], n):.2f}%). Repeated tool names occur in {s['repeated_tool_samples']} samples ({pct(s['repeated_tool_samples'], n):.2f}%); repeated tool names do not imply identical calls.

{summary_table({k: s['numeric'][k] for k in ('tool_calls', 'trajectory_apps', 'required_arguments', 'supplied_arguments', 'reference_arguments', 'id_arguments')})}

Argument totals are summed across each gold trajectory, including repeats; required_arguments comes from schema requirements, while supplied_arguments measures actual top-level keys. ID arguments are identified by id/_id/_ids names. x-ref is inspected in the supplied parameter’s nested schema.''')
    freq(s['gold_tool_frequency'], '### Most selected tools', calls, 20)
    freq(s['gold_first_tool_frequency'], '### First-action tools', n-s['zero_calls'], 15)
    freq(s['gold_app_frequency'], '### Gold app usage', calls, 20)
    freq(s['gold_kind_frequency'], '### Gold call kinds', calls, 10)
    rare_tools = [{'tool': t, 'gold_calls': c} for t, c in sorted(s['gold_tool_frequency'].items(), key=lambda x: (x[1], x[0])) if c <= 5]
    add(f"### Sparse tool families and coverage limits\n\n{len(rare_tools)} observed tools have at most five gold calls each. {catalog_s['tools']-len(s['gold_tool_frequency'])} catalog tools have no gold call. These counts limit reliable per-tool evaluation; preserve training coverage and add synthetic schema/behavior tests after the baseline, without claiming measured generalization for tools absent from labels.\n\n"+markdown_table(rare_tools, ['tool', 'gold_calls'], 25))
    add('### Common final-request trajectories\n\n'+markdown_table(tables['tool_sequences'], ['sequence', 'samples'], 15)+
        '\n\n### Adjacent kind transitions\n\n'+markdown_table(s['kind_transitions'], ['from', 'to', 'count']))
    add(f"Read-before-write (any earlier read followed by any later write) occurs in {s['read_before_write_samples']} samples ({pct(s['read_before_write_samples'], n):.2f}%). sys.ask_missing_info occurs in {s['ask_missing_info_samples']} samples ({pct(s['ask_missing_info_samples'], n):.2f}%). It is a valid meta action, not an execution failure.\n\n"+
        markdown_table([{'heuristic': k, 'count': v} for k, v in s['dependency_heuristics'].items()], ['heuristic', 'count'])+
        '\n\nDependency matching compares nontrivial string argument leaves (length ≥3) with string leaves from **earlier** aligned tool results. It counts general matches and separately marks x-ref and ID-shaped parameters. Common labels, enum text and preknown identifiers can match accidentally; numerical/boolean leaves are excluded to limit coincidence, so numerical dependencies and paraphrases are missed. No causal dependency is asserted from sequence order alone.\n\n'+
        '### Returned-result structure\n\n'+markdown_table(count_rows(s['result_status'], 'status'), ['status', 'count'])+
        '\n\nThe mutually exclusive ok_true/ok_false/ok_absent_or_nonboolean/nonobject categories summarize aligned results; has_error_key, done and waiting_for_user categories overlap them. **ok=true denotes acceptance, not necessarily completed user intent**: the runtime documents done=false for valid no-ops (e.g., unresolved IDs or insufficient balance). A missing done field must not be assumed to mean success. Extra unmatched results, if any, are flagged by length checks. No result values or error messages are exported. Changed-app frequency below uses final_state keys only.\n\n'+
        markdown_table(count_rows(s['changed_app_frequency'], 'app'), ['app', 'count']))
    usage = tables['tool_availability_usage']
    add('### Availability versus gold selection\n\n'+markdown_table(sorted(usage, key=lambda r: -r['gold_calls']), ['tool', 'available_samples', 'runtime_available_samples', 'selected_samples', 'gold_calls', 'selection_percent_when_available'], 20)+
        '\n\navailable_samples counts explicit input permission; runtime_available_samples adds source-defined system exceptions. selected_samples counts at least one gold use, while gold_calls counts all occurrences. The selection percentage uses **runtime availability** as denominator. This is observational use frequency, not a calibrated probability of correct tool choice. Full catalog coverage, including unselected tools, is in the CSV.')
    add('## 8. Difficulty Analysis\n\nThis is a transparent engineering-risk heuristic, **not annotated task difficulty or measured model error**. One point each is assigned for: ≥2 calls; ≥4 calls; ≥2 gold apps; nonempty history; any earlier-result string match; supplied x-ref or ID argument; any time/date marker; ≥40 whitespace tokens; current audio ≥15 seconds; any read-before-write; any ask_missing_info. Scores 0–2 are simple, 3–5 moderate, and 6–11 complex. Features overlap and depend partly on gold labels, so this is a retrospective characterization, not an inference classifier.\n\n'+
        markdown_table(count_rows(s['difficulty_counts'], 'category', n), ['category', 'count', 'percent'])+
        '\n\n### Associations (Spearman rank correlation, descriptive only)\n\n'+
        markdown_table(s['correlations_spearman'], ['a', 'b', 'spearman_rho', 'n'])+
        '\n\nThe score includes tool count, history, timing and length, so correlations with those dimensions are partly mechanical. No significance tests or causal claims are intended. The per-group table reports duration, text length, history, calls, arguments and apps by risk category. Common task patterns are directly listed in the sequence table. Rare sequences are retained in the complete table and should be reviewed as stress cases, without treating every unique sequence as a distinct semantic task.\n\nDeterministic Python can enforce schemas/allowlists, normalize unambiguous numbers, calculate dates against an explicit timezone, and maintain the tool loop. LLM reasoning is a candidate for paraphrase-to-tool mapping, ellipsis, multi-intent decomposition, ambiguous contact/entity choices, corrections and result-conditioned planning. These assignments are recommendations to test, not established sufficiency claims. ASR substitutions in person references, amounts, negation, device IDs and dates may change actions even when general transcript quality is high.')
    add('## 9. Train/Public-Test Structural Comparison\n\n'+(public_report(s, public) if public else 'Skipped by configuration. No public-test labels, transcripts, pseudo-labels, or per-request intent analysis were used.'))
    add('''## 10. Implications for Our AI Architecture

1. **Evaluation first.** Use training gold transcripts to isolate agent behavior from ASR, then compare the same pipeline with ASR text. Score tool-name sequences, schema validity, argument correctness and valid termination separately. Official execution/state scoring is a future isolated evaluation stage and must honor one ToolEnv per sample; this exploration did not run it.
2. **ASR benchmarking.** Benchmark candidate Vietnamese-capable systems on the development manifest only, using documented local deployment and competition rules. Measure normalized WER/CER plus time/number/negation/entity preservation and downstream action success. Include current and historical user audio while tracking their separate distributions. No particular model is selected here and no model-performance estimate is available.
3. **Constrained tool selection.** Build per-sample schemas from the installed-tool allowlist plus the documented always-available system tool. Preserve read/write/meta distinctions, optional fields, enums and x-one-of alternatives. Use catalog descriptions and compact schemas; evaluate the cost of narrowing candidates without hiding required reads.
4. **Argument grounding.** Separate transcript extraction from entity lookup. Resolve x-ref IDs through permitted read calls and actual returned objects; retain a working entity table. Validate types and constraints before a call. Do not invent identifiers or silently coerce ambiguous numbers.
5. **Result-aware execution loop.** Feed actual tool results into planning, track applied writes, and stop after satisfying the final request. Repeating a write after a retry can duplicate actions. History summarizes already-applied actions; never replay it. sys.ask_missing_info and zero-call behavior need explicit handling.
6. **Time and context layer.** Parse relative time against the supplied current_time and timezone, preserving calendar versus duration semantics. Use only authorized profile/location context, and ask when a necessary referent remains ambiguous.
7. **Diagnostics.** Keep failure breakdowns by tool family, trajectory length, history, time/number markers, schema constraints, repeated tools and multi-app behavior. Avoid a single average that conceals rare write failures.''')
    add('## 11. Recommended Validation Strategy\n\n'+
        f"Seed **{SEED}**; training data only. The suggested 1,200/200/200 requires more examples than the observed {n:,}, so the proposal uses 75% / 12.5% / remainder targets, preserving the candidate proportions.\n\n"+
        markdown_table([{'partition': k, **v} for k, v in split_s.items()], ['partition', 'target_samples', 'actual_samples', 'unique_gold_tools', 'history_samples'])+
        f"\n\nThe executed duplicate screen found {s['duplicate_analysis']['nonexact_unique_text_pair_count']} nonexact similar normalized-text pairs, affecting {s['duplicate_analysis']['samples_in_components_with_nonexact_pair']} samples in their connected components. Across exact and near matches there are {s['duplicate_analysis']['groups']} groups; {s['duplicate_analysis']['non_singleton_groups']} are non-singleton and the largest contains {s['duplicate_analysis']['largest_group']} samples. Action-pattern audit: {s['action_pattern_overlap']['distinct_patterns']} distinct tool-name sequences, {s['action_pattern_overlap']['patterns_in_multiple_partitions']} present in multiple partitions, and {s['action_pattern_overlap']['patterns_in_all_three']} present in all three.\n\n"+
        '\n\nGrouping uses normalized exact strings and near-duplicate connected components: normalized character 5-shingle Jaccard ≥0.80, length ratio ≥0.80, and at least 20 characters for nonexact comparisons. Short strings group only by exact normalization. Pair detection compares distinct normalized utterances; transitive components can contain pairs below the threshold. No raw utterances or fingerprints are exported. Largest groups and near-pair statistics are in summary.json; pair sample IDs and similarity values are in the CSV.\n\nGroups are assigned greedily in descending size with seeded ties, balancing size and observed tool/task features. Tools present in fewer than three independent groups preferentially retain those groups in the training partition. Rare-tool coverage in every partition is therefore not guaranteed; exact target sizes can vary when a group does not fit. See split_coverage.csv for all missing tools and partition distributions.\n\nIdentical action sequences are analyzed separately, not globally forced into one group: common generic sequences could create giant components and remove tool coverage. Therefore action-template overlap can remain; report it, and use an optional stricter action-template challenge set for compositional generalization. Semantic paraphrases below the lexical threshold and recurring users/scenarios not identifiable from permitted fields can also leak. Grouped splitting reduces lexical leakage without proving independence.\n\nThe manifests are proposed, not a claim of an optimal split. Review group sizes and coverage before locking. Iterate only on dev, keep holdout unchanged for periodic checks, and version decisions against the fixed seed. Do not tune prompts, models or hyperparameters on public-test observations.')
    add('''## 12. Prioritized Next Experiments

| Rank / owner | Hypothesis | Required implementation | Evaluation metric | Expected benefit | Risk / limitation |
| --- | --- | --- | --- | --- | --- |
| 1 / evaluation engineer | A transcript-oracle baseline exposes agent errors independently of ASR | Lock grouped manifests; create gold comparison and failure taxonomy; prepare isolated official-runtime evaluation respecting one environment per sample | Tool sequence exact match; argument/schema validity; official execution/state outcome when permitted | A trustworthy baseline for every later experiment | Labels can encode one of several valid trajectories; exact sequence match is incomplete |
| 2 / agent and tool engineer | Enforced availability and catalog validation prevent avoidable calls | Build tool-schema compiler, allowlist gating, x-one-of validation, result-aware call loop, valid termination | Unpermitted/invalid call rate; task completion; call count | Prevent structural execution mistakes across families | Local static checks alone do not establish runtime correctness |
| 3 / speech engineer | Entity/number/time preservation predicts action success better than aggregate WER alone | Local ASR benchmark on dev; normalized WER/CER and sensitive-span annotation; historical-audio handling | WER/CER; sensitive-span accuracy; oracle-to-ASR action-success gap; latency | Select speech processing on execution-relevant evidence | Manual span annotation is required; no speaker/region labels are available |
| 4 / grounding and context engineer | Explicit entity memory and time resolution improve multi-step/contextual cases | x-ref result grounding; current-time/timezone module; history summaries; safe missing-info path | Reference-ID accuracy; date/time argument accuracy; history ablation; multi-step task success | Target contextual and result-conditioned failure modes | Lexical heuristics used here do not establish genuine dependency |
| 5 / modeling and data engineer | Balanced training and targeted stress cases improve rare/complex tools without harming common ones | Train-only sampling/augmentation after baseline; tool-family tests for top tools and schema-heavy/rare writes; compare prompt versus fine-tuning only after evidence | Macro tool-family success; locked-holdout task success; rare-tool and write-error rates | Better coverage and controlled model-development choices | Small rare-tool counts give uncertain estimates; generated cases can introduce artifacts |

These priorities are recommendations grounded in call complexity, catalog constraints, history, and argument/dependency measurements above. Expected benefits are qualitative hypotheses, not fabricated performance gains. All five roles can work against the same fixed manifests and metrics; model choice remains an experiment.''')
    add('''## 13. Limitations and Open Questions

- No ASR inference, manual listening study, speaker analysis, noise analysis, regional-accent labeling, transcript-error estimation, LLM evaluation or fine-tuning ran. These require separate evidence.
- No ToolEnv was constructed. Gold calls and histories were never executed. final_state values and all environment-file contents were excluded; state correctness, contact uniqueness, real runtime referential integrity and official task success remain unknown.
- Lexical markers, near-duplicate groups, result-string matches and risk categories are exploratory heuristics. They are neither semantic annotations nor causal dependency labels.
- Static schema checks cover a conservative subset. The inventory preserves other constraints for future validators; no claim of full official-validator equivalence is made.
- Files not explicitly referenced by JSONL were not inventoried. Duplicate audio content, recording sessions and speaker leakage cannot be established from path uniqueness.
- The README is the available competition-specific authority. It does not include a full scoring specification, deployment budget or later private-test rules; those should be obtained before final architecture decisions.
- Public-test results, when enabled, are aggregate structural observations only and must not drive model/prompt/hyperparameter selection.
- **Initial discovery deviation:** before the attachment restrictions were read, the interactive session issued `rg --files -g AGENTS.md` from the root. It returned no matches and did not read environment-file contents, but it could traverse restricted directories. Thus the entire interactive session cannot be certified as having avoided directory traversal. Subsequent discovery used explicit paths; the executed analysis includes an audit guard blocking environment/state access. This deviation is recorded rather than hidden.''')
    add(f'''## Appendix: Reproducibility and Artifacts

Run from the dataset root in PowerShell (single line):

```powershell
python scripts/explore_lifemate.py --root train+public_test --split train --output artifacts/eda --report artifacts/dataset_exploration.md --compare-public-test --seed {SEED}
```

Training-only run: omit --compare-public-test. Tests use Python’s standard unittest because pytest is unavailable:

```powershell
python -m unittest discover -s tests -p test_explore_lifemate.py -v
```

Dependencies are Python, NumPy and Matplotlib; Pandas is not required. Versions: Python {platform.python_version()}, NumPy {np.__version__}, Matplotlib {matplotlib.__version__}. Seeded grouping/allocation and no stochastic model calls make statistics and manifests deterministic. Figure metadata/bytes may vary by library version; use tables for numerical comparisons.

Summary: [eda/summary.json](eda/summary.json). Verification and access audit: [eda/verification.json](eda/verification.json), [eda/access_audit.json](eda/access_audit.json). Table inventory: [eda/tables/index.json](eda/tables/index.json). Figure source counts: [eda/figures/manifest.json](eda/figures/manifest.json). Split IDs: [train](eda/validation_split/train_ids.txt), [dev](eda/validation_split/dev_ids.txt), [holdout](eda/validation_split/holdout_ids.txt).

Only local derived aggregates and ID/feature manifests were written. Source JSONL/catalog/WAV files were opened read-only. No Git commit, remote upload or external model invocation was performed. The audit covers pipeline file events, not pre-pipeline interactive commands. No raw transcript, argument/result payload, profile value, phone number, address, coordinate or final-state value is emitted by the pipeline.

### Figures and captions
''')
    for fig in FIGURES:
        add(f"![{fig['file']}](eda/figures/{fig['file']})\n\n{fig['caption']} Source observations: {fig['observations']:,}.")
    return '\n\n'.join(sections)+'\n'


def public_report(train, public):
    p = public['summary']
    rows = []
    for name, a, b in [('Samples', train['samples'], p['samples']),
                       ('Nonempty history samples', train['history_nonempty'], p['history_nonempty']),
                       ('Mean history turns', train['numeric']['history_turns']['mean'], p['numeric']['history_turns']['mean']),
                       ('Mean installed apps', train['numeric']['installed_apps']['mean'], p['numeric']['installed_apps']['mean']),
                       ('Mean permitted tools', train['numeric']['available_tools']['mean'], p['numeric']['available_tools']['mean']),
                       ('Median current duration (s)', train['audio']['current_duration_seconds']['median'], p['audio']['current_duration_seconds']['median']),
                       ('P95 current duration (s)', train['audio']['current_duration_seconds']['p95'], p['audio']['current_duration_seconds']['p95']),
                       ('Unique referenced WAVs', train['audio']['unique_referenced_files'], p['audio']['unique_referenced_files'])]:
        rows.append({'measure': name, 'train': a, 'public_test': b})
    return ('**Unlabeled distribution audit, not a performance evaluation.** This ran only after training analysis and manifest construction. No public audio transcription, intent/request inspection, pseudo-labeling, evaluation, or tuning took place.\n\n'+
            markdown_table(rows, ['measure', 'train', 'public_test'])+'\n\n'+
            f"Public catalog: {public['catalog']['apps']} apps, {public['catalog']['tools']} tools, schema version {public['catalog']['schema_version']}. Catalog comparison: {len(public['only_train_tools'])} train-only tools, {len(public['only_public_tools'])} public-only tools, {len(public['changed_shared_schemas'])} shared tools with different parameter schemas, {len(public['changed_shared_definitions'])} shared tools with any definition difference. Full structural differences and distributions are in public_test_structural_summary.json and public_* CSVs.\n\n"+
            'Catalog differences describe scope; they do not establish model generalization or success. Public context values are never exported. No training partitions or model recommendations were altered in response to this audit.')


def verify(rows, s, tools, inventory, allocations, groups, tables, output):
    ids = [r['sample_id'] for r in rows]
    checks = {'sample_id_unique': len(ids) == len(set(ids)),
              'frequency_equals_calls': sum(s['gold_tool_frequency'].values()) == sum(r['tool_calls'] for r in rows),
              'kind_equals_calls': sum(s['gold_kind_frequency'].values()) == s['gold_total_calls'],
              'app_equals_calls': sum(s['gold_app_frequency'].values()) == s['gold_total_calls'],
              'catalog_kind_equals_tools': len(inventory) == len(tools),
              'transitions_equal_sum_l_minus_1': sum(r['count'] for r in s['kind_transitions']) == sum(max(r['tool_calls']-1, 0) for r in rows),
              'trajectory_classes_cover_samples': s['zero_calls']+s['single_call']+s['multi_call'] == len(rows),
              'difficulty_covers_samples': sum(s['difficulty_counts'].values()) == len(rows),
              'split_index_disjoint_and_cover': sorted(i for part in allocations for i in part) == list(range(len(rows))),
              'split_id_disjoint': len(set(ids)) == sum(len({ids[i] for i in part}) for part in allocations),
              'pipeline_restricted_access_attempts_zero': not DENIED}
    checks['availability_covers_gold_selection'] = all(r['selected_samples'] <= r['runtime_available_samples'] for r in tables['tool_availability_usage'])
    checks['accepted_result_counts_equal_gold'] = sum(s['result_status'].get(k, 0) for k in ('ok_true', 'ok_false', 'ok_absent_or_nonboolean', 'nonobject')) == s['gold_total_calls']
    membership = {i: p for p, part in enumerate(allocations) for i in part}
    checks['duplicate_groups_kept_together'] = all(len({membership[i] for i in g}) == 1 for g in groups)
    expected = {'01_audio_duration': s['audio']['current_duration_seconds']['n'],
                '02_history_length': len(rows), '03_gold_calls': len(rows),
                '04_top_gold_tools': sum(v for _, v in COUNTERS(s['gold_tool_frequency']).most_common(20)),
                '05_tool_kinds': s['gold_total_calls'], '06_gold_apps': s['gold_total_calls'],
                '07_availability': len(rows), '08_required_arguments': len(tools),
                '09_trajectory_classes': len(rows), '10_difficulty': len(rows),
                '11_transcript_actions': len(rows),
                '12_kind_transitions': sum(max(r['tool_calls']-1, 0) for r in rows),
                '13_catalog_kinds': len(tools), '14_language_markers': sum(s['language_marker_counts'].values()),
                '15_duration_actions': s['audio']['current_duration_seconds']['n'],
                '16_audio_roles': s['audio']['current_duration_seconds']['n']+s['audio']['history_duration_seconds']['n']}
    checks['plot_totals_match_summary'] = all(expected[Path(f['file']).stem] == f['observations'] for f in FIGURES)
    checks['figure_files_exist'] = all((output/'figures'/f['file']).is_file() for f in FIGURES)
    checks['table_files_exist'] = all((output/'tables'/(key+'.csv')).is_file() for key in tables)
    return {'checks': checks, 'all_passed': all(checks.values()), 'figure_count': len(FIGURES),
            'table_count': len(tables), 'privacy': 'No raw utterances, profile/location values, argument/result payloads, or state values are exported.'}


def check_report_privacy(root, report_text):
    """Conservative direct-value check; never emit matched sensitive values."""
    sensitive = set()
    with safe_path(root/'train/train.jsonl').open(encoding='utf-8-sig') as f:
        for line in f:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get('context'), dict):
                continue
            profile = record['context'].get('user_profile', {})
            if isinstance(profile, dict):
                sensitive.update(v.strip().casefold() for v in profile.values() if isinstance(v, str) and len(v.strip()) >= 8)
    lowered = report_text.casefold()
    count = sum(bool(re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)', lowered)) for value in sensitive)
    phones = len(re.findall(r'(?<![\d.])(?:\+84|0)[1-9][0-9]{8,9}(?!\d)', report_text))
    return {'long_profile_values_checked': len(sensitive), 'profile_value_matches_in_report': count,
            'phone_like_strings_in_report': phones,
            'scope': 'Exact profile values >=8 characters, plus Vietnamese phone-like strings; short names, paraphrases and other PII require code review.'}


def main(argv=None):
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='train+public_test')
    parser.add_argument('--split', choices=['train'], default='train', help='Supervised analysis is train-only')
    parser.add_argument('--output', default='artifacts/eda')
    parser.add_argument('--report', default='artifacts/dataset_exploration.md')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--compare-public-test', action='store_true')
    args = parser.parse_args(argv)
    SEED = args.seed
    install_guard()
    root = discover_root(args.root)
    output = safe_path(args.output)
    report_path = safe_path(args.report)
    # Prevent accidental output placement anywhere inside source split directories.
    for p in (output, report_path):
        if any(p.is_relative_to(root/split) for split in ('train', 'public_test')):
            raise ValueError('Outputs must be outside source split directories')
    for folder in (output, output/'tables', output/'figures', output/'validation_split', report_path.parent):
        folder.mkdir(parents=True, exist_ok=True)
    (root/('README_DATA.md' if (root/'README_DATA.md').is_file() else 'README.md')).read_text(encoding='utf-8-sig')
    version = (root/'VERSION').read_text(encoding='utf-8-sig') if (root/'VERSION').is_file() else 'Unavailable'
    tools, inventory, apps, parameters, catalog_s = read_catalog(root, 'train')
    rows, texts, s, tables = analyze_split(root, 'train', tools)
    print('Training measurements complete; grouping duplicates and constructing training-only partitions.', flush=True)
    groups, pairs = duplicate_groups(texts)
    allocations, membership, split_s = make_splits(rows, groups, args.seed)
    pair_members = {i for a, b, _ in pairs for i in groups[next(k for k,g in enumerate(groups) if a in g)]}
    s['duplicate_analysis'] = {'definition': 'NFC/casefold/punctuation normalization; char-5 Jaccard>=0.80; length ratio>=0.80; >=20 chars',
                               'nonexact_unique_text_pair_count': len(pairs), 'samples_in_components_with_nonexact_pair': len(pair_members),
                               'groups': len(groups), 'non_singleton_groups': sum(len(g) > 1 for g in groups),
                               'largest_group': max(map(len, groups), default=0),
                               'group_size_distribution': dict(COUNTERS(map(len, groups)))}
    for part, indexes in zip(('train', 'dev', 'holdout'), allocations):
        (output/'validation_split'/(part+'_ids.txt')).write_text(''.join(rows[i]['sample_id']+'\n' for i in sorted(indexes, key=lambda i: rows[i]['sample_id'])), encoding='utf-8')
    selected_samples = COUNTERS(t for r in rows for t in set(r['tool_sequence']))
    integrity_checks = []
    for check in ('malformed_json_lines', 'nonobject_records', 'call_result_length_mismatch',
                  'unknown_available_tool_occurrences', 'unknown_gold_tool_calls', 'gold_tool_not_permitted_calls'):
        denominator = s['gold_total_calls'] if check.endswith('_calls') else (sum(s['available_tool_frequency'].values()) if check == 'unknown_available_tool_occurrences' else len(rows))
        integrity_checks.append({'check': check, 'anomaly_count': s['quality'].get(check, 0), 'denominator': denominator, 'percent': pct(s['quality'].get(check, 0), denominator)})
    for key in ('id', 'audio', 'context', 'history', 'audio_text', 'label'):
        for kind in ('missing_', 'wrong_type_'):
            check = kind+key
            integrity_checks.append({'check': check, 'anomaly_count': s['quality'].get(check, 0), 'denominator': len(rows), 'percent': pct(s['quality'].get(check, 0), len(rows))})
    for key in ('tool_calls', 'tool_results', 'final_state'):
        for kind in ('label_missing_', 'label_wrong_type_'):
            check = kind+key
            integrity_checks.append({'check': check, 'anomaly_count': s['quality'].get(check, 0), 'denominator': len(rows), 'percent': pct(s['quality'].get(check, 0), len(rows))})
    tables.update({'integrity_checks': integrity_checks, 'catalog_inventory': inventory, 'catalog_apps': apps, 'catalog_parameters': parameters,
                   'language_heuristics': [{'marker': k, 'regex': v, 'samples': s['language_marker_counts'][k], 'percent': pct(s['language_marker_counts'][k], len(rows))} for k, v in PATTERNS.items()],
                   'sample_features': [{k: v for k, v in r.items() if k != 'tool_sequence'} for r in rows],
                   'installed_apps': count_rows(s['installed_app_frequency'], 'app', len(rows)),
                   'gold_tool_frequency': count_rows(s['gold_tool_frequency'], 'tool', s['gold_total_calls']),
                   'gold_app_frequency': count_rows(s['gold_app_frequency'], 'app', s['gold_total_calls']),
                   'catalog_kind_frequency': count_rows(catalog_s['kind_counts'], 'kind', catalog_s['tools']),
                   'gold_kind_frequency': count_rows(s['gold_kind_frequency'], 'kind', s['gold_total_calls']),
                   'near_duplicate_pairs': [{'sample_id_a': rows[a]['sample_id'], 'sample_id_b': rows[b]['sample_id'], 'jaccard': sim} for a, b, sim in pairs],
                   'split_coverage': [{'partition': p, **data} for p, data in split_s.items()],
                   'split_tool_coverage': [{'tool': t, **{p: sum(t in rows[i]['tool_sequence'] for i in allocations[k]) for k, p in enumerate(('train', 'dev', 'holdout'))}} for t in tools],
                   'correlations': s['correlations_spearman']})
    tables['tool_availability_usage'] = [{'tool': t, 'app': tool['app_id'], 'kind': tool.get('kind'),
                'available_samples': s['available_tool_frequency'].get(t, 0),
                                         'runtime_available_samples': s['runtime_available_tool_frequency'].get(t, 0),
                                         'selected_samples': selected_samples[t], 'gold_calls': s['gold_tool_frequency'].get(t, 0),
                                         'selection_percent_when_available': pct(selected_samples[t], s['runtime_available_tool_frequency'].get(t, 0))} for t, tool in tools.items()]
    tables['difficulty_profiles'] = [{'difficulty': d, 'samples': len(sub), **{k+'_mean': stats(r[k] for r in sub).get('mean') for k in ('duration_seconds', 'history_turns', 'tool_calls', 'transcript_tokens', 'trajectory_apps', 'required_arguments')}} for d in ('simple', 'moderate', 'complex') if (sub := [r for r in rows if r['difficulty'] == d])]
    action_partitions = collections.defaultdict(set)
    for i, r in enumerate(rows):
        action_partitions[r['tool_sequence']].add(membership[i])
    s['action_pattern_overlap'] = {'distinct_patterns': len(action_partitions),
                                   'patterns_in_multiple_partitions': sum(len(v) > 1 for v in action_partitions.values()),
                                   'patterns_in_all_three': sum(len(v) == 3 for v in action_partitions.values())}
    plots(rows, s, inventory, output/'figures', tables)
    public = None
    if args.compare_public_test:
        print('Training analysis and split generation complete. Starting aggregate-only public-test audit.', flush=True)
        pt, pi, pa, pp, pc = read_catalog(root, 'public_test')
        _, _, ps, p_tables = analyze_split(root, 'public_test', pt, supervised=False)
        public = {'summary': ps, 'catalog': pc,
                  'only_train_tools': sorted(set(tools)-set(pt)), 'only_public_tools': sorted(set(pt)-set(tools)),
                  'changed_shared_schemas': sorted(t for t in tools.keys() & pt.keys() if tools[t].get('parameters') != pt[t].get('parameters')),
                  'changed_shared_definitions': sorted(t for t in tools.keys() & pt.keys() if tools[t] != pt[t])}
        for key, values in p_tables.items():
            tables['public_'+key] = values
        tables['public_installed_apps'] = count_rows(ps['installed_app_frequency'], 'app', ps['samples'])
        tables['public_available_tools'] = count_rows(ps['available_tool_frequency'], 'tool', ps['samples'])
        tables['public_catalog_apps'] = pa
        tables['public_catalog_inventory'] = pi
        write_json(output/'public_test_structural_summary.json', public)
        (output/'public_test_structural_summary.md').write_text(public_report(s, public)+'\n', encoding='utf-8')
    for name, values in tables.items():
        write_csv(output/'tables'/(name+'.csv'), values)
    write_json(output/'tables/index.json', {k+'.csv': len(v) for k, v in tables.items()})
    write_json(output/'figures/manifest.json', FIGURES)
    write_json(output/'summary.json', {'train': s, 'catalog': catalog_s, 'validation_split': split_s,
                                     'public_test': public, 'seed': args.seed,
                                     'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}})
    report_text = build_report(root, output, s, catalog_s, rows, inventory, tables, split_s, public, version)
    relative_output = Path(os.path.relpath(output, report_path.parent)).as_posix()
    report_text = report_text.replace('(eda/', '('+relative_output+'/')
    dataset_readme = root/('README_DATA.md' if (root/'README_DATA.md').is_file() else 'README.md')
    report_text = report_text.replace('(../README.md)', '('+Path(os.path.relpath(dataset_readme, report_path.parent)).as_posix()+')')
    runtime_prefix = Path(os.path.relpath(root/'train/zaic_tools', report_path.parent)).as_posix()
    report_text = report_text.replace('(../train/zaic_tools/', '('+runtime_prefix+'/')
    report_path.write_text(report_text, encoding='utf-8')
    verification = verify(rows, s, tools, inventory, allocations, groups, tables, output)
    verification['privacy_scan'] = check_report_privacy(root, report_text)
    verification['checks']['report_privacy_scan'] = verification['privacy_scan']['profile_value_matches_in_report'] == 0 and verification['privacy_scan']['phone_like_strings_in_report'] == 0
    verification['checks']['report_exists'] = report_path.is_file()
    verification['checks']['manifests_exist'] = all((output/'validation_split'/(p+'_ids.txt')).is_file() for p in ('train', 'dev', 'holdout'))
    verification['all_passed'] = all(verification['checks'].values())
    write_json(output/'verification.json', verification)
    write_json(output/'access_audit.json', {'guard': 'Python audit hook blocks open/listdir/scandir on any *_environment component or state.json; explicit referenced-audio paths only',
                                         'restricted_attempts_blocked': len(DENIED), 'file_open_events_by_suffix': dict(ACCESS),
                                         'ToolEnv_instances': 0, 'gold_or_history_executions': 0,
                                         'scope': 'Pipeline only; pre-pipeline recursive filename discovery deviation documented in report',
                                         'network_or_external_model_calls': 0})
    print(json.dumps({'samples': s['samples'], 'gold_calls': s['gold_total_calls'], 'figures': len(FIGURES),
                      'tables': len(tables), 'verification_passed': verification['all_passed'], 'report': str(report_path)}, indent=2))
    return 0 if verification['all_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
