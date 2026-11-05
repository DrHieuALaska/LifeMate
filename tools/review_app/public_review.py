"""Read downloaded Kaggle predictions and persist human annotations separately."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import threading
import wave
try:
    from .dataset_source import DatasetSource
except ImportError:
    from dataset_source import DatasetSource

KERNEL = 'drhieualaska/20-sample-public-test'
JUDGEMENTS = {'', 'correct', 'acceptable', 'wrong', 'unsure'}
ISSUES = {'asr', 'name', 'english', 'number', 'time', 'negation', 'tool', 'arguments', 'context', 'missing', 'other'}


def jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate sample ID: ' + path.name)
    return result


class PublicReview:
    def __init__(self, root, dataset, annotation_path=None, source=None):
        self.root, self.dataset = Path(root), Path(dataset)
        self.dataset_source = DatasetSource(self.dataset)
        self.source = Path(source) if source else self.root / 'imports/kaggle_20_sample_public_test/lifemate_benchmark_634'
        self.asr = jsonl(self.source / 'batch_asr_transcripts.jsonl')
        self.predictions = jsonl(self.source / 'BENCHMARK_NOT_FOR_SUBMISSION.jsonl')
        selected = json.loads((self.source / 'selected_ids.json').read_text(encoding='utf-8'))
        if len(selected) != len(set(selected)) or set(selected) != set(self.asr) or set(selected) != set(self.predictions):
            raise ValueError('Kaggle selected IDs, ASR and reasoning outputs do not match')
        self.ids = sorted(selected)
        self.numbers = {uid: i + 1 for i, uid in enumerate(self.ids)}
        source_bytes = b''.join((self.source / p).read_bytes() for p in (
            'selected_ids.json', 'batch_asr_transcripts.jsonl', 'BENCHMARK_NOT_FOR_SUBMISSION.jsonl'))
        self.fingerprint = hashlib.sha256(source_bytes).hexdigest()
        self.audio_members, self.rows, self.metrics, self.durations = {}, {}, {}, {}
        with self.dataset_source as archive:
            manifest = archive.manifest('public_test')
            parent = PurePosixPath(manifest).parent
            self.prefix = '' if str(parent) == '.' else str(parent) + '/'
            originals = {r['id']: r for r in (json.loads(line) for line in
                archive.read(manifest).decode('utf-8').splitlines() if line.strip())}
            self.original_rows = {uid: originals[uid] for uid in self.ids}
            catalog = json.loads(archive.read(self.prefix + 'public_test_tools_catalog.json'))
            self.catalog = {t['name']: t for app in catalog['apps'] for t in app['tools']}
            names = set(archive.namelist())
            for uid in self.ids:
                row = json.loads(json.dumps(originals[uid]))
                if row.get('label'):
                    raise ValueError('Unexpected gold label in public sample')
                self.audio_members[uid] = self.prefix + row['audio']
                asr_history = iter(self.asr[uid].get('history_audio_text', []))
                for index, turn in enumerate(row.get('history', [])):
                    if turn.get('role') == 'user' and turn.get('audio'):
                        turn['asr_text'] = next(asr_history)
                        turn['audio_key'] = uid + '_history_' + str(index)
                        self.audio_members[turn['audio_key']] = self.prefix + turn['audio']
                if next(asr_history, None) is not None:
                    raise ValueError('Extra history ASR: ' + uid)
                journal = json.loads((self.source / 'metrics/journal' / (uid + '.json')).read_text(encoding='utf-8'))
                if journal['submission'] != self.predictions[uid]:
                    raise ValueError('Journal prediction mismatch: ' + uid)
                self.metrics[uid] = journal['metrics']
                self.rows[uid] = row
            if any(member not in names for member in self.audio_members.values()):
                raise ValueError('Missing public-test audio in selected dataset source')
        self.annotation_path = Path(annotation_path) if annotation_path else self.root / 'data/manual_public_labels.json'
        self.lock = threading.RLock()
        self.notes = {}
        if self.annotation_path.exists():
            saved = json.loads(self.annotation_path.read_text(encoding='utf-8'))
            if saved['source_fingerprint'] != self.fingerprint:
                raise ValueError('Annotations belong to a different set of Kaggle predictions')
            self.notes = saved['annotations']
            if set(self.notes) - set(self.ids):
                raise ValueError('Unknown annotated ID')

    def index(self):
        with self.lock:
            samples = []
            for uid in self.ids:
                note = self.notes.get(uid, {})
                samples.append(dict(id=uid, conversation_no=self.numbers[uid], asr_text=self.asr[uid]['audio_text'],
                    reviewed=note.get('reviewed', False), asr_judgement=note.get('asr_judgement', ''),
                    reasoning_judgement=note.get('reasoning_judgement', ''),
                    status=self.metrics[uid].get('status', ''),
                    search_text=' '.join((uid, self.asr[uid]['audio_text'],
                        json.dumps(self.predictions[uid], ensure_ascii=False), note.get('comment', ''),
                        note.get('corrected_transcript', ''))).lower()))
            return dict(kernel=KERNEL, source_fingerprint=self.fingerprint, samples=samples,
                        n=len(samples), reviewed=sum(r['reviewed'] for r in samples),
                        has_official_gold=False)

    def detail(self, uid):
        row = self.rows[uid]
        groups = []
        for app, entry in row['context']['installed_apps'].items():
            groups.append(dict(app=app, tools=[dict(self.catalog.get(name, {}), name=name) for name in entry['tools']]))
        if uid not in self.durations:
            with self.dataset_source as archive:
                with wave.open(io.BytesIO(archive.read(self.audio_members[uid])), 'rb') as audio:
                    self.durations[uid] = audio.getnframes() / audio.getframerate()
        allowed = {t['name'] for group in groups for t in group['tools']}
        meta_tools = [t for t in self.catalog.values() if t.get('kind') == 'meta']
        with self.lock:
            note = dict(self.notes.get(uid, {}))
        return dict(id=uid, conversation_no=self.numbers[uid], conversation_count=len(self.ids),
                    asr=self.asr[uid], prediction=self.predictions[uid], metrics=self.metrics[uid],
                    duration_s=self.durations[uid], context=row['context'], history=row.get('history', []),
                    allowed_tool_groups=groups, meta_tools=meta_tools,
                    tools_not_in_installed_apps=[t['name'] for t in self.predictions[uid].get('tool_calls', []) if t['name'] not in allowed],
                    annotation=note, source_fingerprint=self.fingerprint, has_official_gold=False)

    def environment(self, uid):
        if uid not in self.rows:
            raise KeyError(uid)
        with self.dataset_source as archive:
            return json.loads(archive.read(self.prefix + 'public_test_environment/' + uid + '/state.json'))

    def full_input(self, uid):
        original = self.original_rows[uid]
        agent_record = json.loads(json.dumps(original))
        agent_record['audio_text'] = self.asr[uid]['audio_text']
        users = [turn for turn in agent_record.get('history', []) if turn.get('role') == 'user']
        history_text = self.asr[uid]['history_audio_text']
        if len(users) != len(history_text):
            raise ValueError('Cannot reconstruct agent history')
        for turn, spoken in zip(users, history_text):
            turn['audio_text'] = spoken
        allowed = {name for app in original['context']['installed_apps'].values() for name in app['tools']}
        return dict(
            id=uid,
            agent_record_after_asr=agent_record,
            original_manifest_record=original,
            initial_environment_state=self.environment(uid),
            allowed_tool_schemas=[self.catalog[name] for name in sorted(allowed) if name in self.catalog],
            system_meta_tool_schemas=[t for t in self.catalog.values() if t.get('kind') == 'meta'],
            asr_cache_record=self.asr[uid],
            source=dict(
                original_manifest=self.prefix + 'public_test.jsonl',
                initial_environment=self.prefix + 'public_test_environment/' + uid + '/state.json',
                agent_record_reconstruction='run_benchmark.py lines 277-280: inject cached audio_text for current and history user turns',
                exact_llm_prompt_dump_available=False,
                raw_runtime_tool_observations_saved=False),
            explanation='The agent record is reconstructed from the original public record and saved ASR cache. '
                        'The environment state belongs to the tool environment; it is not assumed to be included '
                        'in full in the LLM prompt. Raw runtime tool observations and the exact LLM prompt were not saved.')

    def save(self, uid, payload):
        if uid not in self.rows:
            raise ValueError('Unknown public sample')
        if payload.get('source_fingerprint') != self.fingerprint:
            raise ValueError('Predictions changed; reload before labeling')
        note = payload.get('annotation')
        fields = {'corrected_transcript', 'asr_judgement', 'reasoning_judgement', 'expected_output', 'issues', 'comment', 'reviewed'}
        if not isinstance(note, dict) or set(note) - fields:
            raise ValueError('Invalid annotation fields')
        clean = {}
        for key in ('corrected_transcript', 'comment'):
            value = note.get(key, '')
            if not isinstance(value, str) or len(value) > 20000:
                raise ValueError('Invalid text: ' + key)
            clean[key] = value
        for key in ('asr_judgement', 'reasoning_judgement'):
            value = note.get(key, '')
            if value not in JUDGEMENTS:
                raise ValueError('Invalid judgement')
            clean[key] = value
        expected = note.get('expected_output')
        if expected is not None:
            if not isinstance(expected, dict) or not isinstance(expected.get('tool_calls'), list):
                raise ValueError('Expected output must be a JSON object with a tool_calls list')
            if any(not isinstance(t, dict) or not isinstance(t.get('name'), str) or not isinstance(t.get('arguments'), dict)
                   for t in expected['tool_calls']):
                raise ValueError('Each expected tool call needs name and arguments')
        clean['expected_output'] = expected
        issues = note.get('issues', [])
        if not isinstance(issues, list) or any(not isinstance(x, str) or x not in ISSUES for x in issues):
            raise ValueError('Invalid issue tags')
        clean['issues'] = sorted(set(issues))
        if not isinstance(note.get('reviewed', False), bool):
            raise ValueError('reviewed must be boolean')
        clean['reviewed'] = note.get('reviewed', False)
        clean['updated_at'] = datetime.now(timezone.utc).isoformat()
        with self.lock:
            updated = dict(self.notes, **{uid: clean})
            content = json.dumps(dict(version=1, kernel=KERNEL, source_fingerprint=self.fingerprint,
                                      annotations=updated), ensure_ascii=False, indent=2)
            self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.annotation_path.with_suffix('.json.tmp')
            temporary.write_text(content + '\n', encoding='utf-8')
            if self.annotation_path.exists():
                backup = self.annotation_path.with_suffix('.json.bak')
                backup.write_bytes(self.annotation_path.read_bytes())
            temporary.replace(self.annotation_path)
            self.notes = updated
        return clean

    def export(self):
        with self.lock:
            return ''.join(json.dumps(dict(id=uid, conversation_no=self.numbers[uid], kernel=KERNEL,
                source_fingerprint=self.fingerprint, asr=self.asr[uid], prediction=self.predictions[uid],
                human_annotation=self.notes[uid]), ensure_ascii=False) + '\n'
                for uid in self.ids if uid in self.notes).encode('utf-8')
