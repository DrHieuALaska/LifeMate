"""Synthetic tests only. No private app states or real transcript values."""
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('explore_lifemate', Path(__file__).parents[1]/'scripts/explore_lifemate.py')
eda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eda)


class ExplorationTests(unittest.TestCase):
    def test_normalization_and_language(self):
        self.assertEqual(eda.normalize('  BẬT, đèn! '), 'bật đèn')
        f = eda.language_flags('Ngày mai đừng bật rồi tắt đèn.')
        self.assertTrue(f['relative_time'])
        self.assertTrue(f['negation'])
        self.assertTrue(f['conjunction'])

    def test_restricted_paths(self):
        for name in ('train/train_environment/item/anything.bin', 'other_environment/a.wav', 'state.json'):
            self.assertTrue(eda.restricted(name))
            with self.assertRaises(PermissionError):
                eda.safe_path(name)
        self.assertFalse(eda.restricted('train/audio/synthetic.wav'))

    def test_statistics_and_static_schema(self):
        self.assertEqual(eda.stats([]), {'n': 0})
        self.assertEqual(eda.stats([1, 2, 3])['median'], 2)
        schema = {'properties': {'n': {'type': 'integer', 'minimum': 1}, 'x': {'enum': ['on', 'off']}}, 'required': ['n']}
        self.assertIn('missing_required', eda.validate_arguments({}, schema))
        self.assertIn('wrong_type', eda.validate_arguments({'n': True}, schema))
        self.assertIn('outside_enum', eda.validate_arguments({'n': 1, 'x': 'bad'}, schema))
        self.assertEqual(eda.validate_arguments({'n': 1}, schema), [])

    def test_empty_history_missing_fields_zero_calls_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train/audio').mkdir(parents=True)
            with wave.open(str(root/'train/audio/synthetic.wav'), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x00\x00'*16000)
            records = [
                {'id': 'synthetic_a', 'audio': 'audio/synthetic.wav', 'audio_text': 'Bật đèn', 'context': {}, 'history': [], 'label': {'tool_calls': [], 'tool_results': [], 'final_state': {}}},
                {'id': 'synthetic_b', 'audio': 'audio/missing.wav', 'context': {}, 'history': []},
                {'id': 'synthetic_c', 'audio': '../train_environment/a.wav', 'audio_text': None, 'history': None, 'label': {'tool_calls': None}},
            ]
            (root/'train/train.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n{malformed}\n', encoding='utf-8')
            rows, texts, summary, tables = eda.analyze_split(root, 'train', {}, progress=False)
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary['zero_calls'], 3)
            self.assertEqual(summary['quality']['malformed_json_lines'], 1)
            self.assertEqual(summary['quality']['missing_audio_text'], 1)
            self.assertEqual(summary['quality']['wrong_type_history'], 1)
            self.assertEqual(summary['audio']['reference_status_counts']['missing_file'], 1)
            self.assertEqual(rows[0]['duration_seconds'], 1.0)
            self.assertTrue(any('reference' in status for status in summary['audio']['reference_status_counts']))

    def test_duplicates_and_grouped_split(self):
        texts = ['bật đèn phòng', 'bật đèn phòng', 'tắt quạt', 'xem lịch hôm nay']
        groups, pairs = eda.duplicate_groups(texts)
        self.assertIn([0, 1], groups)
        rows = [{'sample_id': f'synthetic_{i}', 'tool_sequence': ('mock.read',), 'difficulty': 'simple', 'history_turns': 0, 'tool_calls': 1} for i in range(4)]
        a, membership, s = eda.make_splits(rows, groups, seed=5)
        self.assertEqual(membership[0], membership[1])
        self.assertEqual(sorted(i for part in a for i in part), list(range(4)))
        self.assertEqual(a, eda.make_splits(rows, groups, seed=5)[0])

    def test_public_has_no_supervised_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'public_test').mkdir()
            item = {'id': 'synthetic_public', 'audio': 'audio/missing.wav', 'context': {}, 'history': [], 'audio_text': 'do not analyze', 'label': {'tool_calls': [{'name': 'forbidden'}]}}
            (root/'public_test/public_test.jsonl').write_text(json.dumps(item), encoding='utf-8')
            rows, texts, summary, tables = eda.analyze_split(root, 'public_test', {}, supervised=False, progress=False)
            self.assertEqual(texts, [])
            self.assertNotIn('gold_total_calls', summary)
            self.assertNotIn('transcript_tokens', rows[0])
            self.assertFalse(any(r['path'].startswith(('label', 'audio_text')) for r in tables['field_types']))

    def test_source_defined_system_exception_without_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train/zaic_tools').mkdir(parents=True)
            (root/'train/zaic_tools/executor.py').write_text('self.allowed = set() | {"sys.ask_missing_info"}\nraise RuntimeError("must never import")\n', encoding='utf-8')
            self.assertEqual(eda.runtime_system_tools(root, 'train'), {'sys.ask_missing_info'})
            record = {'id': 'synthetic_sys', 'audio': 'audio/missing.wav', 'audio_text': '',
                      'context': {'installed_apps': {}}, 'history': [],
                      'label': {'tool_calls': [{'name': 'sys.ask_missing_info', 'arguments': {}}],
                                'tool_results': [{'ok': True, 'waiting_for_user': True}], 'final_state': {}}}
            (root/'train/train.jsonl').write_text(json.dumps(record), encoding='utf-8')
            tools = {'sys.ask_missing_info': {'kind': 'meta', 'app_id': 'sys', 'parameters': {}}}
            rows, texts, summary, tables = eda.analyze_split(root, 'train', tools, progress=False)
            self.assertEqual(summary['gold_calls_absent_from_explicit_list'], {'sys.ask_missing_info': 1})
            self.assertEqual(summary['quality'].get('gold_tool_not_permitted_calls', 0), 0)
            self.assertEqual(rows[0]['runtime_available_tools'], 1)
            self.assertEqual(summary['result_status']['waiting_for_user_true'], 1)


if __name__ == '__main__':
    unittest.main()
