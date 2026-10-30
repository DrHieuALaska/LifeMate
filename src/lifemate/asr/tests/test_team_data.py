"""Verify reuse of team partitions without private data or inference."""
import json
from pathlib import Path
import tempfile
import unittest

from lifemate.asr.team_data import load_dev_audio


class TeamDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'train/audio').mkdir(parents=True)
        for name in ('a', 'b', 'history'):
            (self.root / f'train/audio/{name}.wav').write_bytes(b'synthetic fixture')
        self.manifest = self.root / 'dev_ids.txt'
        self.manifest.write_text('b\na\n', encoding='utf-8')
        self.records = [
            {'id': 'a', 'audio': 'audio/a.wav', 'audio_text': 'synthetic reference A',
             'history': [{'role': 'user', 'audio': 'audio/history.wav', 'audio_text': 'synthetic history'},
                         {'role': 'assistant', 'text': 'synthetic tool response'}],
             'label': {'never_exposed': 'private evaluator field'}, 'context': {}},
            {'id': 'b', 'audio': 'audio/b.wav', 'audio_text': 'synthetic reference B'},
        ]

    def load(self):
        (self.root / 'train/train.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in self.records), encoding='utf-8')
        return load_dev_audio(self.root, self.manifest, expected_count=2)

    def test_manifest_order_history_and_allowlisted_fields(self):
        rows = self.load()
        self.assertEqual([(r.sample_id, r.kind) for r in rows],
                         [('b', 'final'), ('a', 'final'), ('a', 'history')])
        self.assertEqual(set(vars(rows[0])), {'sample_id', 'kind', 'audio_path', 'reference'})
        self.assertNotIn('synthetic reference', repr(rows[0]))
        self.assertEqual(self.manifest.read_text(), 'b\na\n')

    def test_rejects_overlap_and_unknown_manifest_ids(self):
        (self.root / 'holdout_ids.txt').write_text('a\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.load()
        (self.root / 'holdout_ids.txt').write_text('missing\n')
        with self.assertRaisesRegex(ValueError, 'absent'):
            self.load()

    def test_rejects_missing_audio_or_reference(self):
        for key in ('audio', 'audio_text'):
            original = self.records[0].pop(key)
            with self.assertRaises(ValueError):
                self.load()
            self.records[0][key] = original
        (self.root / 'train/audio/a.wav').unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_rejects_public_environment_and_outside_paths(self):
        for path in ('../public_test/audio/x.wav', 'train_environment/state.json', '../../outside.wav'):
            self.records[0]['audio'] = path
            with self.subTest(path=path), self.assertRaises((ValueError, PermissionError)):
                self.load()

    def test_rejects_changed_split_count_and_holdout_entrypoint(self):
        with self.assertRaisesRegex(ValueError, '193'):
            load_dev_audio(self.root, self.manifest)
        with self.assertRaisesRegex(ValueError, 'dev_ids'):
            load_dev_audio(self.root, self.root / 'holdout_ids.txt')

    def test_rejects_duplicates_and_ambiguous_history(self):
        self.records.append(self.records[0].copy())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load()
        self.records.pop()
        self.records[0]['history'][0].pop('role')
        with self.assertRaisesRegex(ValueError, 'role'):
            self.load()


if __name__ == '__main__':
    unittest.main()
