import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from _support import fixture, record, catalog
from lifemate.evaluation.dataset import load_dev, load_predictions, build_agent_input


class DatasetTests(unittest.TestCase):
    def test_manifest_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, manifest = fixture(Path(tmp))
            ds = load_dev(data, manifest, expected_count=2)
            self.assertEqual([r['id'] for r in ds.records], ['dev_2', 'dev_1'])
            self.assertEqual(ds.training_tool_frequency, {'a.read': 1})

    def test_wrong_count_duplicate_unknown_and_overlap(self):
        for contents in ('dev_1\n', 'dev_1\ndev_1\n', 'dev_1\nunknown\n', 'dev_1\ntrain_1\n'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as tmp:
                data, manifest = fixture(Path(tmp)); manifest.write_text(contents)
                with self.assertRaises(ValueError): load_dev(data, manifest, expected_count=2)

    def test_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, manifest = fixture(Path(tmp)); manifest.unlink()
            with self.assertRaisesRegex(ValueError, 'Manifest missing'): load_dev(data, manifest)

    def test_agent_input_no_gold_or_alias(self):
        r = record(); r['gold_results'] = ['NEVER']
        r['context']['label'] = 'NEVER'
        r['context']['installed_apps']['a']['state'] = {'secret': 'NEVER'}
        request = build_agent_input(r, catalog())
        serialized = json.dumps(dataclasses.asdict(request))
        self.assertNotIn('GOLD_RESULT_SECRET', serialized); self.assertNotIn('NEVER', serialized)
        for key in ('label', 'gold_results', 'final_state'):
            self.assertNotIn(key, dataclasses.asdict(request))
        request.context['user_profile']['name'] = 'changed'
        self.assertEqual(r['context']['user_profile']['name'], 'SYNTHETIC_PRIVATE_VALUE')

    def test_prediction_file_anomalies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pred.jsonl'
            path.write_text('\n'.join(['{broken}', '{"id":"dev_1","tool_calls":[]}', '{"id":"dev_1","tool_calls":[]}', '{"id":"dev_2","tool_calls":{}}', '{"id":"extra","tool_calls":[]}']))
            p = load_predictions(path, ['dev_1', 'dev_2', 'missing'])
            self.assertEqual(p.calls, {})
            self.assertEqual(p.malformed_ids, {'dev_1', 'dev_2'})
            self.assertEqual(p.file_errors['malformed_json_lines'], 1)
            self.assertEqual(p.file_errors['extra_id_records'], 1)

    def test_subset_label_and_no_new_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, manifest = fixture(Path(tmp)); before = manifest.read_bytes()
            ds = load_dev(data, manifest, expected_count=2, limit=1)
            self.assertTrue(ds.partition_validation['is_subset'])
            self.assertEqual(manifest.read_bytes(), before)

    def test_refuses_holdout_as_development_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, manifest = fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'holdout'):
                load_dev(data, manifest.parent/'holdout_ids.txt', expected_count=1)
