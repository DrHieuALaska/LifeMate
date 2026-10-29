import json
import tempfile
import unittest
from pathlib import Path
from _support import fixture, call, FakeEnv
from lifemate.evaluation.benchmark import run
from lifemate.tools.env_runner import EnvRunner


class BenchmarkTests(unittest.TestCase):
    def test_full_fixture_reports_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, manifest = fixture(root)
            predictions = root/'pred.jsonl'
            predictions.write_text(json.dumps({'id': 'dev_2', 'tool_calls': []})+'\n'+json.dumps({'id': 'dev_1', 'tool_calls': [call(x=1)]})+'\n')
            output = root/'artifacts/benchmarks/synthetic'
            result = run(data, manifest, predictions, output, expected_count=2)
            self.assertTrue(result['coverage']['complete'])
            self.assertEqual(result['overall']['exact_trace_accuracy'], 1)
            for name in ('metrics.json', 'summary.md', 'stratified_metrics.csv', 'per_tool_metrics.csv', 'error_taxonomy.csv', 'per_sample.jsonl'):
                self.assertTrue((output/name).is_file())
            for path in output.iterdir():
                text = path.read_text(encoding='utf-8')
                self.assertNotIn('SYNTHETIC_PRIVATE_VALUE', text)
                self.assertNotIn('GOLD_RESULT_SECRET', text)
            with self.assertRaises(ValueError): run(data, manifest, predictions, output, expected_count=2)

    def test_missing_predictions_count_in_full_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, manifest = fixture(root)
            predictions = root/'pred.jsonl'; predictions.write_text('')
            result = run(data, manifest, predictions, root/'artifacts/missing', expected_count=2)
            self.assertEqual(result['coverage']['missing'], 2)
            self.assertEqual(result['overall']['exact_trace_accuracy'], 0)
            self.assertEqual(result['overall']['error_counts']['MISSING_PREDICTION'], 2)
            self.assertEqual(result['overall']['no_action']['fn'], 1)
            self.assertFalse(result['coverage']['complete'])

    def test_fake_replay_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data, manifest = fixture(root)
            predictions = root/'pred.jsonl'
            predictions.write_text(json.dumps({'id': 'dev_2', 'tool_calls': []})+'\n'+json.dumps({'id': 'dev_1', 'tool_calls': [call(x=1)]}))
            output = root/'artifacts/replay'
            result = run(data, manifest, predictions, output, mode='replay', expected_count=2, runner=EnvRunner(lambda sid: FakeEnv()))
            summary = json.loads((output/'execution_summary.json').read_text())
            self.assertEqual(summary['samples_replayed'], 2)
            self.assertEqual(result['overall']['exact_trace_accuracy'], 1)
            self.assertIn('FINAL_STATE_MISMATCH', result['overall']['error_counts'])
