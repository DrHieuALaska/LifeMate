import unittest
from _support import call, catalog, record
from lifemate.evaluation.stratification import strata
from lifemate.evaluation.metrics import wilson, prf, aggregate, stratified
from lifemate.evaluation.dataset import Predictions, EvaluationDataset
from lifemate.evaluation.benchmark import evaluate
from collections import Counter


class StratificationTests(unittest.TestCase):
    def test_multi_app_sys_distinction(self):
        tags = strata(record(calls=[call('a.read'), call('sys.ask_missing_info')]), catalog(), Counter())
        self.assertIn('apps:multi_including_sys', tags)
        self.assertIn('apps:single_excluding_sys', tags)
        tags = strata(record(calls=[call('a.read'), call('b.write')]), catalog(), Counter())
        self.assertIn('apps:multi_excluding_sys', tags)

    def test_history_boundaries(self):
        for count, label in [(0, 'none'), (1, 'short_1_4'), (4, 'short_1_4'), (5, 'medium_5_12'), (12, 'medium_5_12'), (13, 'long_13+')]:
            self.assertIn('history:'+label, strata(record(history=count), catalog(), Counter()))

    def test_training_frequency_only(self):
        self.assertIn('frequency:rare_1_5', strata(record(calls=[call(x=1)]), catalog(), Counter({'a.write': 2})))
        self.assertIn('frequency:common_20+', strata(record(calls=[call(x=1)]), catalog(), Counter({'a.write': 20})))

    def test_catalog_traits(self):
        tags = strata(record(calls=[call(x=1), call('a.choice', ref='synthetic')]), catalog(), Counter())
        for label in ('xref', 'one_of', 'enum', 'fuzzy'):
            self.assertIn('argument:contains_'+label, tags)

    def test_no_zero_division(self):
        self.assertIsNone(prf(0, 0, 0)['f1'])
        self.assertEqual(wilson(0, 0), [None, None])
        self.assertIsNone(aggregate([])['exact_trace_accuracy'])

    def test_denominators_and_exclusive_action_buckets(self):
        records = [record('x', []), record('y', [call(x=1)])]
        ds = EvaluationDataset(records, catalog(), ['x', 'y'], Counter(), 'synthetic', {'training_frequency_available': True})
        rows = evaluate(ds, Predictions({'x': []}, set(), set(), {}))
        overall = aggregate(rows)
        self.assertEqual(overall['n'], 2); self.assertEqual(overall['exact_trace_accuracy'], .5)
        self.assertEqual(overall['edit_distance_evaluable_samples'], 1)
        groups = stratified(rows)
        self.assertEqual(sum(x['n'] for x in groups if x['stratum'].startswith('actions:')), 2)
        self.assertEqual(next(x['n'] for x in groups if x['stratum'] == 'all'), 2)

    def test_markers_reused(self):
        r = record(); r['audio_text'] = 'Mai đừng bật nó rồi tắt.'
        tags = strata(r, catalog(), Counter())
        self.assertIn('language:relative_time', tags)
        self.assertIn('language:negation', tags)
