import unittest
from _support import call, catalog
from lifemate.evaluation.traces import compare_traces, align_names
from lifemate.utils.json import equal, loads


class TraceTests(unittest.TestCase):
    def compare(self, gold, predicted):
        return compare_traces(gold, predicted, catalog())

    def test_exact(self):
        self.assertTrue(self.compare([call(x=1)], [call(x=1)])['exact'])

    def test_dict_order(self):
        self.assertTrue(self.compare([call(x=1, ref='synthetic')], [call(ref='synthetic', x=1)])['exact'])

    def test_value(self):
        r = self.compare([call(x=1)], [call(x=2)])
        self.assertFalse(r['exact']); self.assertTrue(r['names_exact'])
        self.assertEqual(r['arguments']['value_mismatches'], 1)

    def test_boolean_number_separation(self):
        self.assertFalse(equal(True, 1))
        self.assertFalse(equal({'x': False}, {'x': 0}))
        self.assertTrue(equal(1, 1.0))
        self.assertFalse(self.compare([call(x=1)], [call(x=True)])['exact'])

    def test_nested_types_arrays_null(self):
        self.assertTrue(equal({'x': [None, {'a': 1}]}, {'x': [None, {'a': 1.0}]}))
        self.assertFalse(equal([1, 2], [2, 1]))
        self.assertFalse(equal('1', 1))

    def test_missing_action(self):
        r = self.compare([call(x=1)], [])
        self.assertEqual(r['fn'], 1); self.assertEqual(r['edit_distance'], 1)

    def test_extra_action(self):
        self.assertEqual(self.compare([], [call(x=1)])['fp'], 1)

    def test_swap(self):
        r = self.compare([call('a.read'), call(x=1)], [call(x=1), call('a.read')])
        self.assertTrue(r['order_mismatch']); self.assertEqual(r['tp'], 2)
        self.assertFalse(r['names_exact'])

    def test_repeat_arguments(self):
        r = self.compare([call(x=1), call(x=2)], [call(x=2), call(x=1)])
        self.assertTrue(r['names_exact']); self.assertFalse(r['exact'])
        self.assertEqual(r['arguments']['argument_objects_total'], 2)

    def test_multiset_repeated_occurrences(self):
        r = self.compare([call(x=1), call(x=2)], [call(x=1)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 0, 1))

    def test_empty(self):
        r = self.compare([], [])
        self.assertTrue(r['exact']); self.assertEqual(r['normalized_edit_distance'], 0)

    def test_alignment_tie_deterministic(self):
        self.assertEqual(align_names(['a', 'b'], ['b', 'a']), (2, [('substitute', 0, 0), ('substitute', 1, 1)]))

    def test_required_optional_and_constraints(self):
        r = self.compare([call(x=1, ref='synthetic_a', mode='on', text='synthetic')], [call(x=1, ref='synthetic_b', mode='off', text='changed')])['arguments']
        self.assertEqual(r['required_correct'], 1)
        self.assertEqual(r['xref_correct'], 0)
        self.assertEqual(r['enum_total'], 1)
        self.assertEqual(r['fuzzy_total'], 1)

    def test_missing_and_extra_arguments(self):
        r = self.compare([call(x=1, ref='synthetic')], [call(x=1, text='synthetic')])['arguments']
        self.assertEqual(r['missing_keys'], 1); self.assertEqual(r['extra_keys'], 1)

    def test_strict_json(self):
        for text in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}']:
            with self.assertRaises(ValueError): loads(text)
