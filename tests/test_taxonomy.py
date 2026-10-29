import unittest
from _support import call, catalog
from lifemate.evaluation.taxonomy import classify
from lifemate.evaluation.traces import compare_traces


class TaxonomyTests(unittest.TestCase):
    def labels(self, gold, predicted, allowed=None):
        cat = catalog()
        return classify(gold, predicted, compare_traces(gold, predicted, cat), cat, set(cat.tools) if allowed is None else allowed)

    def test_no_action(self):
        self.assertIn('WRONG_NO_ACTION_DECISION', self.labels([], [call(x=1)])[0])

    def test_clarification_any_position(self):
        gold = [call('a.read'), call('sys.ask_missing_info')]
        self.assertIn('MISSED_MISSING_INFO', self.labels(gold, [call('a.read')])[0])
        self.assertIn('SPURIOUS_MISSING_INFO', self.labels([], [call('sys.ask_missing_info')])[0])

    def test_invalid_tool(self):
        self.assertIn('INVALID_TOOL_SCHEMA', self.labels([], [call('unknown')])[0])

    def test_unavailable(self):
        labels, primary = self.labels([], [call('b.write')], {'a.read'})
        self.assertIn('UNAVAILABLE_TOOL', labels); self.assertEqual(primary, 'UNAVAILABLE_TOOL')

    def test_missing_required(self):
        self.assertIn('MISSING_ARGUMENT', self.labels([call(x=1)], [call()])[0])

    def test_xref_disagreement_not_invalid(self):
        labels, _ = self.labels([call(x=1, ref='synthetic_a')], [call(x=1, ref='synthetic_b')])
        self.assertIn('XREF_ARGUMENT_DISAGREEMENT', labels)
        self.assertNotIn('INVALID_TOOL_SCHEMA', labels)

    def test_time_disagreement(self):
        labels, _ = self.labels([call(x=1, when='2026-01-01T00:00:00')], [call(x=1, when='2026-01-02T00:00:00')])
        self.assertIn('TIME_ARGUMENT_DISAGREEMENT', labels)

    def test_order_precedence(self):
        labels, primary = self.labels([call('a.read'), call(x=1)], [call(x=1), call('a.read')])
        self.assertEqual(primary, 'WRONG_TOOL_ORDER')

    def test_legitimate_repeats_not_flagged(self):
        gold = [call(x=1), call(x=1)]
        self.assertNotIn('POSSIBLE_DUPLICATE_WRITE', self.labels(gold, gold)[0])
        self.assertIn('POSSIBLE_DUPLICATE_WRITE', self.labels([call(x=1)], gold)[0])

    def test_no_asr_root_cause(self):
        labels, _ = self.labels([call(x=1)], [call(x=2)])
        self.assertNotIn('ASR_ERROR', labels)

    def test_one_of_and_enum_type(self):
        cat = catalog()
        self.assertIn('x_one_of', cat.validate(call('a.choice'), set(cat.tools)))
        self.assertNotIn('x_one_of', cat.validate(call('a.choice', ref='synthetic'), set(cat.tools)))
        self.assertIn('enum', cat.validate(call(x=1, mode='bad'), set(cat.tools)))
