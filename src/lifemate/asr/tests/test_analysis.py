"""Small independent checks for the offline experiment review, without real data."""
import unittest
import pandas as pd

from lifemate.asr.tools import analyze_asr_outputs as a


class AnalysisTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([
            dict(ref_uid='a',sample_uid='s1',kind='final',duration=1.,reference='một hai ba'),
            dict(ref_uid='b',sample_uid='s1',kind='history',duration=2.,reference='không gửi tiền'),
            dict(ref_uid='c',sample_uid='s2',kind='final',duration=1.,reference='xin chào')])

    def test_normalization_preserves_accents_and_numbers(self):
        self.assertEqual(a.normalize(' CHÀO, bạn! 250000 '),'chào bạn 250000')
        self.assertNotEqual(a.normalize('250000'),a.normalize('hai trăm năm mươi nghìn'))

    def test_corpus_denominator_failure_and_missing(self):
        records=[dict(ref_uid='a',status='ok',text_model='một hai ba',latency_ms=100.),
            dict(ref_uid='b',status='error',text_model='diagnostic partial output')]
        d=a.recompute(self.rows(),records);m=a.aggregate(d)
        self.assertEqual(m['word_edits'],5)
        self.assertEqual(m['n_words'],8)
        self.assertEqual(m['wer'],5/8)
        self.assertEqual(m['wer_success_only'],0.)
        self.assertEqual(m['n_error'],1)
        self.assertEqual(m['n_missing'],1)
        self.assertIsNone(a.aggregate(a.recompute(self.rows(),[]))['wer'])

    def test_cluster_bootstrap_identical_models_and_counts(self):
        records=[dict(ref_uid=r.ref_uid,status='ok',text_model=r.reference) for r in self.rows().itertuples()]
        d=a.recompute(self.rows(),records)
        result=a.paired_statistics({'a':d,'b':d},n_boot=100,seed=7)[0]
        self.assertEqual(result['n_conversations'],2)
        self.assertEqual(result['wer_ci95_low_pp'],0.)
        self.assertEqual(result['wer_ci95_high_pp'],0.)
        self.assertEqual(result['equal_word_edits'],3)
        with self.assertRaises(ValueError):a.paired_statistics({'a':d,'b':d.iloc[:2]},n_boot=100)

    def test_path_filter_before_reading(self):
        self.assertTrue(a.blocked('data/example_environment/state.json'))
        self.assertTrue(a.blocked('example_environment/references.csv'))
        self.assertTrue(a.blocked('../outside.csv'))
        self.assertFalse(a.blocked('run/predictions.jsonl'))

    def test_script_and_repetition_flags_are_literal(self):
        flags=a.text_flags('xin chào','xin chào bạn')
        self.assertFalse(flags['non_latin_script'])
        self.assertTrue(a.text_flags('xin chào','ภาษาไทย')['non_latin_script'])
        self.assertTrue(a.text_flags('xin chào bạn nhé','xin chào bạn '*20)['repeated_trigram_hint'])
        self.assertTrue(a.text_flags('xin chào','')['raw_empty'])


if __name__=='__main__':unittest.main()
