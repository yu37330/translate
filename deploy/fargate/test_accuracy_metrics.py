"""Tests for character error rates and term recall, no model or cloud required."""
import unittest
from accuracy_metrics import normalized, distance, cer_counts, evaluate_pairs


class AccuracyMetricsTest(unittest.TestCase):
    def test_japanese_punctuation_and_width(self):
        self.assertEqual(normalized("ＡＢＣ、 生産計画。"), "abc生産計画")

    def test_levenshtein(self):
        self.assertEqual(distance("abc", "axc"), 1)
        self.assertEqual(distance("abc", "ab"), 1)
        self.assertEqual(distance("abc", "abcd"), 1)
        self.assertEqual(distance("", "abc"), 3)

    def test_japanese_cer(self):
        self.assertEqual(cer_counts("生産計画", "生産予定"), (2, 4))

    def test_micro_and_empty_reference_hallucination(self):
        result = evaluate_pairs([
            {"id":"a", "reference":"生産計画", "hypothesis":"生産計画"},
            {"id":"b", "reference":"ABC", "hypothesis":"AB"},
            {"id":"c", "reference":"", "hypothesis":"幻覚"},
        ])
        self.assertEqual(result["micro_cer"], round(1 / 7, 4))
        self.assertEqual(result["empty_reference_false_positives"], 1)
        self.assertEqual(result["samples_detail"][2]["cer"], None)

    def test_term_recall(self):
        result = evaluate_pairs([
            {"reference":"BOMとPLMを確認", "hypothesis":"BOMと日程を確認"},
            {"reference":"ECNを確認", "hypothesis":"ECNを確認"},
        ], terms=["BOM", "PLM", "ECN"])
        self.assertEqual(result["term_recall"], round(2 / 3, 4))
        self.assertEqual(result["term_occurrences"], 3)


if __name__ == "__main__":
    unittest.main()
