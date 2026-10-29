import unittest
from types import SimpleNamespace
from transcript_stability import LocalAgreement, shared_prefix
from quality_guard import reject_low_confidence_segment, has_suspicious_repetition


class LocalAgreementTests(unittest.TestCase):
    def test_japanese_common_prefix(self):
        self.assertEqual(
            shared_prefix("来月の生産計画について確認", "来月の生産計画について変更"),
            "来月の生産計画について",
        )

    def test_two_pass_stability_never_early_commits(self):
        agreement = LocalAgreement()
        self.assertEqual(agreement.observe("来月の生産計画について確認します"), "")
        stable = agreement.observe("来月の生産計画について変更を確認します")
        self.assertTrue(stable.startswith("来月の生産計画"))
        self.assertLess(len(stable), len("来月の生産計画について"))
        agreement.reset()
        self.assertEqual(agreement.observe("新しい議題に移ります"), "")

    def test_revisions_clear_stability(self):
        agreement = LocalAgreement()
        agreement.observe("来月の生産計画について")
        self.assertEqual(agreement.observe("明日のライン停止について"), "")

    def test_english_works_too(self):
        agreement = LocalAgreement()
        agreement.observe("We need to confirm the schedule today")
        self.assertIn("We need to confirm", agreement.observe("We need to confirm the schedule again"))


class HallucinationGuardTests(unittest.TestCase):
    def test_very_bad_confidence_suppressed(self):
        s = SimpleNamespace(no_speech_prob=.98, avg_logprob=-1.25)
        self.assertTrue(reject_low_confidence_segment(s))

    def test_one_signal_alone_never_discards(self):
        self.assertFalse(reject_low_confidence_segment(
            SimpleNamespace(no_speech_prob=.98, avg_logprob=-.3)
        ))
        self.assertFalse(reject_low_confidence_segment(
            SimpleNamespace(no_speech_prob=.15, avg_logprob=-1.5)
        ))

    def test_short_meeting_replies_are_never_removed(self):
        for text in ["はい", "そうですね", "Yes.", "Okay."]:
            self.assertFalse(has_suspicious_repetition(text))
            self.assertFalse(reject_low_confidence_segment(
                SimpleNamespace(text=text, no_speech_prob=.3, avg_logprob=-.5)
            ))

    def test_japanese_repetition_observed_only(self):
        self.assertTrue(has_suspicious_repetition("ご視聴ありがとうございました" * 3))
        self.assertFalse(has_suspicious_repetition("会議ありがとうございました"))


if __name__ == "__main__":
    unittest.main()
