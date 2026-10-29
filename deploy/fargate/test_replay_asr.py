"""Replay tests with deterministic synthetic audio and fake ASR."""
import unittest
import numpy as np
from replay_asr import run_clip


def pcm(duration, amp):
    return np.full(int(duration * 16000), amp, dtype=np.float32)


class ReplayTests(unittest.TestCase):
    def test_replay_finalizes_short_utterance(self):
        chunks = np.concatenate([pcm(.5, 0), pcm(.5, .02), pcm(1.0, 0)])
        result = run_clip(chunks, lambda audio: "生産計画")
        self.assertEqual(result["utterances"], 1)
        self.assertEqual(result["text"], "生産計画")
        self.assertEqual(result["windows"], [(3200, 32000)])

    def test_replay_long_monologue_stability(self):
        chunks = np.concatenate([pcm(8, .02), pcm(1, 0)])
        result = run_clip(chunks, lambda audio: "来月の生産計画を確認します")
        self.assertGreaterEqual(result["interim_updates"], 2)
        self.assertGreaterEqual(result["stability_matches"], 1)
        self.assertEqual(result["utterances"], 1)

    def test_silence_not_transcribed(self):
        result = run_clip(pcm(4, 0), lambda audio: self.fail("ASR should not run on silence"))
        self.assertEqual(result["utterances"], 0)


if __name__ == "__main__":
    unittest.main()
