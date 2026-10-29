"""Offline tests: synthetic PCM only; no meeting audio, AWS, or Whisper model."""
import unittest
import numpy as np

from streaming_asr import UtteranceBuffer


def pcm(seconds: float, amplitude: float = 0.01) -> np.ndarray:
    return np.full(int(seconds * 16000), amplitude, dtype=np.float32)


class UtteranceBufferTests(unittest.TestCase):
    def test_ignores_initial_silence(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2, 0)))
        self.assertFalse(buffer.active)

    def test_short_utterance_finalized_on_silence(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2)))
        outcome = buffer.feed(pcm(2, 0))
        self.assertEqual(outcome[0], "final")
        self.assertEqual(outcome[1].size, 4 * 16000)
        self.assertFalse(buffer.active)

    def test_interim_replaces_snapshot_not_incremental_concat(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2)))
        first = buffer.feed(pcm(2))
        self.assertEqual(first[0], "interim")
        self.assertEqual(first[1].size, 4 * 16000)
        self.assertIsNone(buffer.feed(pcm(2)))
        second = buffer.feed(pcm(2))
        self.assertEqual(second[0], "interim")
        self.assertEqual(second[1].size, 8 * 16000)
        final = buffer.feed(pcm(2, 0))
        self.assertEqual(final[0], "final")

    def test_low_volume_not_dropped(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2, .0008)))
        self.assertEqual(buffer.feed(pcm(2, 0))[0], "final")

    def test_finalize_when_last_second_silent_inside_chunk(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2)))
        mixed = np.concatenate([pcm(0.5), pcm(1.5, 0)])
        self.assertEqual(buffer.feed(mixed)[0], "final")

    def test_max_length_bounded(self):
        buffer = UtteranceBuffer()
        for _ in range(5):
            buffer.feed(pcm(2))
        self.assertEqual(buffer.feed(pcm(2))[0], "final")
        self.assertFalse(buffer.active)

    def test_new_speech_after_final_is_independent(self):
        buffer = UtteranceBuffer()
        buffer.feed(pcm(2))
        buffer.feed(pcm(2, 0))
        self.assertIsNone(buffer.feed(pcm(2, 0)))
        self.assertIsNone(buffer.feed(pcm(2)))
        self.assertEqual(buffer.feed(pcm(2, 0))[0], "final")


if __name__ == "__main__":
    unittest.main()
