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

    def test_half_second_pcm_packets_finalize_on_silence(self):
        buffer = UtteranceBuffer(end_silence_sec=0.8, preroll_sec=0.3)
        # The current browser sends 0.5-second PCM packets over WSS.
        self.assertIsNone(buffer.feed(pcm(0.5)))
        self.assertIsNone(buffer.feed(pcm(0.5)))
        self.assertIsNone(buffer.feed(pcm(0.5, 0)))
        event = buffer.feed(pcm(0.5, 0))
        self.assertEqual(event[0], "final")
        self.assertEqual(event[1].size, int(2.0 * 16000))

    def test_short_utterance_and_stop_does_not_lose_words(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(0.5)))
        flushed = buffer.flush()
        self.assertEqual(flushed.size, 8000)

    def test_preroll_preserves_quiet_onset(self):
        buffer = UtteranceBuffer(preroll_sec=0.3)
        quiet_onset = np.full(int(0.5 * 16000), 0.0002, dtype=np.float32)
        self.assertIsNone(buffer.feed(quiet_onset))
        self.assertIsNone(buffer.feed(pcm(0.5)))
        event = buffer.feed(pcm(1.0, 0))
        self.assertEqual(event[0], "final")
        self.assertEqual(event[1].size, int(1.8 * 16000))
        np.testing.assert_allclose(event[1][:160], 0.0002)

    def test_preroll_is_bounded_during_long_silence(self):
        buffer = UtteranceBuffer(preroll_sec=0.3)
        for _ in range(60):
            self.assertIsNone(buffer.feed(pcm(0.5, 0)))
        self.assertLessEqual(buffer._preroll.size, int(0.3 * 16000))

    def test_flush_short_speech_on_stop(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(2)))
        remaining = buffer.flush()
        self.assertIsNotNone(remaining)
        self.assertEqual(remaining.size, 2 * 16000)
        self.assertFalse(buffer.active)
        self.assertIsNone(buffer.flush())

    def test_new_speech_after_final_is_independent(self):
        buffer = UtteranceBuffer()
        buffer.feed(pcm(2))
        buffer.feed(pcm(2, 0))
        self.assertIsNone(buffer.feed(pcm(2, 0)))
        self.assertIsNone(buffer.feed(pcm(2)))
        self.assertEqual(buffer.feed(pcm(2, 0))[0], "final")


if __name__ == "__main__":
    unittest.main()
