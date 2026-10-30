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

    def test_feed_many_preserves_new_speech_after_midpacket_silence(self):
        buffer = UtteranceBuffer(end_silence_sec=0.8, preroll_sec=0.3)
        # Speech ends; the next 0.5s packet contains enough silence to close
        # this utterance AND the beginning of a new utterance.
        events = []
        events += buffer.feed_many(pcm(1.0, 0.01))
        self.assertEqual(events, [])
        events += buffer.feed_many(pcm(0.5, 0.0))
        events += buffer.feed_many(np.concatenate([pcm(0.35, 0.0), pcm(0.15, 0.02)]))
        self.assertEqual([e[0] for e in events], ["final"])
        first = events[0]
        self.assertEqual(first[3], int(1.8 * 16000))  # 0.8s of silence after 1s speech
        self.assertTrue(buffer.active, "second utterance must not be lost")
        second_events = buffer.feed_many(pcm(1.0, 0.0))
        self.assertEqual([e[0] for e in second_events], ["final"])
        second = second_events[0]
        self.assertGreater(second[2], first[2])
        self.assertGreater(second[3], first[3])
        self.assertTrue(np.any(second[1] > 0.015), "new utterance speech retained")

    def test_feed_many_timestamps_monotonic_for_multiple_events(self):
        buffer = UtteranceBuffer(end_silence_sec=0.8)
        speech = np.concatenate([
            pcm(0.5, 0.01), pcm(1.0, 0.0),
            pcm(0.5, 0.03), pcm(1.0, 0.0),
        ])
        events = buffer.feed_many(speech)
        self.assertEqual([e[0] for e in events], ["final", "final"])
        self.assertEqual(events[0][2], 0)
        self.assertLess(events[0][3], events[1][3])
        self.assertTrue(np.any(events[0][1] > 0.009))
        self.assertTrue(np.any(events[1][1] > 0.029))
        self.assertEqual(buffer.total_received_samples, speech.size)

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

    def test_audio_timestamps_advance_from_pcm_samples(self):
        buffer = UtteranceBuffer(preroll_sec=0.3)
        self.assertIsNone(buffer.feed(pcm(0.5, 0)))
        self.assertIsNone(buffer.feed(pcm(0.5)))
        self.assertIsNone(buffer.feed(pcm(0.5, 0)))
        event = buffer.feed(pcm(0.5, 0))
        self.assertEqual(event[0], "final")
        self.assertEqual(buffer.total_received_samples, 32000)
        self.assertEqual(buffer.last_event_start_sample, 3200)  # 0.2s of quiet preroll
        self.assertEqual(buffer.last_event_end_sample, 32000)
        self.assertIsNone(buffer.feed(pcm(0.5, 0)))
        self.assertEqual(buffer.last_event_end_sample, 32000)

    def test_silence_duration_uses_pcm_clock_and_survives_final_reset(self):
        buffer = UtteranceBuffer()
        self.assertIsNone(buffer.feed(pcm(0.5)))
        buffer.feed(pcm(1.0, 0))  # closes first utterance
        for _ in range(6):
            buffer.feed(pcm(0.5, 0))
        self.assertGreaterEqual(buffer.silence_since_voice_sec, 3.0)
        buffer.feed(pcm(0.5))
        self.assertEqual(buffer.silence_since_voice_sec, 0.0)

    def test_flush_tracks_last_time_range(self):
        buffer = UtteranceBuffer()
        buffer.feed(pcm(0.5))
        buffer.flush()
        self.assertEqual(buffer.last_event_start_sample, 0)
        self.assertEqual(buffer.last_event_end_sample, 8000)

    def test_new_speech_after_final_is_independent(self):
        buffer = UtteranceBuffer()
        buffer.feed(pcm(2))
        buffer.feed(pcm(2, 0))
        self.assertIsNone(buffer.feed(pcm(2, 0)))
        self.assertIsNone(buffer.feed(pcm(2)))
        self.assertEqual(buffer.feed(pcm(2, 0))[0], "final")


if __name__ == "__main__":
    unittest.main()
