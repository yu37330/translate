"""Continuous utterance buffering for low-latency streaming ASR.

Keep 2-second transport chunks intact; run Whisper on 4-12 seconds of context.
Do not discard silent audio on the browser. Each utterance is finalized once.
This module contains no cloud or model dependency and is unit-testable.
"""
import numpy as np


class UtteranceBuffer:
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.0006,
        end_silence_sec: float = 1.0,
        min_interim_sec: float = 4.0,
        interim_interval_sec: float = 4.0,
        max_utterance_sec: float = 12.0,
    ):
        self.sample_rate = sample_rate
        self.threshold = speech_threshold
        self.end_silence_sec = end_silence_sec
        self.min_interim_sec = min_interim_sec
        self.interim_interval_sec = interim_interval_sec
        self.max_utterance_sec = max_utterance_sec
        self.reset()

    def reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._silence_samples = 0
        self._last_decode_samples = 0

    @property
    def active(self) -> bool:
        return bool(self._chunks)

    def _voice_frames(self, audio: np.ndarray) -> tuple[bool, int]:
        """Conservative 20-ms energy gate; Silero VAD runs inside Whisper."""
        frame_size = max(1, int(self.sample_rate * 0.02))
        n_frames = (audio.size + frame_size - 1) // frame_size
        padded = np.pad(audio, (0, n_frames * frame_size - audio.size))
        rms = np.sqrt(np.mean(np.square(padded.reshape(n_frames, frame_size)), axis=1))
        voiced = np.flatnonzero(rms >= self.threshold)
        if voiced.size < 3:  # tolerate a few isolated clicks, not silent/small speech segments
            return False, audio.size
        trailing = min(audio.size, (n_frames - int(voiced[-1]) - 1) * frame_size)
        return True, trailing

    def flush(self) -> np.ndarray | None:
        """Return the last in-progress utterance when a user stops recording."""
        if not self.active:
            return None
        audio = np.concatenate(self._chunks)
        self.reset()
        return audio

    def feed(self, audio: np.ndarray) -> tuple[str, np.ndarray] | None:
        """Return ('interim'|'final', full utterance audio), or None.

        End-silence is measured on received PCM samples, not wall-clock.
        The bounded maximum prevents increasing CPU work and memory indefinitely.
        """
        if audio.size == 0:
            return None
        has_voice, trailing = self._voice_frames(audio)
        if not self.active and not has_voice:
            return None
        self._chunks.append(audio.copy())
        self._samples += audio.size
        self._silence_samples = trailing if has_voice else self._silence_samples + audio.size

        final = (
            self._silence_samples >= self.end_silence_sec * self.sample_rate
            or self._samples >= self.max_utterance_sec * self.sample_rate
        )
        interim = (
            self._samples >= self.min_interim_sec * self.sample_rate
            and self._samples - self._last_decode_samples >= self.interim_interval_sec * self.sample_rate
        )
        if not (final or interim):
            return None
        utterance = np.concatenate(self._chunks)
        if final:
            self.reset()
            return "final", utterance
        self._last_decode_samples = self._samples
        return "interim", utterance
