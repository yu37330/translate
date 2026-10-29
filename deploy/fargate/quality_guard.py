"""Conservative checks inspired by Vexa's hallucination gate.

Do not blacklist natural acknowledgements ('はい', 'そうですね', etc.):
they are valid short meeting responses. Strong rejection requires BOTH
contradictory Whisper confidence signals. Text-only rules are observational
and must not silently delete genuine speech.
"""
import re


def repeated_loop(text: str) -> bool:
    """Detect a pure 3+ repetition loop; Unicode-safe for Japanese sentences."""
    s = re.sub(r"\s+", "", text.strip())
    if len(s) < 12:
        return False
    for length in range(3, min(30, len(s) // 3) + 1):
        repeated = s[:length]
        if repeated * (len(s) // length) == s[:(len(s) // length) * length]:
            if len(s) % length <= 2 and len(s) // length >= 3:
                return True
    return False


def reject_low_confidence_segment(segment: object) -> bool:
    """Drop only when *both* indicators strongly contradict real speech.

    This is intentionally more conservative than rejecting short phrases
    or fixed strings, which can remove legitimate utterances.
    """
    no_speech = getattr(segment, "no_speech_prob", None)
    logprob = getattr(segment, "avg_logprob", None)
    return (
        isinstance(no_speech, (float, int))
        and isinstance(logprob, (float, int))
        and no_speech >= 0.95
        and logprob <= -1.1
    )


def has_suspicious_repetition(text: str) -> bool:
    """Observation for evaluation only: repetition by itself is NOT deleted."""
    return repeated_loop(text)
