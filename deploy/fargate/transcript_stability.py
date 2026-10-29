"""Vexa-inspired LocalAgreement-2 for *display-only* partial subtitles.

Never commits a prefix to the permanent transcript. Japanese has no mandatory
word spaces, so common leading Unicode text is used rather than whitespace
tokenization. A stable prefix is an advisory display cue: the final recognition
may still differ. Audio ownership is handled separately by UtteranceBuffer.
"""


def shared_prefix(first: str, second: str) -> str:
    n = min(len(first), len(second))
    i = 0
    while i < n and first[i] == second[i]:
        i += 1
    return first[:i]


class LocalAgreement:
    """Compare consecutive independent hypotheses for one ongoing utterance."""

    def __init__(self, min_chars: int = 8, guard_chars: int = 3):
        self.min_chars = min_chars
        self.guard_chars = guard_chars
        self.last_hypothesis = ""

    def observe(self, hypothesis: str) -> str:
        previous = self.last_hypothesis
        self.last_hypothesis = hypothesis
        if not previous or not hypothesis:
            return ""
        prefix = shared_prefix(previous, hypothesis)
        # Avoid displaying a forming final word/particle as stable.
        if prefix and prefix[-1] not in "。！？!?、，, ":
            prefix = prefix[:-self.guard_chars] if len(prefix) > self.guard_chars else ""
        prefix = prefix.rstrip()
        return prefix if len(prefix) >= self.min_chars else ""

    def reset(self) -> None:
        self.last_hypothesis = ""
