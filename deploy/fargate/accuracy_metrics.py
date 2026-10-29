"""Offline Japanese/multilingual ASR quality metrics.

No network requests; use reference/hypothesis pairs from a consented evaluation set.
Character error rate (CER) counts substitutions+insertions+deletions divided by
reference length; it may exceed 100% when hypotheses hallucinate extra text.
"""
import unicodedata


def normalized(text: str) -> str:
    """NFKC, case-fold, remove punctuation and whitespace for lexical CER."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not (
        c.isspace() or unicodedata.category(c).startswith("P")
    ))


def distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance, O(len(reference)) memory."""
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    row = list(range(len(reference) + 1))
    for j, char in enumerate(hypothesis, 1):
        nxt = [j]
        for i, refchar in enumerate(reference, 1):
            nxt.append(min(nxt[-1] + 1, row[i] + 1, row[i - 1] + (refchar != char)))
        row = nxt
    return row[-1]


def cer_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalized(reference), normalized(hypothesis)
    if not ref:
        # Skip empty reference; measure false positives separately in the report.
        return 0, 0
    return distance(ref, hyp), len(ref)


def evaluate_pairs(rows: list[dict], terms: list[str] | None = None) -> dict:
    """Metrics with no transcript contents returned in the summary."""
    total_errors, total_chars = 0, 0
    term_found, term_correct = 0, 0
    empty_reference_hallucinations = 0
    result = []
    norm_terms = [normalized(term) for term in (terms or []) if normalized(term)]
    for row in rows:
        ref = str(row["reference"])
        hyp = str(row["hypothesis"])
        errors, chars = cer_counts(ref, hyp)
        if not normalized(ref) and normalized(hyp):
            empty_reference_hallucinations += 1
        total_errors += errors
        total_chars += chars
        normal_ref, normal_hyp = normalized(ref), normalized(hyp)
        for term in norm_terms:
            if term in normal_ref:
                term_found += 1
                term_correct += int(term in normal_hyp)
        result.append({
            "id": str(row.get("id", len(result)+1)),
            "cer": round(errors / chars, 4) if chars else None,
            "reference_chars": chars,
            "edits": errors,
        })
    return {
        "samples": len(rows),
        "evaluated_chars": total_chars,
        "micro_cer": round(total_errors / total_chars, 4) if total_chars else None,
        "empty_reference_false_positives": empty_reference_hallucinations,
        "term_recall": round(term_correct / term_found, 4) if term_found else None,
        "term_occurrences": term_found,
        "samples_detail": result,
    }
