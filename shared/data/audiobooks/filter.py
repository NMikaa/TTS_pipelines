"""
Quality filtering for aligned speech segments.

Tier 1: Heuristic filters (cheap, fast)
- Alignment CER threshold
- Georgian script ratio
- Characters per second bounds
- Compression ratio (repetition detection)
- Repeated character/word detection
- Empty/near-empty transcriptions
"""

import logging
import re
import zlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds
MAX_ALIGNMENT_CER = 0.20       # Alignment quality
MIN_GEORGIAN_RATIO = 0.80      # Script consistency
MIN_CHARS_PER_SEC = 3.0        # Too few chars = missed speech
MAX_CHARS_PER_SEC = 25.0       # Too many chars = hallucination
MAX_COMPRESSION_RATIO = 3.5    # Repetitive text detection (Georgian compresses well)
MAX_REPEATED_CHARS = 4         # Same char repeated consecutively
MAX_REPEATED_WORDS = 3         # Same word repeated consecutively
MIN_TEXT_CHARS = 5             # Minimum text length


def _georgian_char_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Georgian (U+10A0-U+10FF, U+2D00-U+2D2F)."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    georgian = sum(1 for c in alpha_chars if '\u10a0' <= c <= '\u10ff' or '\u2d00' <= c <= '\u2d2f')
    return georgian / len(alpha_chars)


def _compression_ratio(text: str) -> float:
    """Ratio of original to compressed text length. High = repetitive."""
    if not text:
        return 0.0
    text_bytes = text.encode("utf-8")
    compressed = zlib.compress(text_bytes)
    return len(text_bytes) / len(compressed)


def _has_repeated_chars(text: str, max_repeat: int = MAX_REPEATED_CHARS) -> bool:
    """Check for suspiciously repeated characters."""
    pattern = r'(.)\1{' + str(max_repeat) + r',}'
    return bool(re.search(pattern, text))


def _has_repeated_words(text: str, max_repeat: int = MAX_REPEATED_WORDS) -> bool:
    """Check for suspiciously repeated words."""
    words = text.split()
    if len(words) < max_repeat + 1:
        return False
    for i in range(len(words) - max_repeat):
        if len(set(words[i:i + max_repeat + 1])) == 1:
            return True
    return False


@dataclass
class FilterResult:
    chunk_id: str
    passed: bool
    reason: str  # Empty if passed, reason for rejection otherwise
    asr_text: str
    book_text: str
    punctuated_text: str
    duration_sec: float
    cer: float
    georgian_ratio: float
    chars_per_sec: float
    compression_ratio: float


def filter_chunks(
    aligned_chunks: list,
    chunk_durations: dict[str, float],
) -> tuple[list[FilterResult], dict]:
    """
    Apply quality filters to aligned chunks.

    Args:
        aligned_chunks: List of AlignedChunk objects from align.py
        chunk_durations: Dict mapping chunk_id -> duration in seconds

    Returns:
        (list of FilterResult, summary stats dict)
    """
    results = []
    reasons_count = {}

    for chunk in aligned_chunks:
        duration = chunk_durations.get(chunk.chunk_id, 0.0)
        asr_text = chunk.asr_text
        book_text = chunk.book_text
        punctuated_text = getattr(chunk, 'punctuated_text', '') or asr_text
        cer = chunk.cer
        reason = ""

        # Use ASR text for quality checks (it's the accurate content)
        check_text = asr_text

        # Filter 1: Empty text
        if len(check_text.strip()) < MIN_TEXT_CHARS:
            reason = "empty_text"

        # Filter 2: Alignment CER
        elif cer > MAX_ALIGNMENT_CER:
            reason = f"high_cer:{cer:.3f}"

        # Filter 3: Georgian script ratio
        elif (gr := _georgian_char_ratio(check_text)) < MIN_GEORGIAN_RATIO:
            reason = f"low_georgian:{gr:.2f}"

        # Filter 4: Chars per second
        elif duration > 0:
            cps = len(check_text) / duration
            if cps < MIN_CHARS_PER_SEC:
                reason = f"low_cps:{cps:.1f}"
            elif cps > MAX_CHARS_PER_SEC:
                reason = f"high_cps:{cps:.1f}"
        else:
            cps = 0.0

        # Filter 5: Compression ratio
        if not reason and (cr := _compression_ratio(check_text)) > MAX_COMPRESSION_RATIO:
            reason = f"high_compression:{cr:.2f}"

        # Filter 6: Repeated chars
        if not reason and _has_repeated_chars(check_text):
            reason = "repeated_chars"

        # Filter 7: Repeated words
        if not reason and _has_repeated_words(check_text):
            reason = "repeated_words"

        passed = reason == ""
        if not passed:
            tag = reason.split(":")[0]
            reasons_count[tag] = reasons_count.get(tag, 0) + 1

        results.append(FilterResult(
            chunk_id=chunk.chunk_id,
            passed=passed,
            reason=reason,
            asr_text=asr_text,
            book_text=book_text if passed else "",
            punctuated_text=punctuated_text if passed else "",
            duration_sec=duration,
            cer=cer,
            georgian_ratio=_georgian_char_ratio(check_text) if check_text else 0.0,
            chars_per_sec=len(check_text) / duration if duration > 0 else 0.0,
            compression_ratio=_compression_ratio(check_text),
        ))

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    logger.info(f"Filter results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    for reason, count in sorted(reasons_count.items(), key=lambda x: -x[1]):
        logger.info(f"  Rejected by {reason}: {count}")

    summary = {
        "total": total,
        "passed": passed,
        "rejected": total - passed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "rejection_reasons": reasons_count,
    }

    return results, summary
