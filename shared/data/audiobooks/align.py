"""
Align ASR transcriptions to book text and transfer punctuation.

Strategy:
1. Use normalized (no-punctuation) text for LOCATING where each ASR chunk falls in the book
2. Map the match region back to the ORIGINAL book text (with punctuation)
3. Do word-level alignment (ASR words ↔ book words) to transfer punctuation
4. Output = ASR words + book punctuation → best of both worlds

ASR gives accurate content; book text gives punctuation and formatting.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AlignedChunk:
    chunk_id: str
    asr_text: str           # raw ASR output (no punctuation)
    book_text: str           # matched book region WITH punctuation
    punctuated_text: str     # final output: ASR words + book punctuation
    book_offset_start: int   # offset in ORIGINAL book text
    book_offset_end: int
    cer: float               # CER between normalized ASR and normalized book match
    alignment_score: float   # 1 - CER


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Strip markdown headers, punctuation, collapse whitespace."""
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _strip_punct(word: str) -> str:
    """Strip punctuation from a single word for comparison."""
    return re.sub(r'[^\w]', '', word, flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Offset mapping: normalized ↔ original
# ---------------------------------------------------------------------------

def _build_norm_to_orig_map(original: str) -> tuple[str, list[int]]:
    """
    Normalize text and build a character-level map from normalized positions
    back to original positions.

    Returns:
        (normalized_text, norm_to_orig) where norm_to_orig[i] = index in original
    """
    # First strip markdown headers
    cleaned = re.sub(r'^#+\s+', '', original, flags=re.MULTILINE)

    norm_chars = []
    norm_to_orig = []
    prev_was_space = False

    for orig_idx, ch in enumerate(cleaned):
        if re.match(r'[^\w\s]', ch, re.UNICODE):
            # Punctuation — skip in normalized form
            continue
        if re.match(r'\s', ch):
            if not prev_was_space and norm_chars:
                norm_chars.append(' ')
                norm_to_orig.append(orig_idx)
                prev_was_space = True
            continue
        norm_chars.append(ch)
        norm_to_orig.append(orig_idx)
        prev_was_space = False

    normalized = ''.join(norm_chars).strip()
    # Trim the map to match the stripped result
    if norm_chars and norm_chars[0] == ' ':
        norm_to_orig = norm_to_orig[1:]
    if norm_chars and norm_chars[-1] == ' ':
        norm_to_orig = norm_to_orig[:-1]

    # Ensure lengths match
    if len(normalized) != len(norm_to_orig):
        # Rebuild more carefully
        norm_to_orig = norm_to_orig[:len(normalized)]

    return normalized, norm_to_orig


# ---------------------------------------------------------------------------
# Levenshtein / CER
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def _compute_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# N-gram search (operates on normalized text)
# ---------------------------------------------------------------------------

def _extract_ngrams(text: str, n: int = 8) -> list[str]:
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(0, len(text) - n + 1, n // 2)]


def _find_best_match_ngram(query: str, text: str, search_start: int = 0,
                           search_window: int = None) -> tuple[int, int, float]:
    """Find best matching substring in `text` for `query` using n-gram anchoring."""
    query_len = len(query)
    if query_len == 0:
        return search_start, search_start, 1.0

    search_end = min(search_start + search_window, len(text)) if search_window else len(text)
    if search_start >= search_end:
        return search_start, min(search_start + query_len, len(text)), 1.0

    search_region = text[search_start:search_end]

    # Find n-gram anchor positions
    ngram_size = min(8, query_len // 2) if query_len > 4 else query_len
    ngrams = _extract_ngrams(query, ngram_size)

    hit_positions = []
    for ng in ngrams:
        pos = 0
        while True:
            idx = search_region.find(ng, pos)
            if idx == -1:
                break
            hit_positions.append(idx)
            pos = idx + 1

    if not hit_positions:
        ngram_size = max(3, ngram_size // 2)
        ngrams = _extract_ngrams(query, ngram_size)
        for ng in ngrams:
            pos = 0
            while True:
                idx = search_region.find(ng, pos)
                if idx == -1:
                    break
                hit_positions.append(idx)
                pos = idx + 1

    if not hit_positions:
        candidate_end = min(query_len, len(search_region))
        substr = search_region[:candidate_end]
        cer = _compute_cer(substr, query) if substr else 1.0
        return search_start, search_start + candidate_end, cer

    hit_positions.sort()

    # Cluster hits
    best_cluster_start = hit_positions[0]
    best_count = 0
    for anchor in set(hit_positions):
        count = sum(1 for h in hit_positions if anchor <= h <= anchor + query_len)
        if count > best_count:
            best_count = count
            best_cluster_start = anchor

    # Refine boundaries
    best_cer = 1.0
    best_start = best_cluster_start
    best_end = min(best_cluster_start + query_len, len(search_region))

    for start_delta in range(-query_len // 3, query_len // 6 + 1, max(1, query_len // 10)):
        trial_start = max(0, best_cluster_start + start_delta)
        for length_factor in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
            trial_end = min(trial_start + int(query_len * length_factor), len(search_region))
            if trial_end <= trial_start:
                continue
            substr = search_region[trial_start:trial_end]
            cer = _compute_cer(substr, query)
            if cer < best_cer:
                best_cer = cer
                best_start = trial_start
                best_end = trial_end

    return search_start + best_start, search_start + best_end, best_cer


# ---------------------------------------------------------------------------
# Word-level punctuation transfer
# ---------------------------------------------------------------------------

def _word_level_align(asr_words: list[str], book_words: list[str]) -> list[tuple[int, int]]:
    """
    Align ASR words to book words using word-level Levenshtein on stripped forms.
    Returns list of (asr_idx, book_idx) pairs. Uses DP backtrace.
    """
    n, m = len(asr_words), len(book_words)
    asr_stripped = [_strip_punct(w).lower() for w in asr_words]
    book_stripped = [_strip_punct(w).lower() for w in book_words]

    # DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if asr_stripped[i-1] == book_stripped[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete (ASR word not in book)
                dp[i][j-1] + 1,      # insert (book word not in ASR)
                dp[i-1][j-1] + cost,  # match/substitute
            )

    # Backtrace — include both exact matches and close substitutions
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if asr_stripped[i-1] == book_stripped[j-1] else 1
        if dp[i][j] == dp[i-1][j-1] + cost:
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


def _word_cer(a: str, b: str) -> float:
    """Character-level edit distance ratio between two words."""
    a, b = a.lower(), b.lower()
    if not a and not b:
        return 0.0
    return _levenshtein_distance(a, b) / max(len(a), len(b))


def _transfer_punctuation(asr_text: str, book_text_with_punct: str) -> str:
    """
    Merge ASR and book text at word level.

    For each aligned word pair:
    - If book word is close to ASR word (CER < 0.3): USE BOOK WORD entirely
      (correct spelling + punctuation)
    - Otherwise: keep ASR word as-is (book match is wrong region)

    This fixes ASR spelling errors (ქ/კ, თ/ტ, ფ/პ, etc.) using book text
    while keeping ASR word order and boundaries.
    """
    asr_words = asr_text.split()
    book_words = book_text_with_punct.split()

    if not asr_words or not book_words:
        return asr_text

    pairs = _word_level_align(asr_words, book_words)

    result_words = list(asr_words)  # start with ASR words

    for asr_idx, book_idx in pairs:
        book_word = book_words[book_idx]
        asr_word = asr_words[asr_idx]

        asr_stripped = _strip_punct(asr_word)
        book_stripped = _strip_punct(book_word)

        # Check if the words are close enough to trust the book spelling
        cer = _word_cer(asr_stripped, book_stripped)

        if cer < 0.3:
            # Close match — use the book word (correct spelling + punctuation)
            result_words[asr_idx] = book_word
        # else: keep ASR word as-is (the book match is probably wrong)

    return ' '.join(result_words)


# ---------------------------------------------------------------------------
# Map normalized offsets back to original text
# ---------------------------------------------------------------------------

def _snap_to_word_boundary(text: str, pos: int, direction: str = 'left') -> int:
    """Snap a position to the nearest word boundary."""
    pos = max(0, min(pos, len(text)))
    if direction == 'left':
        while pos > 0 and pos <= len(text) and not text[pos - 1].isspace():
            pos -= 1
        return pos
    else:
        while pos < len(text) and not text[pos].isspace():
            pos += 1
        return pos


# ---------------------------------------------------------------------------
# Main alignment
# ---------------------------------------------------------------------------

def align_chunks_to_book(
    chunk_transcriptions: dict[str, str],
    book_text: str,
    chunk_order: list[str] = None,
) -> list[AlignedChunk]:
    """
    Align ASR transcriptions to book text and transfer punctuation.

    Returns AlignedChunk with:
    - asr_text: raw ASR output
    - book_text: matched book region with original punctuation
    - punctuated_text: ASR words with punctuation transferred from book
    """
    if chunk_order is None:
        chunk_order = sorted(chunk_transcriptions.keys())

    # Build normalized book + offset map
    norm_book, norm_to_orig = _build_norm_to_orig_map(book_text)

    # Also keep a "clean" version of book (markdown stripped, but punctuation kept)
    clean_book = re.sub(r'^#+\s+', '', book_text, flags=re.MULTILINE)
    clean_book = re.sub(r'\s+', ' ', clean_book).strip()

    logger.info(f"Book text: {len(clean_book)} chars original, {len(norm_book)} chars normalized")
    logger.info(f"Aligning {len(chunk_order)} chunks...")

    results = []
    search_cursor = 0

    for i, chunk_id in enumerate(chunk_order):
        asr_text = chunk_transcriptions.get(chunk_id, "")
        norm_asr = _normalize(asr_text)

        if not norm_asr:
            results.append(AlignedChunk(
                chunk_id=chunk_id, asr_text=asr_text, book_text="",
                punctuated_text="",
                book_offset_start=search_cursor, book_offset_end=search_cursor,
                cer=1.0, alignment_score=0.0,
            ))
            continue

        # Search in normalized space
        backtrack = min(300, search_cursor)
        search_from = search_cursor - backtrack
        window = max(len(norm_asr) * 5, 3000)

        norm_start, norm_end, cer = _find_best_match_ngram(
            norm_asr, norm_book,
            search_start=search_from,
            search_window=window + backtrack,
        )

        # Wider search if poor match
        if cer > 0.4:
            wider_start = max(0, search_cursor - 2000)
            wider_window = min(len(norm_asr) * 15, len(norm_book) - wider_start)
            s2, e2, c2 = _find_best_match_ngram(
                norm_asr, norm_book,
                search_start=wider_start,
                search_window=wider_window,
            )
            if c2 < cer:
                norm_start, norm_end, cer = s2, e2, c2

        # Map normalized offsets back to original text positions
        if norm_start < len(norm_to_orig) and norm_end <= len(norm_to_orig):
            orig_start = norm_to_orig[norm_start]
            orig_end = norm_to_orig[min(norm_end - 1, len(norm_to_orig) - 1)] + 1
        else:
            # Fallback: approximate
            ratio = len(clean_book) / max(len(norm_book), 1)
            orig_start = int(norm_start * ratio)
            orig_end = int(norm_end * ratio)

        # Snap to word boundaries in original text
        orig_start = _snap_to_word_boundary(clean_book, orig_start, 'left')
        orig_end = _snap_to_word_boundary(clean_book, orig_end, 'right')

        # Clamp
        orig_start = max(0, orig_start)
        orig_end = min(len(clean_book), orig_end)

        book_text_matched = clean_book[orig_start:orig_end].strip()

        # Transfer punctuation from book text to ASR text
        punctuated = _transfer_punctuation(asr_text, book_text_matched)

        # Advance cursor in normalized space
        if cer < 0.5:
            search_cursor = norm_end

        results.append(AlignedChunk(
            chunk_id=chunk_id,
            asr_text=asr_text,
            book_text=book_text_matched,
            punctuated_text=punctuated,
            book_offset_start=orig_start,
            book_offset_end=orig_end,
            cer=cer,
            alignment_score=max(0.0, 1.0 - cer),
        ))

        if (i + 1) % 50 == 0 or i == len(chunk_order) - 1:
            avg_cer = sum(r.cer for r in results) / len(results)
            logger.info(
                f"  Aligned {i + 1}/{len(chunk_order)} chunks, "
                f"avg CER: {avg_cer:.3f}, cursor at {search_cursor}/{len(norm_book)}"
            )

    cers = [r.cer for r in results if r.asr_text]
    if cers:
        logger.info(
            f"Alignment complete: {len(results)} chunks, "
            f"avg CER: {sum(cers)/len(cers):.3f}, "
            f"median CER: {sorted(cers)[len(cers)//2]:.3f}, "
            f"good (<0.2): {sum(1 for c in cers if c < 0.2)}/{len(cers)}"
        )

    return results
