"""
Speech corpus pipeline v1: chunk → transcribe → align → filter → manifest.

Usage:
    python -m shared.data.audiobooks.pipeline \
        --audio-dir ./data/wavs \
        --parquet reference_texts.parquet \
        --output-dir ./data/output
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("corpus_pipeline")


def _setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(output_dir / "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
        force=True,
    )
    return log_file


def _find_wav_for_title(audio_dir: Path, title: str) -> str:
    """Find WAV file matching a book title (filename pattern: title_author_narrator.wav)."""
    for f in audio_dir.glob("*.wav"):
        if f.name.startswith(title + "_"):
            return str(f)
    raise FileNotFoundError(f"No WAV file found for title '{title}' in {audio_dir}")


def _sanitize_book_id(title: str) -> str:
    """Create a safe filesystem ID from a book title."""
    # Keep Georgian chars and alphanumeric, replace rest with underscore
    safe = re.sub(r'[^\w]', '_', title, flags=re.UNICODE)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:80]


def _extract_narrator(wav_path: str) -> str:
    """Extract narrator name from filename pattern: title_author_narrator.wav"""
    stem = Path(wav_path).stem
    parts = stem.split('_')
    if len(parts) >= 3:
        return parts[-1]  # last part is narrator
    return "unknown"


def process_book(
    wav_path: str,
    book_text: str,
    book_id: str,
    output_dir: Path,
    num_gpus: int = 1,
    skip_transcribe: bool = False,
) -> dict:
    """
    Process a single audio file through the full pipeline.

    Returns dict with stats and output paths.
    """
    from .chunk import chunk_audiobook
    from .align import align_chunks_to_book
    from .filter import filter_chunks

    book_dir = output_dir / book_id
    chunks_dir = book_dir / "chunks"
    transcription_path = book_dir / "transcriptions.json"
    alignment_path = book_dir / "alignments.json"
    manifest_path = book_dir / "manifest.json"
    stats_path = book_dir / "stats.json"

    book_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # --- Step 1: Chunk ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 1: Chunking {Path(wav_path).name}")
    logger.info(f"{'='*60}")
    t0 = time.time()

    chunks = chunk_audiobook(
        audio_path=wav_path,
        output_dir=str(chunks_dir),
        book_id=book_id,
    )
    chunk_time = time.time() - t0
    logger.info(f"Chunking done in {chunk_time:.1f}s: {len(chunks)} chunks")

    if not chunks:
        logger.warning("No chunks produced! Skipping remaining steps.")
        return {"book_id": book_id, "status": "no_chunks"}

    # Build chunk durations and order
    chunk_durations = {c.chunk_id: c.duration_sec for c in chunks}
    chunk_order = [c.chunk_id for c in chunks]

    # --- Step 2: Transcribe ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 2: Transcribing {len(chunks)} chunks")
    logger.info(f"{'='*60}")
    t0 = time.time()

    if skip_transcribe and transcription_path.exists():
        logger.info("Loading cached transcriptions...")
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)
    else:
        from .transcribe import transcribe_chunks
        transcriptions = transcribe_chunks(
            chunk_dir=str(chunks_dir),
            output_path=str(transcription_path),
            num_gpus=num_gpus,
        )

    transcribe_time = time.time() - t0
    logger.info(f"Transcription done in {transcribe_time:.1f}s: {len(transcriptions)} texts")

    # --- Step 3: Align ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 3: Aligning to book text")
    logger.info(f"{'='*60}")
    t0 = time.time()

    aligned = align_chunks_to_book(
        chunk_transcriptions=transcriptions,
        book_text=book_text,
        chunk_order=chunk_order,
    )
    align_time = time.time() - t0
    logger.info(f"Alignment done in {align_time:.1f}s")

    # Detect text mismatch: if avg CER > 0.4, the book text doesn't match the audio
    # (e.g. different translation). Fall back to ASR-only (no punctuation).
    TEXT_MISMATCH_THRESHOLD = 0.4
    cers = [a.cer for a in aligned if a.asr_text]
    avg_cer = sum(cers) / len(cers) if cers else 1.0
    text_matched = avg_cer < TEXT_MISMATCH_THRESHOLD

    if not text_matched:
        logger.warning(
            f"TEXT MISMATCH DETECTED: avg CER = {avg_cer:.3f} "
            f"(threshold {TEXT_MISMATCH_THRESHOLD}). "
            f"Book text does not match audio (different translation?). "
            f"Falling back to ASR-only output (no punctuation)."
        )
        # Override: use raw ASR text, clear book text
        for a in aligned:
            a.punctuated_text = a.asr_text
            a.book_text = ""
            a.cer = 0.0  # Don't penalize in filter — ASR is the ground truth now
            a.alignment_score = 1.0
    else:
        logger.info(f"Text match OK: avg CER = {avg_cer:.3f}")

    # Save alignments for inspection
    alignment_data = [
        {
            "chunk_id": a.chunk_id,
            "asr_text": a.asr_text,
            "book_text": a.book_text,
            "punctuated_text": a.punctuated_text,
            "book_offset_start": a.book_offset_start,
            "book_offset_end": a.book_offset_end,
            "cer": a.cer,
            "alignment_score": a.alignment_score,
            "text_matched": text_matched,
        }
        for a in aligned
    ]
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(alignment_data, f, ensure_ascii=False, indent=2)

    # --- Step 4: Filter ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 4: Quality filtering")
    logger.info(f"{'='*60}")
    t0 = time.time()

    filter_results, filter_summary = filter_chunks(aligned, chunk_durations)
    filter_time = time.time() - t0

    # --- Step 5: Generate manifest ---
    manifest_entries = []
    for fr in filter_results:
        if fr.passed:
            chunk_path = chunks_dir / f"{fr.chunk_id}.wav"
            manifest_entries.append({
                "id": fr.chunk_id,
                "audio_filepath": str(chunk_path),
                "text": fr.punctuated_text,
                "asr_text": fr.asr_text,
                "book_text": fr.book_text,
                "speaker_id": _extract_narrator(wav_path),
                "duration": fr.duration_sec,
                "cer": fr.cer,
                "chars_per_sec": fr.chars_per_sec,
                "georgian_ratio": fr.georgian_ratio,
            })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_total
    total_duration_kept = sum(e["duration"] for e in manifest_entries)

    stats = {
        "book_id": book_id,
        "status": "complete",
        "wav_path": wav_path,
        "total_chunks": len(chunks),
        "transcribed": len(transcriptions),
        "passed_filter": len(manifest_entries),
        "rejected": len(chunks) - len(manifest_entries),
        "pass_rate": len(manifest_entries) / len(chunks) if chunks else 0,
        "total_duration_sec": sum(c.duration_sec for c in chunks),
        "kept_duration_sec": total_duration_kept,
        "kept_duration_min": total_duration_kept / 60,
        "filter_summary": filter_summary,
        "timing": {
            "chunk_sec": chunk_time,
            "transcribe_sec": transcribe_time,
            "align_sec": align_time,
            "filter_sec": filter_time,
            "total_sec": total_time,
        },
        "manifest_path": str(manifest_path),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Book complete: {book_id}")
    logger.info(f"  Chunks: {len(chunks)} total, {len(manifest_entries)} kept ({stats['pass_rate']:.1%})")
    logger.info(f"  Duration kept: {total_duration_kept/60:.1f} min")
    logger.info(f"  Time: {total_time:.1f}s")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Speech corpus pipeline")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing WAV files")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Path to parquet with reference texts")
    parser.add_argument("--output-dir", type=str, default="./data/corpus_out",
                        help="Output directory")
    parser.add_argument("--books", type=str, nargs="+", required=True,
                        help="Book titles to process")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for ASR")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription if cached results exist")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)

    # Load book metadata
    logger.info(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    logger.info(f"  {len(df)} books in parquet")

    audio_dir = Path(args.audio_dir)
    all_stats = []

    for title in args.books:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Processing: {title}")
        logger.info(f"{'#'*60}")

        # Find book text
        match = df[df["title"] == title]
        if len(match) == 0:
            logger.error(f"Title '{title}' not found in parquet!")
            continue

        book_text = match["text"].iloc[0]
        book_id = _sanitize_book_id(title)

        # Find WAV
        try:
            wav_path = _find_wav_for_title(audio_dir, title)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        logger.info(f"  WAV: {wav_path}")
        logger.info(f"  Text: {len(book_text)} chars")
        logger.info(f"  Book ID: {book_id}")

        stats = process_book(
            wav_path=wav_path,
            book_text=book_text,
            book_id=book_id,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            skip_transcribe=args.skip_transcribe,
        )
        all_stats.append(stats)

    # Save overall summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'#'*60}")
    logger.info("ALL DONE")
    for s in all_stats:
        if s.get("status") == "complete":
            logger.info(
                f"  {s['book_id']}: {s['passed_filter']}/{s['total_chunks']} chunks, "
                f"{s['kept_duration_min']:.1f} min kept"
            )
    logger.info(f"Summary: {summary_path}")
    logger.info(f"{'#'*60}")


if __name__ == "__main__":
    main()
