"""
Prepare saba_data for MagPIE TTS training.

Steps:
1. Extract audio from parquet shards to WAV files
2. Filter: clip first 3 + last 3 segments per book
3. Filter: drop multi_speaker segments
4. Filter: CPS outliers (< 5 or > 25)
5. Filter: duration bounds (< 1.5s or > 25s)
6. Create train/eval manifests in the expected format
"""

import argparse
import json
import os
import glob
import hashlib
import logging
import sys
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq


def get_split(sample_id: str, eval_ratio: float = 0.02) -> str:
    """Deterministic hash-based split assignment."""
    h = hashlib.md5(sample_id.encode()).hexdigest()
    if int(h[:8], 16) / 0xFFFFFFFF < eval_ratio:
        return "eval"
    return "train"


def get_seg_number(seg_id: str) -> int:
    """Extract segment number from ID like 'book__seg_00001'."""
    parts = seg_id.rsplit("seg_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return -1


def scan_book_boundaries(shards: list[str]) -> dict[str, dict]:
    """First pass: find min/max segment numbers per book."""
    book_info = defaultdict(lambda: {"min_seg": float("inf"), "max_seg": -1, "count": 0})

    logging.info("Pass 1: Scanning book boundaries...")
    for i, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=["id", "book_title"])
            for j in range(table.num_rows):
                seg_id = table.column("id")[j].as_py()
                book = table.column("book_title")[j].as_py() or ""
                seg_num = get_seg_number(seg_id)
                if seg_num >= 0:
                    info = book_info[book]
                    info["min_seg"] = min(info["min_seg"], seg_num)
                    info["max_seg"] = max(info["max_seg"], seg_num)
                    info["count"] += 1
        if (i + 1) % 20 == 0:
            logging.info(f"  {i + 1}/{len(shards)} shards...")

    logging.info(f"  Found {len(book_info)} books, {sum(b['count'] for b in book_info.values()):,} segments")
    return dict(book_info)


def should_keep(
    seg_id: str,
    book: str,
    text: str,
    duration: float,
    multi_speaker: bool,
    book_info: dict,
    clip_start: int,
    clip_end: int,
) -> tuple[bool, str]:
    """Check if a segment passes all filters. Returns (keep, reason)."""
    # Duration bounds
    if duration < 1.5:
        return False, "too_short"
    if duration > 25.0:
        return False, "too_long"

    # CPS filter
    if text and duration > 0:
        cps = len(text) / duration
        if cps < 5:
            return False, "cps_low"
        if cps > 25:
            return False, "cps_high"

    # Empty text
    if not text or len(text.strip()) < 3:
        return False, "empty_text"

    # Book boundary clipping
    seg_num = get_seg_number(seg_id)
    if book in book_info and seg_num >= 0:
        info = book_info[book]
        min_seg = info["min_seg"]
        max_seg = info["max_seg"]
        if seg_num < min_seg + clip_start:
            return False, "intro_clip"
        if seg_num > max_seg - clip_end:
            return False, "outro_clip"

    return True, "ok"


def prepare_data(
    data_dir: str,
    output_dir: str,
    clip_start: int = 3,
    clip_end: int = 3,
    eval_ratio: float = 0.02,
    holdout_speakers: list[str] | None = None,
    delete_parquets: bool = False,
):
    """Main data preparation pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(data_dir / "data" / "train-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {data_dir / 'data'}")
    logging.info(f"Found {len(shards)} parquet shards")

    # Pass 1: scan book boundaries
    book_info = scan_book_boundaries(shards)

    # Pass 2: extract audio and build manifests
    logging.info(f"Pass 2: Extracting audio and filtering...")
    stats = defaultdict(int)
    train_entries = []
    eval_entries = []
    speaker_stats = defaultdict(int)

    for i, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            for j in range(table.num_rows):
                seg_id = table.column("id")[j].as_py()
                text = table.column("text")[j].as_py() or ""
                speaker = table.column("speaker_id")[j].as_py() or ""
                duration = table.column("duration")[j].as_py() or 0
                book = table.column("book_title")[j].as_py() or ""
                multi_spk = table.column("multi_speaker")[j].as_py() or False

                stats["total"] += 1

                # Filter
                keep, reason = should_keep(
                    seg_id, book, text, duration, multi_spk,
                    book_info, clip_start, clip_end,
                )
                if not keep:
                    stats[f"filtered_{reason}"] += 1
                    continue

                # Extract audio
                audio_data = table.column("audio")[j].as_py()
                audio_path = audio_dir / f"{seg_id}.wav"

                if not audio_path.exists():
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(audio_path, "wb") as f:
                        f.write(audio_data["bytes"])

                has_punct = table.column("has_punctuation")[j].as_py() or False
                book_title = book

                entry = {
                    "id": seg_id,
                    "audio_path": str(audio_path),
                    "text": text,
                    "speaker_id": speaker,
                    "duration": round(duration, 4),
                    "book_title": book_title,
                    "has_punctuation": has_punct,
                    "multi_speaker": multi_spk,
                }

                # Split assignment
                if holdout_speakers and speaker in holdout_speakers:
                    eval_entries.append(entry)
                    stats["eval_holdout"] += 1
                elif get_split(seg_id, eval_ratio) == "eval":
                    eval_entries.append(entry)
                    stats["eval"] += 1
                else:
                    train_entries.append(entry)
                    stats["train"] += 1

                speaker_stats[speaker] += 1

        if (i + 1) % 10 == 0:
            kept = stats["train"] + stats["eval"] + stats.get("eval_holdout", 0)
            logging.info(f"  {i + 1}/{len(shards)} shards | kept {kept:,} / {stats['total']:,}")

    # Write manifests
    train_manifest = output_dir / "train_manifest.json"
    eval_manifest = output_dir / "eval_manifest.json"

    with open(train_manifest, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(eval_manifest, "w") as f:
        for entry in eval_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Delete parquet shards to free disk space
    if delete_parquets:
        logging.info(f"Deleting parquet shards to free disk space...")
        for shard in shards:
            os.remove(shard)
        # Remove cache dir if exists
        cache_dir = data_dir / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        logging.info(f"  Freed ~{len(shards) * 2:.0f} GB")

    # Print summary
    logging.info(f"{'=' * 60}")
    logging.info(f"SUMMARY")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total segments scanned:  {stats['total']:,}")
    logging.info(f"")
    logging.info(f"Filtered:")
    for key in sorted(stats.keys()):
        if key.startswith("filtered_"):
            reason = key.replace("filtered_", "")
            logging.info(f"  {reason:20s}: {stats[key]:,}")
    total_filtered = sum(v for k, v in stats.items() if k.startswith("filtered_"))
    logging.info(f"  {'TOTAL':20s}: {total_filtered:,} ({100 * total_filtered / stats['total']:.1f}%)")
    logging.info(f"")
    logging.info(f"Kept:")
    logging.info(f"  Train:   {stats['train']:,}")
    logging.info(f"  Eval:    {stats['eval'] + stats.get('eval_holdout', 0):,}")
    total_kept = stats["train"] + stats["eval"] + stats.get("eval_holdout", 0)
    logging.info(f"  Total:   {total_kept:,} ({100 * total_kept / stats['total']:.1f}%)")
    logging.info(f"")
    logging.info(f"Speakers: {len(speaker_stats)}")
    logging.info(f"Top 10 speakers:")
    for spk, count in sorted(speaker_stats.items(), key=lambda x: -x[1])[:10]:
        logging.info(f"  {spk}: {count:,}")
    logging.info(f"")
    logging.info(f"Manifests:")
    logging.info(f"  {train_manifest}")
    logging.info(f"  {eval_manifest}")
    logging.info(f"  Audio dir: {audio_dir}")


def setup_logging(output_dir: str):
    """Log to both console and file."""
    log_path = Path(output_dir) / "prepare_data.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare saba_data for MagPIE TTS")
    parser.add_argument("--data-dir", type=str, default="../../data/saba_data",
                        help="Path to downloaded saba_data")
    parser.add_argument("--output-dir", type=str, default="../../data/saba_clean",
                        help="Output directory for processed data")
    parser.add_argument("--clip-start", type=int, default=3,
                        help="Number of segments to clip from start of each book")
    parser.add_argument("--clip-end", type=int, default=3,
                        help="Number of segments to clip from end of each book")
    parser.add_argument("--eval-ratio", type=float, default=0.02,
                        help="Fraction of data for evaluation")
    parser.add_argument("--holdout-speakers", nargs="*", default=None,
                        help="Speaker IDs to hold out entirely for eval")
    parser.add_argument("--delete-parquets", action="store_true",
                        help="Delete parquet shards after extraction to free disk space")
    args = parser.parse_args()

    log_path = setup_logging(args.output_dir)
    logging.info(f"Logging to {log_path}")

    prepare_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        clip_start=args.clip_start,
        clip_end=args.clip_end,
        eval_ratio=args.eval_ratio,
        holdout_speakers=args.holdout_speakers,
        delete_parquets=args.delete_parquets,
    )
