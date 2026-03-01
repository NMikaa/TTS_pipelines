"""
Word-level alignment processing for CSM-1B streaming training.

Processes CTM alignment files to create word-level timestamps with:
- Weighted gap distribution between words
- Short word merging (< 80ms)
- Frame-snapped audio slicing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class AlignmentProcessor:
    """Process CTM alignment files and create word-level timestamps."""

    def __init__(self, alignment_base_path: str):
        self.alignment_base_path = Path(alignment_base_path)
        self.outputs_dir = self.alignment_base_path / "outputs"
        self.alignment_lookup = self._build_alignment_lookup()
        print(f"Indexed {len(self.alignment_lookup)} alignment files")

    def _build_alignment_lookup(self) -> Dict:
        """Build lookup dictionary mapping audio paths to alignment files."""
        print("Building alignment lookup index...")
        lookup = {}
        batch_dirs = sorted(self.outputs_dir.glob("batch_*"))

        for batch_dir in tqdm(batch_dirs, desc="Indexing alignments"):
            manifest_files = list(batch_dir.glob("manifest_*_with_output_file_paths.json"))
            if not manifest_files:
                continue

            manifest_path = manifest_files[0]
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    audio_path = item["audio_filepath"]
                    lookup[audio_path] = item

        return lookup

    def find_alignment(self, audio_path: str) -> Optional[Dict]:
        return self.alignment_lookup.get(audio_path, None)

    @staticmethod
    def read_ctm(ctm_path: str) -> List[Dict]:
        """Read CTM file and return list of words with timestamps."""
        words = []
        with open(ctm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    start = float(parts[2])
                    duration = float(parts[3])
                    words.append({
                        "word": parts[4],
                        "start": start,
                        "duration": duration,
                        "end": start + duration,
                    })
        return words

    @staticmethod
    def adjust_word_boundaries_weighted(
        words: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Apply weighted gap distribution to adjust word boundaries.

        Gap between two words is distributed inversely proportional to
        their durations, so shorter words absorb a larger share of the gap.
        """
        if not words:
            return words

        adjusted = []
        for i, word in enumerate(words):
            adj = word.copy()

            if i == 0:
                adj["start"] = 0.0
            else:
                prev_end = words[i - 1]["end"]
                gap = word["start"] - prev_end
                if gap > 0:
                    prev_dur = words[i - 1]["duration"]
                    curr_dur = word["duration"]
                    w_prev = 1.0 / prev_dur if prev_dur > 0 else 1.0
                    w_curr = 1.0 / curr_dur if curr_dur > 0 else 1.0
                    total_w = w_prev + w_curr
                    prev_share = (w_prev / total_w) * gap
                    curr_share = (w_curr / total_w) * gap
                    adj["start"] = word["start"] - curr_share
                    if adjusted:
                        adjusted[-1]["end"] = prev_end + prev_share
                        adjusted[-1]["duration"] = (
                            adjusted[-1]["end"] - adjusted[-1]["start"]
                        )

            if i == len(words) - 1:
                adj["end"] = total_duration

            adj["duration"] = adj["end"] - adj["start"]
            adjusted.append(adj)

        return adjusted

    @staticmethod
    def merge_short_words(
        words: List[Dict], min_duration: float = 0.08
    ) -> List[Dict]:
        """Merge words shorter than min_duration with adjacent words."""
        if not words:
            return words

        merged = []
        i = 0
        while i < len(words):
            current = words[i].copy()
            if current["duration"] < min_duration:
                if i + 1 < len(words):
                    nxt = words[i + 1]
                    merged.append({
                        "word": current["word"] + " " + nxt["word"],
                        "start": current["start"],
                        "end": nxt["end"],
                        "duration": nxt["end"] - current["start"],
                    })
                    i += 2
                    continue
                elif merged:
                    prev = merged[-1]
                    prev["word"] = prev["word"] + " " + current["word"]
                    prev["end"] = current["end"]
                    prev["duration"] = prev["end"] - prev["start"]
                    i += 1
                    continue
            merged.append(current)
            i += 1

        return merged

    def process_paths_with_durations(
        self,
        audio_paths: List[str],
        audio_durations: Dict[str, float],
        min_duration: float = 0.08,
    ) -> Dict[str, List[Dict]]:
        """Process audio paths and return word-level timestamps.

        Args:
            audio_paths: List of audio file paths.
            audio_durations: Mapping of audio_path -> duration in seconds.
            min_duration: Minimum word duration (default 80ms).

        Returns:
            Dict mapping audio_path to list of {word, start, end}.
        """
        result = {}
        stats = {
            "processed": 0,
            "skipped_no_alignment": 0,
            "skipped_no_ctm": 0,
            "skipped_no_duration": 0,
            "errors": 0,
            "total_words": 0,
            "merged_words": 0,
        }

        for audio_path in tqdm(audio_paths, desc="Processing alignments"):
            try:
                if audio_path not in audio_durations:
                    stats["skipped_no_duration"] += 1
                    continue

                alignment_info = self.find_alignment(audio_path)
                if alignment_info is None:
                    stats["skipped_no_alignment"] += 1
                    continue

                ctm_path = alignment_info.get("words_level_ctm_filepath")
                if not ctm_path or not Path(ctm_path).exists():
                    stats["skipped_no_ctm"] += 1
                    continue

                total_duration = audio_durations[audio_path]
                original = self.read_ctm(ctm_path)
                adjusted = self.adjust_word_boundaries_weighted(original, total_duration)
                words_before = len(adjusted)
                final = self.merge_short_words(adjusted, min_duration=min_duration)
                stats["merged_words"] += words_before - len(final)

                result[audio_path] = [
                    {"word": w["word"], "start": w["start"], "end": w["end"]}
                    for w in final
                ]
                stats["processed"] += 1
                stats["total_words"] += len(final)

            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                stats["errors"] += 1

        print(f"\nAlignment stats: {stats['processed']}/{len(audio_paths)} processed, "
              f"{stats['total_words']} words, {stats['merged_words']} merged")
        return result
