"""Stage: Duration outlier filter (Emilia-style IQR method).

Computes average character duration (audio_duration / num_chars) per clip,
then drops entries outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
Catches bad transcripts, non-speech audio, and tempo anomalies.
"""

import numpy as np
from tqdm import tqdm

from ..audio_io import info
from ..config import PipelineContext

NAME = "duration_outlier"
DESCRIPTION = "Filter char-duration outliers via IQR method"


def run(entries, ctx: PipelineContext):
    char_durations = []
    valid_entries = []

    for entry in entries:
        text = entry.get("text", "")
        num_chars = len(text.replace(" ", ""))
        if num_chars == 0:
            continue

        duration = entry.get("duration")
        if duration is None or duration <= 0:
            try:
                ai = info(entry["audio_path"])
                duration = ai.num_frames / ai.sample_rate
            except Exception:
                continue

        char_dur = duration / num_chars
        char_durations.append(char_dur)
        entry_copy = entry.copy()
        entry_copy["char_duration"] = char_dur
        entry_copy["duration"] = duration
        valid_entries.append(entry_copy)

    if not char_durations:
        ctx.logger.warning("No valid entries for duration outlier filter")
        return entries

    arr = np.array(char_durations)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    kept = [e for e in valid_entries if lower <= e["char_duration"] <= upper]
    dropped = len(valid_entries) - len(kept)

    ctx.logger.info(
        f"Duration outlier: {len(kept)}/{len(entries)} kept (dropped {dropped}). "
        f"Char dur (s/char): Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, "
        f"bounds=[{lower:.4f}, {upper:.4f}]"
    )
    return kept
