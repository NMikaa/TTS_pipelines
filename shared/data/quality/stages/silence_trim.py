"""Stage: Sox-style silence trimming.

Follows the Catalan TTS paper (arXiv 2410.13357):
- Remove silences longer than 0.1s and below -55dB from beginning and end
- Add 0.1s padding at both start and end
"""

import subprocess
import tempfile
from pathlib import Path

import torchaudio
from tqdm import tqdm

from ..config import PipelineContext, SILENCE_THRESHOLD_DB, SILENCE_MIN_DURATION, SILENCE_PADDING

NAME = "silence_trim"
DESCRIPTION = "Remove leading/trailing silence (sox), add 0.1s padding"


def run(entries, ctx: PipelineContext):
    kept = []
    dropped = 0

    for entry in tqdm(entries, desc="Silence trim"):
        audio_path = entry["audio_path"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp_path = tmp.name

                # Sox silence trim: remove silence from beginning and end
                # silence 1 = trim from beginning: 1 occurrence of silence
                #   duration threshold (0.1s), amplitude threshold (-55dB)
                # reverse + silence 1 + reverse = trim from end
                subprocess.run(
                    [
                        "sox", audio_path, tmp_path,
                        "silence", "1", str(SILENCE_MIN_DURATION), f"{SILENCE_THRESHOLD_DB}d",
                        "reverse",
                        "silence", "1", str(SILENCE_MIN_DURATION), f"{SILENCE_THRESHOLD_DB}d",
                        "reverse",
                        "pad", str(SILENCE_PADDING), str(SILENCE_PADDING),
                    ],
                    check=True, capture_output=True,
                )

                # Check output is valid
                info = torchaudio.info(tmp_path)
                duration = info.num_frames / info.sample_rate
                if duration < 0.3:
                    dropped += 1
                    continue

                # Overwrite original
                waveform, sr = torchaudio.load(tmp_path)
                torchaudio.save(audio_path, waveform, sr)

                entry_copy = entry.copy()
                entry_copy["duration"] = duration
                kept.append(entry_copy)

        except Exception as e:
            ctx.logger.debug(f"Silence trim error for {entry['id']}: {e}")
            dropped += 1

    ctx.logger.info(f"Silence trim: {len(kept)}/{len(entries)} kept (dropped {dropped})")
    return kept
