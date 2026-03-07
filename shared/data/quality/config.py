"""Pipeline configuration — thresholds, defaults, stage order."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import logging

DEFAULT_STAGES = [
    "standardize",
    "enhance",
    "nisqa_filter",
    "duration_outlier",
    "transcript_verify",
    "speaker_select",
]

# Audio
TARGET_SR = 24000
TARGET_LUFS = -23.0
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 30.0

# NISQA
NISQA_THRESHOLD = 0.0

# Silence trim (sox-style)
SILENCE_THRESHOLD_DB = -55
SILENCE_MIN_DURATION = 0.1  # seconds
SILENCE_PADDING = 0.1  # seconds

# Duration outlier (Emilia IQR)
# No threshold — uses IQR method

# Transcript verify
CER_THRESHOLD = 0.20

# Speaker select
MIN_SPEAKER_DURATION_SEC = 1400.0
MAX_SPEAKERS = 50


@dataclass
class PipelineContext:
    """Shared context passed to every stage."""
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    audio_dir: Path
    device: str
    logger: logging.Logger
