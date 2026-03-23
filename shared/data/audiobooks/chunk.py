"""
VAD-based chunking of long-form audio using Silero VAD.

Splits audio at silence boundaries into segments suitable for TTS training.
Target: 3-25 seconds per chunk, aiming for ~8-10s average.
"""

import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Silero VAD requires 16kHz

# Chunking parameters
MIN_CHUNK_SEC = 1.5
MAX_CHUNK_SEC = 40.0
MIN_SILENCE_MS = 500    # Split at silences >= 500ms
MERGE_THRESHOLD_MS = 300  # Merge segments separated by < 300ms
PADDING_SEC = 0.1         # Keep 100ms padding around chunks


@dataclass
class AudioChunk:
    start_sec: float
    end_sec: float
    duration_sec: float
    chunk_id: str


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    return model, utils


def _get_speech_timestamps(model, utils, waveform: torch.Tensor, sr: int):
    """Run Silero VAD and return speech timestamps."""
    get_speech_timestamps = utils[0]

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform_16k = resampler(waveform)
    else:
        waveform_16k = waveform

    # Squeeze to 1D
    wav = waveform_16k.squeeze()

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=int(PADDING_SEC * 1000),
        min_speech_duration_ms=int(MIN_CHUNK_SEC * 1000 * 0.5),  # Allow short segments, filter later
    )

    # Convert from 16kHz sample indices to seconds
    result = []
    for ts in timestamps:
        start_sec = ts["start"] / SAMPLE_RATE
        end_sec = ts["end"] / SAMPLE_RATE
        result.append({"start": start_sec, "end": end_sec})

    return result


def _merge_short_segments(segments: list, min_gap_sec: float = MERGE_THRESHOLD_MS / 1000) -> list:
    """Merge segments separated by very short gaps."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_gap_sec:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def _split_long_segments(segments: list, waveform: torch.Tensor, sr: int,
                         model, utils) -> list:
    """Re-split segments that exceed MAX_CHUNK_SEC using tighter VAD."""
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration <= MAX_CHUNK_SEC:
            result.append(seg)
            continue

        # Extract this segment's audio and run VAD with tighter silence threshold
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        sub_wav = waveform[:, start_sample:end_sample]

        get_speech_timestamps = utils[0]
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            sub_16k = resampler(sub_wav)
        else:
            sub_16k = sub_wav

        sub_timestamps = get_speech_timestamps(
            sub_16k.squeeze(),
            model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=200,  # Tighter threshold for long segments
            speech_pad_ms=50,
            min_speech_duration_ms=1000,
        )

        if len(sub_timestamps) <= 1:
            # Can't split further - just use MAX_CHUNK_SEC windows
            offset = 0.0
            while offset < duration:
                chunk_end = min(offset + MAX_CHUNK_SEC, duration)
                result.append({
                    "start": seg["start"] + offset,
                    "end": seg["start"] + chunk_end,
                })
                offset = chunk_end
        else:
            # Use sub-timestamps, offset back to original timeline
            for ts in sub_timestamps:
                result.append({
                    "start": seg["start"] + ts["start"] / SAMPLE_RATE,
                    "end": seg["start"] + ts["end"] / SAMPLE_RATE,
                })

    return result


def chunk_audiobook(
    audio_path: str,
    output_dir: str,
    book_id: str,
    original_sr: int = None,
) -> list[AudioChunk]:
    """
    Chunk a long-form WAV into shorter segments.

    Args:
        audio_path: Path to WAV file.
        output_dir: Directory to save chunk WAV files.
        book_id: Identifier for this recording (used in chunk filenames).
        original_sr: If set, keep this sample rate. Otherwise use file's native rate.

    Returns:
        List of AudioChunk objects with paths to saved chunk files.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_duration = waveform.shape[1] / sr
    logger.info(f"  Duration: {total_duration / 60:.1f} min, SR: {sr}")

    # Load VAD model
    logger.info("  Loading Silero VAD...")
    model, utils = load_silero_vad()

    # Step 1: Get speech timestamps
    logger.info("  Running VAD...")
    segments = _get_speech_timestamps(model, utils, waveform, sr)
    logger.info(f"  VAD found {len(segments)} speech segments")

    # Step 2: Merge very close segments
    segments = _merge_short_segments(segments)
    logger.info(f"  After merging close segments: {len(segments)}")

    # Step 3: Split segments that are too long
    segments = _split_long_segments(segments, waveform, sr, model, utils)
    logger.info(f"  After splitting long segments: {len(segments)}")

    # Step 4: Filter by duration
    chunks = []
    skipped_short = 0
    skipped_long = 0

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]

        if duration < MIN_CHUNK_SEC:
            skipped_short += 1
            continue
        if duration > MAX_CHUNK_SEC * 1.1:  # Small tolerance
            skipped_long += 1
            continue

        chunk_id = f"{book_id}_chunk_{i:05d}"
        chunk_path = out_path / f"{chunk_id}.wav"

        # Extract and save chunk
        start_sample = int(seg["start"] * sr)
        end_sample = min(int(seg["end"] * sr), waveform.shape[1])
        chunk_wav = waveform[:, start_sample:end_sample]

        torchaudio.save(str(chunk_path), chunk_wav, sr)

        chunks.append(AudioChunk(
            start_sec=seg["start"],
            end_sec=seg["end"],
            duration_sec=duration,
            chunk_id=chunk_id,
        ))

    logger.info(
        f"  Saved {len(chunks)} chunks "
        f"(skipped {skipped_short} short, {skipped_long} long)"
    )

    if chunks:
        durations = [c.duration_sec for c in chunks]
        logger.info(
            f"  Duration stats: min={min(durations):.1f}s, "
            f"max={max(durations):.1f}s, "
            f"mean={np.mean(durations):.1f}s, "
            f"total={sum(durations) / 60:.1f}min"
        )

    return chunks
