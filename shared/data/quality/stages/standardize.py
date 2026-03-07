"""Stage: Standardize audio to 24kHz mono with LUFS normalization."""

import warnings
warnings.filterwarnings("ignore", message=".*clipped samples.*")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from ..config import PipelineContext, TARGET_SR, TARGET_LUFS, MIN_DURATION_SEC, MAX_DURATION_SEC

NAME = "standardize"
DESCRIPTION = "Resample to 24kHz mono, normalize loudness to -23 LUFS, filter by duration"


def run(entries, ctx: PipelineContext):
    import pyloudnorm as pyln

    ctx.audio_dir.mkdir(parents=True, exist_ok=True)
    meter = pyln.Meter(TARGET_SR)
    kept = []
    dropped_short = 0
    dropped_long = 0
    dropped_error = 0

    for entry in tqdm(entries, desc="Standardize"):
        try:
            audio_np, sr = sf.read(entry["audio_path"], dtype="float32")
        except Exception:
            dropped_error += 1
            continue

        # Mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Resample if needed
        if sr != TARGET_SR:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
            audio_np = waveform.squeeze(0).numpy()

        duration = len(audio_np) / TARGET_SR
        if duration < MIN_DURATION_SEC:
            dropped_short += 1
            continue
        if duration > MAX_DURATION_SEC:
            dropped_long += 1
            continue

        try:
            loudness = meter.integrated_loudness(audio_np)
            if np.isinf(loudness) or np.isnan(loudness):
                dropped_error += 1
                continue
            audio_np = pyln.normalize.loudness(audio_np, loudness, TARGET_LUFS)
        except Exception:
            dropped_error += 1
            continue

        audio_np = np.clip(audio_np, -1.0, 1.0)

        out_path = ctx.audio_dir / f"{entry['id']}.wav"
        sf.write(str(out_path), audio_np, TARGET_SR)

        new_entry = entry.copy()
        new_entry["audio_path"] = str(out_path)
        new_entry["duration"] = duration
        kept.append(new_entry)

    ctx.logger.info(
        f"Standardize: {len(kept)}/{len(entries)} kept "
        f"(dropped: {dropped_short} short, {dropped_long} long, {dropped_error} errors)"
    )
    return kept
