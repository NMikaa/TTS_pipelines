"""Audio I/O helpers using soundfile (avoids torchaudio backend issues)."""

import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass


@dataclass
class AudioInfo:
    sample_rate: int
    num_frames: int


def load(path: str) -> tuple:
    """Load audio file, returns (waveform_tensor [1, N], sample_rate)."""
    audio_np, sr = sf.read(path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    return waveform, sr


def save(path: str, waveform: torch.Tensor, sample_rate: int):
    """Save waveform tensor to file."""
    audio_np = waveform.squeeze().numpy()
    sf.write(path, audio_np, sample_rate)


def info(path: str) -> AudioInfo:
    """Get audio file info."""
    with sf.SoundFile(path) as f:
        return AudioInfo(sample_rate=f.samplerate, num_frames=f.frames)
