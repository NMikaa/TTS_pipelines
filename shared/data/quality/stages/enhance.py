"""Stage: VoiceFixer enhancement + sox spectral noise subtraction.

Follows the Catalan TTS paper (arXiv 2410.13357):
1. VoiceFixer restores speech from noise/reverb/clipping (outputs 44.1kHz)
2. Sox spectral subtraction removes residual noise using a noise profile
   built from the last 0.5s of audio
3. Resample back to 24kHz
"""

import subprocess
import tempfile
from pathlib import Path

import torchaudio
from tqdm import tqdm

from ..audio_io import load, save, info
from ..config import PipelineContext, TARGET_SR

NAME = "enhance"
DESCRIPTION = "VoiceFixer restoration + sox spectral noise subtraction"


def run(entries, ctx: PipelineContext):
    from voicefixer import VoiceFixer
    vf = VoiceFixer()

    kept = []
    dropped = 0

    for entry in tqdm(entries, desc="Enhance"):
        audio_path = entry["audio_path"]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Step 1: VoiceFixer (outputs 44.1kHz)
                vf_out = str(Path(tmpdir) / "vf.wav")
                vf.restore(input=audio_path, output=vf_out, cuda=ctx.device.startswith("cuda"), mode=0)

                # Step 2: Sox spectral noise subtraction
                noise_profile = str(Path(tmpdir) / "noise.prof")
                denoised = str(Path(tmpdir) / "denoised.wav")

                # Get duration for trimming last 0.5s
                vf_info = info(vf_out)
                total_dur = vf_info.num_frames / vf_info.sample_rate

                # Extract last 0.5s as noise sample
                noise_start = max(0, total_dur - 0.5)
                subprocess.run(
                    ["sox", vf_out, "-n", "trim", str(noise_start), "0.5", "noiseprof", noise_profile],
                    check=True, capture_output=True,
                )

                # Apply noise reduction
                subprocess.run(
                    ["sox", vf_out, denoised, "noisered", noise_profile, "0.21"],
                    check=True, capture_output=True,
                )

                # Step 3: Resample back to 24kHz and overwrite
                waveform, sr = load(denoised)
                if sr != TARGET_SR:
                    waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
                save(audio_path, waveform, TARGET_SR)

            kept.append(entry.copy())

        except Exception as e:
            ctx.logger.debug(f"Enhance error for {entry['id']}: {e}")
            dropped += 1

    ctx.logger.info(f"Enhance: {len(kept)}/{len(entries)} kept (dropped {dropped} errors)")
    return kept
