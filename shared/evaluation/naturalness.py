"""
Naturalness evaluation via automatic MOS prediction.

Uses UTMOS — predicts Mean Opinion Score using wav2vec 2.0 + ensemble.
No reference audio needed.

CAVEAT: UTMOS is trained on English data. Absolute scores are NOT calibrated
for Georgian. Use only for relative ranking between models, not as an
absolute quality measure. This is clearly flagged in all reports.

Usage:
    from shared.evaluation.naturalness import compute_utmos

    results = compute_utmos(generated_dir="outputs/")
"""

from pathlib import Path

import torch
import torchaudio


def compute_utmos(generated_dir: str, device: str = "cuda") -> dict:
    """
    Compute UTMOS scores for all generated audio files.

    No reference audio needed — scores are predicted from the generated audio alone.

    Args:
        generated_dir: Directory containing generated .wav files
        device: "cuda" or "cpu"

    Returns:
        Dict with per-sample scores and aggregate statistics
    """
    gen_path = Path(generated_dir)
    audio_files = sorted(gen_path.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {generated_dir}")

    # Load UTMOS predictor
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
    )
    predictor = predictor.to(device)
    predictor.eval()

    per_sample = {}
    for audio_file in audio_files:
        wav, sr = torchaudio.load(str(audio_file))
        # Resample to 16kHz (UTMOS expects 16kHz)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.to(device)

        with torch.no_grad():
            score = predictor(wav, sr=16000).item()

        per_sample[audio_file.name] = {"utmos": score}

    scores = [v["utmos"] for v in per_sample.values()]
    return {
        "metric": "utmos",
        "caveat": "English-biased. Use for relative ranking only, not absolute quality.",
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "median_score": sorted(scores)[len(scores) // 2] if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "num_samples": len(scores),
        "per_sample": per_sample,
    }
