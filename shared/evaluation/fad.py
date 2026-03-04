"""
Fréchet Audio Distance (FAD) evaluation.

Measures distributional similarity between generated and reference audio
collections. Lower FAD = closer to the real audio distribution.

Uses VGGish embeddings by default. Requires a reasonably large set of samples
(50+) for stable statistics.

Usage:
    from shared.evaluation.fad import compute_fad

    result = compute_fad(generated_dir="outputs/", reference_dir="data/audio/test/")
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import linalg


def _load_audio_batch(audio_dir: str, target_sr: int = 16000) -> list[torch.Tensor]:
    """Load all .wav files from a directory, resampled to target_sr."""
    audio_path = Path(audio_dir)
    files = sorted(audio_path.glob("*.wav"))
    waveforms = []
    for f in files:
        wav, sr = torchaudio.load(str(f))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        waveforms.append(wav.squeeze(0))
    return waveforms


def _get_embeddings_vggish(waveforms: list[torch.Tensor], sr: int = 16000) -> np.ndarray:
    """Extract VGGish embeddings for a list of waveforms."""
    model = torch.hub.load("harritaylor/torchvggish", "vggish")
    model.eval()

    embeddings = []
    for wav in waveforms:
        with torch.no_grad():
            emb = model.forward(wav.numpy(), sr)
            # VGGish returns [T, 128], average over time
            emb = emb.mean(dim=0).numpy()
            embeddings.append(emb)

    return np.stack(embeddings)


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fad(generated_dir: str, reference_dir: str) -> dict:
    """
    Compute Fréchet Audio Distance between generated and reference sets.

    Args:
        generated_dir: Directory with generated .wav files
        reference_dir: Directory with reference .wav files

    Returns:
        Dict with FAD score and sample counts
    """
    gen_waveforms = _load_audio_batch(generated_dir)
    ref_waveforms = _load_audio_batch(reference_dir)

    if len(gen_waveforms) < 10 or len(ref_waveforms) < 10:
        raise ValueError(
            f"FAD needs at least 10 samples per set. "
            f"Got {len(gen_waveforms)} generated, {len(ref_waveforms)} reference."
        )

    gen_embs = _get_embeddings_vggish(gen_waveforms)
    ref_embs = _get_embeddings_vggish(ref_waveforms)

    mu_gen, sigma_gen = gen_embs.mean(axis=0), np.cov(gen_embs, rowvar=False)
    mu_ref, sigma_ref = ref_embs.mean(axis=0), np.cov(ref_embs, rowvar=False)

    fad = _frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

    return {
        "metric": "fad",
        "fad_score": fad,
        "embedding_model": "vggish",
        "num_generated": len(gen_waveforms),
        "num_reference": len(ref_waveforms),
    }
