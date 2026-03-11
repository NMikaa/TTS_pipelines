"""
Speaker similarity evaluation via ECAPA-TDNN embeddings.

ONLY meaningful for voice-cloning models where a reference speaker clip
was provided as input. Measures whether the generated audio sounds like
the target speaker.

NOT meaningful when comparing across different speakers (e.g. TTS output
in one voice vs FLEURS recording in another voice). In that case, you're
measuring speaker difference, not synthesis quality.

Models that support voice cloning: F5-TTS, Orpheus, Qwen3-TTS
Models that don't: CSM-1B (multi-speaker, no voice cloning condition)

Usage:
    from shared.evaluation.speaker_similarity import compute_speaker_similarity

    # Only use this if the model was given reference_audio as a voice prompt
    results = compute_speaker_similarity(
        generated_dir="outputs/",
        reference_dir="voice_prompts/",
        pairs={"gen_001.wav": "prompt_001.wav", ...},
    )
"""

from pathlib import Path

import torch
import torchaudio


def compute_speaker_similarity(
    generated_dir: str,
    reference_dir: str,
    pairs: dict[str, str],
    device: str = "cuda",
) -> dict:
    """
    Compute speaker similarity between generated audio and the voice prompt
    that was used to condition the model.

    Args:
        generated_dir: Directory containing generated .wav files
        reference_dir: Directory containing voice prompt .wav files
        pairs: Dict mapping generated filename -> voice prompt filename
        device: "cuda" or "cpu"

    Returns:
        Dict with per-pair cosine similarity and aggregate statistics
    """
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    gen_path = Path(generated_dir)
    ref_path = Path(reference_dir)

    per_sample = {}
    for gen_name, ref_name in pairs.items():
        gen_file = gen_path / gen_name
        ref_file = ref_path / ref_name

        if not gen_file.exists() or not ref_file.exists():
            continue

        gen_emb = classifier.encode_batch(classifier.load_audio(str(gen_file)).unsqueeze(0))
        ref_emb = classifier.encode_batch(classifier.load_audio(str(ref_file)).unsqueeze(0))

        similarity = torch.nn.functional.cosine_similarity(
            gen_emb.squeeze(), ref_emb.squeeze(), dim=0
        ).item()

        per_sample[gen_name] = {
            "similarity": similarity,
            "reference": ref_name,
        }

    sims = [v["similarity"] for v in per_sample.values()]
    return {
        "metric": "speaker_similarity",
        "embedding_model": "ecapa-tdnn",
        "condition": "voice_cloning",
        "mean_similarity": sum(sims) / len(sims) if sims else 0.0,
        "median_similarity": sorted(sims)[len(sims) // 2] if sims else 0.0,
        "num_pairs": len(sims),
        "per_sample": per_sample,
    }
