"""
Unified evaluation runner.

Core metrics (always run):
- CER: round-trip intelligibility via Meta Omnilingual ASR (no reference audio needed)
- FAD: distributional similarity to real Georgian speech
- UTMOS: naturalness prediction (English-biased, directional only)

Conditional metric (voice-cloning models only):
- Speaker similarity: ECAPA-TDNN cosine sim (only when a voice prompt was used)

Usage:
    from shared.evaluation.evaluate import run_full_evaluation

    results = run_full_evaluation(
        generated_dir="outputs/",
        references={"sample_001.wav": "გამარჯობა", ...},
        reference_audio_dir="data/fleurs_audio/",  # for FAD
    )
"""

import json
from pathlib import Path

from .intelligibility import compute_cer
from .naturalness import compute_utmos
from .speaker_similarity import compute_speaker_similarity
from .fad import compute_fad


def run_full_evaluation(
    generated_dir: str,
    references: dict[str, str],
    reference_audio_dir: str | None = None,
    voice_prompt_dir: str | None = None,
    voice_prompt_pairs: dict[str, str] | None = None,
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """
    Run evaluation metrics on generated audio.

    Args:
        generated_dir: Directory with generated .wav files
        references: Dict mapping generated filename -> ground truth text (for CER)
        reference_audio_dir: Directory with real Georgian speech for FAD comparison
        voice_prompt_dir: Directory with voice prompts used for cloning (optional)
        voice_prompt_pairs: Dict mapping gen filename -> prompt filename (optional)
        device: "cuda" or "cpu"
        output_path: If set, save results as JSON

    Returns:
        Dict with all metric results
    """
    results = {}

    # 1. CER (round-trip: TTS → ASR → compare to input text)
    print("Computing CER (Meta Omnilingual ASR)...")
    try:
        results["intelligibility"] = compute_cer(generated_dir, references)
        print(f"  Mean CER: {results['intelligibility']['mean_cer']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
        results["intelligibility"] = {"error": str(e)}

    # 2. UTMOS (reference-free naturalness — English-biased caveat)
    print("Computing UTMOS (English-biased, directional only)...")
    try:
        results["naturalness"] = compute_utmos(generated_dir, device=device)
        print(f"  Mean UTMOS: {results['naturalness']['mean_score']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
        results["naturalness"] = {"error": str(e)}

    # 3. FAD (distribution-level comparison to real speech)
    if reference_audio_dir:
        print("Computing FAD...")
        try:
            results["fad"] = compute_fad(generated_dir, reference_audio_dir)
            print(f"  FAD: {results['fad']['fad_score']:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            results["fad"] = {"error": str(e)}

    # 4. Speaker similarity (ONLY for voice-cloning models)
    if voice_prompt_dir and voice_prompt_pairs:
        print("Computing speaker similarity (voice cloning condition)...")
        try:
            results["speaker_similarity"] = compute_speaker_similarity(
                generated_dir, voice_prompt_dir, voice_prompt_pairs, device=device
            )
            print(f"  Mean similarity: {results['speaker_similarity']['mean_similarity']:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            results["speaker_similarity"] = {"error": str(e)}

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary = {}
        for key, val in results.items():
            if isinstance(val, dict):
                summary[key] = {k: v for k, v in val.items() if k != "per_sample"}
            else:
                summary[key] = val
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results
