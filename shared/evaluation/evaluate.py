"""
Unified evaluation runner.

Metrics:
- CER: round-trip intelligibility via Meta Omnilingual ASR
- WER: round-trip word error rate via Meta Omnilingual ASR

Usage:
    from shared.evaluation.evaluate import run_full_evaluation

    results = run_full_evaluation(
        generated_dir="outputs/",
        references={"sample_001.wav": "გამარჯობა", ...},
    )
"""

import json
from pathlib import Path

from .intelligibility import compute_cer


def run_full_evaluation(
    generated_dir: str,
    references: dict[str, str],
    device: str = "cuda",
    output_path: str | None = None,
) -> dict:
    """
    Run evaluation metrics on generated audio.

    Args:
        generated_dir: Directory with generated .wav files
        references: Dict mapping generated filename -> ground truth text
        device: "cuda" or "cpu"
        output_path: If set, save results as JSON

    Returns:
        Dict with all metric results
    """
    results = {}

    # CER + WER (round-trip: TTS → ASR → compare to input text)
    print("Computing CER/WER (Meta Omnilingual ASR)...")
    try:
        results["intelligibility"] = compute_cer(generated_dir, references)
        print(f"  Mean CER: {results['intelligibility']['mean_cer']:.4f}")
        print(f"  Mean WER: {results['intelligibility']['mean_wer']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
        results["intelligibility"] = {"error": str(e)}

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
