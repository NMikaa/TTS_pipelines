"""
Evaluate a trained F5-TTS model on the FLEURS Georgian test set.

Metrics:
- CER: round-trip intelligibility via Meta Omnilingual ASR
- UTMOS: naturalness (English-biased, directional only)
- FAD: distributional similarity to real Georgian speech
- Speaker similarity: only if --voice-prompt-dir is provided (voice cloning)

Usage:
    python evaluate.py --checkpoint ckpts/georgian_tts/model_110000.pt --data-dir ./data --output-dir results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import get_splits
from shared.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate F5-TTS on Georgian")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--reference-audio-dir", type=str, default=None,
                        help="Directory with real Georgian speech for FAD (e.g. FLEURS audio)")
    parser.add_argument("--voice-prompt-dir", type=str, default=None,
                        help="Directory with voice prompts used for cloning (enables speaker sim)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated"

    _, _, test_entries = get_splits(args.data_dir)
    references = {f"{e['id']}.wav": e["text"] for e in test_entries}

    print("Step 1: Generating test set audio...")
    from infer import load_georgian_model, infer_eval, get_default_ref
    model, vocoder = load_georgian_model(args.checkpoint, device=args.device)
    ref_audio, ref_text = get_default_ref(args.data_dir)
    infer_eval(model, vocoder, str(Path(args.data_dir) / "test_manifest.json"),
               str(generated_dir), ref_audio, ref_text, device=args.device)

    print("Step 2: Running evaluation...")
    results = run_full_evaluation(
        generated_dir=str(generated_dir),
        references=references,
        reference_audio_dir=args.reference_audio_dir,
        device=args.device,
        output_path=str(output_dir / "evaluation.json"),
    )

    print("\n" + "=" * 50)
    print("F5-TTS EVALUATION RESULTS")
    print("=" * 50)
    for key, label, val_key in [
        ("intelligibility", "CER", "mean_cer"),
        ("naturalness", "UTMOS", "mean_score"),
        ("fad", "FAD", "fad_score"),
        ("speaker_similarity", "Speaker sim", "mean_similarity"),
    ]:
        if key in results and val_key in results.get(key, {}):
            print(f"  {label:20s} {results[key][val_key]:.4f}")


if __name__ == "__main__":
    main()
