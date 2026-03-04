"""
Evaluate CosyVoice 3 on the FLEURS Georgian test set.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --data-dir ./data --output-dir results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits
from shared.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate CosyVoice 3 on Georgian")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--reference-audio-dir", type=str, default=None)
    parser.add_argument("--voice-prompt-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated"

    print("Step 1: Generating test set audio...")
    from infer import generate_test_set
    generate_test_set(args.checkpoint, args.data_dir, str(generated_dir))

    entries = prepare_dataset(args.data_dir)
    _, _, test_ids = get_splits(args.data_dir)
    test_entries = [e for e in entries if e["id"] in set(test_ids)]
    references = {f"{e['id']}.wav": e["text"] for e in test_entries}

    print("Step 2: Running evaluation...")
    results = run_full_evaluation(
        generated_dir=str(generated_dir),
        references=references,
        reference_audio_dir=args.reference_audio_dir,
        device=args.device,
        output_path=str(output_dir / "evaluation.json"),
    )

    print("\n" + "=" * 50)
    print("COSYVOICE 3 EVALUATION RESULTS")
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
