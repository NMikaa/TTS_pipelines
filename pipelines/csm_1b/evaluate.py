"""
Evaluate CSM-1B on the FLEURS Georgian test set.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --data-dir ./data --output-dir results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import get_splits
from shared.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSM-1B on Georgian")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (default: NMikka/CSM-1B-Georgian from HF)")
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--reference-audio-dir", type=str, default=None)
    parser.add_argument("--speaker-id", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated"

    _, _, test_entries = get_splits(args.data_dir)
    references = {f"{e['id']}.wav": e["text"] for e in test_entries}

    print("Step 1: Generating test set audio...")
    from infer import load_model, generate_batch
    import soundfile as sf
    model, processor = load_model(args.checkpoint, args.device)
    generated_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(test_entries), 8):
        batch = test_entries[i:i+8]
        batch_texts = [e["text"] for e in batch]
        audios = generate_batch(model, processor, batch_texts,
                                speaker_id=args.speaker_id, device=args.device)
        for e, audio in zip(batch, audios):
            sf.write(str(generated_dir / f"{e['id']}.wav"), audio, 24000)
        print(f"  [{min(i+8, len(test_entries))}/{len(test_entries)}] done")

    print("Step 2: Running evaluation...")
    results = run_full_evaluation(
        generated_dir=str(generated_dir),
        references=references,
        reference_audio_dir=args.reference_audio_dir,
        device=args.device,
        output_path=str(output_dir / "evaluation.json"),
    )

    print("\n" + "=" * 50)
    print("CSM-1B EVALUATION RESULTS")
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
