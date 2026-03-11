"""
Evaluate a fine-tuned MagPIE TTS model on the FLEURS Georgian test set.

Metrics:
- CER: round-trip intelligibility via Meta Omnilingual ASR
- WER: round-trip word error rate via Meta Omnilingual ASR

Usage:
    python evaluate.py
    python evaluate.py --checkpoint path/to/checkpoint.ckpt
    python evaluate.py --num-samples 100    # quick eval on subset
    python evaluate.py --speaker 1          # use baked speaker 1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NeMo"))

SAMPLE_RATE = 22050


def generate_fleurs(model, output_dir, speaker_idx=0, num_samples=None):
    """Generate audio for FLEURS Georgian test set.

    Args:
        model: Loaded MagpieTTSModel (from generate.load_model)
        output_dir: Directory to save generated .wav files
        speaker_idx: Baked speaker index (0-4)
        num_samples: If set, limit to first N samples

    Returns:
        List of dicts with 'audio_path', 'text', 'fleurs_id'
    """
    from generate import generate

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS Georgian test set...")
    ds = load_dataset("google/fleurs", "ka_ge", split="test", trust_remote_code=True)
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    print(f"  {len(ds)} samples to generate")

    metadata = []
    for i, sample in enumerate(ds):
        text = sample["transcription"]
        fleurs_id = sample.get("id", i)
        filename = f"fleurs_{i:04d}.wav"
        filepath = out / filename

        if filepath.exists():
            print(f"  [{i+1}/{len(ds)}] Skipping {filename} (already exists)")
        else:
            print(f"  [{i+1}/{len(ds)}] Generating: {text[:60]}...")
            audio, duration, rtf = generate(model, text, speaker_idx=speaker_idx)
            if audio is not None:
                torchaudio.save(str(filepath), audio, SAMPLE_RATE)
            else:
                print(f"    WARNING: No audio generated for sample {i}")
                continue

        metadata.append({
            "audio_path": str(filepath),
            "text": text,
            "fleurs_id": fleurs_id,
            "filename": filename,
        })

    # Save metadata
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Generated {len(metadata)} samples -> {out}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Evaluate MagPIE TTS on Georgian")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt (default: latest)")
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--speaker", type=int, default=0, help="Baked speaker index (0-4)")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit FLEURS samples (for quick testing)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, use existing audio")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated"

    # Step 1: Generate FLEURS test set audio
    if args.skip_generation:
        print("Step 1: Skipping generation (using existing audio)...")
        meta_path = generated_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {generated_dir}. Run without --skip-generation first.")
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        print("Step 1: Generating FLEURS test set audio...")
        from generate import load_model
        model = load_model(args.checkpoint)
        metadata = generate_fleurs(model, str(generated_dir), args.speaker, args.num_samples)
        del model
        torch.cuda.empty_cache()

    # Build references dict: filename -> ground truth text
    references = {entry["filename"]: entry["text"] for entry in metadata}

    # Step 2: Run evaluation
    print("\nStep 2: Running evaluation metrics...")
    from shared.evaluation import run_full_evaluation

    results = run_full_evaluation(
        generated_dir=str(generated_dir),
        references=references,
        device=args.device,
        output_path=str(output_dir / "evaluation.json"),
    )

    # Print results
    print("\n" + "=" * 50)
    print("MAGPIE TTS EVALUATION RESULTS")
    print("=" * 50)
    intel = results.get("intelligibility", {})
    if "mean_cer" in intel:
        print(f"  {'CER':20s} {intel['mean_cer']:.4f}")
    if "mean_wer" in intel:
        print(f"  {'WER':20s} {intel['mean_wer']:.4f}")

    print(f"\nDetailed results saved to {output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
