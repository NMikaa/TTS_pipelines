"""
CSM-1B inference for Georgian.

Usage:
    # Single utterance (uses merged model from HuggingFace)
    python infer.py --text "გამარჯობა, როგორ ხარ?"

    # With local checkpoint
    python infer.py --checkpoint checkpoints/checkpoint-2212 --text "გამარჯობა"

    # Custom speaker (0-11)
    python infer.py --text "გამარჯობა" --speaker-id 7

    # Batch evaluation from manifest
    python infer.py --eval-manifest ../../data/clean/eval_manifest.json --output-dir outputs/
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

HF_MODEL = "NMikka/CSM-1B-Georgian"
SAMPLE_RATE = 24000


def load_model(checkpoint_path=None, device="cuda"):
    """Load CSM-1B model (merged weights from HF or local checkpoint)."""
    model_id = checkpoint_path or HF_MODEL

    print(f"Loading model from {model_id}...")
    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    model = CsmForConditionalGeneration.from_pretrained(
        model_id, device_map=device,
    )
    model.eval()
    return model, processor


def generate(model, processor, text, speaker_id=7, device="cuda",
             max_new_tokens=125 * 10):
    """Generate audio from text."""
    inputs = processor(
        f"[{speaker_id}]{text}",
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        audio = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=max_new_tokens,
        )

    return audio[0].cpu().float().numpy()


def generate_batch(model, processor, texts, speaker_id=7, device="cuda",
                   max_new_tokens=125 * 15):
    """Generate audio for a batch of texts."""
    formatted = [f"[{speaker_id}]{t}" for t in texts]
    inputs = processor(
        formatted,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        audios = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=max_new_tokens,
        )

    return [a.cpu().float().numpy() for a in audios]


def main():
    parser = argparse.ArgumentParser(description="CSM-1B Georgian inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=f"Model checkpoint path (default: {HF_MODEL})")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--speaker-id", type=int, default=7,
                        help="Speaker ID 0-11 (default: 7, best CER)")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint, args.device)

    if args.text:
        audio_np = generate(model, processor, args.text,
                            speaker_id=args.speaker_id, device=args.device)
        sf.write(args.output, audio_np, SAMPLE_RATE)
        print(f"Saved: {args.output} ({len(audio_np)/SAMPLE_RATE:.1f}s)")

    elif args.eval_manifest:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        with open(args.eval_manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        print(f"Generating {len(entries)} samples (speaker {args.speaker_id})...")
        for batch_start in range(0, len(entries), args.batch_size):
            batch = entries[batch_start:batch_start + args.batch_size]
            texts = [e["text"] for e in batch]

            audio_list = generate_batch(model, processor, texts,
                                        speaker_id=args.speaker_id,
                                        device=args.device)

            for e, audio_np in zip(batch, audio_list):
                out_path = out_dir / f"{e['id']}.wav"
                sf.write(str(out_path), audio_np, SAMPLE_RATE)

            done = min(batch_start + args.batch_size, len(entries))
            print(f"  [{done}/{len(entries)}] done")

        print(f"Done. Generated {len(entries)} samples in {out_dir}")
    else:
        parser.error("Provide either --text or --eval-manifest")


if __name__ == "__main__":
    main()
