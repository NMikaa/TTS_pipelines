"""
CSM-1B inference for Georgian.

Usage:
    python infer.py --checkpoint checkpoints/final --text "გამარჯობა"
    python infer.py --checkpoint checkpoints/final --text "გამარჯობა" --output output.wav
    python infer.py --checkpoint checkpoints/final --eval-manifest ../../data/clean/eval_manifest.json --output-dir outputs/
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel


def load_model(checkpoint_path, device="cuda"):
    """Load CSM-1B with optional LoRA adapter."""
    checkpoint = Path(checkpoint_path)

    # Check if this is a LoRA adapter (has adapter_config.json)
    is_lora = (checkpoint / "adapter_config.json").exists()

    if is_lora:
        print(f"Loading base model + LoRA adapter from {checkpoint}")
        processor = AutoProcessor.from_pretrained(checkpoint)
        base_model = CsmForConditionalGeneration.from_pretrained(
            "sesame/csm-1b", torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
        model = model.merge_and_unload()
    else:
        print(f"Loading model from {checkpoint}")
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = CsmForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
        )

    model = model.to(device).eval()
    return model, processor


def generate(model, processor, text, speaker_id="1", device="cuda",
             max_new_tokens=750, temperature=0.9, depth_temperature=0.5):
    """Generate audio from text."""
    conversation = [
        {
            "role": speaker_id,
            "content": [{"type": "text", "text": text}],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        audio = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            depth_decoder_temperature=depth_temperature,
        )

    return audio[0].cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="CSM-1B inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--speaker-id", type=str, default="1")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--eval-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint, args.device)

    if args.text:
        audio_np = generate(model, processor, args.text,
                            speaker_id=args.speaker_id, device=args.device,
                            temperature=args.temperature)
        sf.write(args.output, audio_np, 24000)
        print(f"Saved: {args.output} ({len(audio_np)/24000:.1f}s)")

    elif args.eval_manifest:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        with open(args.eval_manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        print(f"Generating {len(entries)} samples...")
        for i, entry in enumerate(entries):
            audio_np = generate(model, processor, entry["text"],
                                speaker_id=str(entry.get("speaker_id", "1")),
                                device=args.device, temperature=args.temperature)
            out_path = out_dir / f"{entry['id']}.wav"
            sf.write(str(out_path), audio_np, 24000)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(entries)} done")

        print(f"Done. Generated {len(entries)} samples in {out_dir}")
    else:
        parser.error("Provide either --text or --eval-manifest")


if __name__ == "__main__":
    main()
