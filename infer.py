#!/usr/bin/env python3
"""
Inference script for fine-tuned Pocket TTS.

Based on the CALM paper (arxiv.org/abs/2509.06926):
- Temperature 0.7 (Gaussian noise scaled by sqrt(0.7))
- LSD decode steps = 1 (NFE=1)
- CFG alpha=1.5 is baked into the distilled student model
- No short-context transformer or noise augmentation for TTS

Usage:
    python infer.py --text "გამარჯობა, როგორ ხარ?" --checkpoint checkpoints/pocket_tts_georgian/epoch_2.pt
    python infer.py --text "Hello world" --voice alba --temp 0.7
    python infer.py --text "Test" --no-finetune
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent / "pocket-tts"))

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states


def load_finetuned_model(
    checkpoint_path: str,
    device: str = "cpu",
    use_ema: bool = True,
    temp: float = 0.7,
    lsd_steps: int = 1,
) -> TTSModel:
    """Load pretrained Pocket TTS and replace FlowLM weights with fine-tuned checkpoint."""
    print("Loading pretrained Pocket TTS model...")
    model = TTSModel.load_model(
        config="b6369a24",
        temp=temp,
        lsd_decode_steps=lsd_steps,
    )

    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load fine-tuned FlowLM weights (includes emb_mean, emb_std buffers)
    model.flow_lm.load_state_dict(ckpt["flow_lm_state_dict"], strict=True)
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, step {ckpt['global_step']}")

    # Optionally load EMA weights (often better quality, only available after ema_start_step)
    if use_ema and "ema_state_dict" in ckpt and ckpt["global_step"] >= 1000:
        print("  Applying EMA weights (step >= 1000)...")
        ema_sd = ckpt["ema_state_dict"]
        for name, param in model.flow_lm.named_parameters():
            if name in ema_sd:
                param.data.copy_(ema_sd[name])
        print(f"  Applied EMA for {len(ema_sd)} parameters")
    elif use_ema and ckpt["global_step"] < 1000:
        print(f"  EMA not applied (step {ckpt['global_step']} < 1000, EMA not yet active)")

    model.eval()
    return model


def generate(
    model: TTSModel,
    text: str,
    voice: str = "alba",
    output_path: str = "output.wav",
    device: str = "cpu",
):
    """Generate audio from text and save to WAV."""
    # Get voice state on CPU first (Mimi encoder lives on CPU)
    print(f"Getting voice state for '{voice}'...")
    model_state = model.get_state_for_audio_prompt(voice)

    # Move model and state to target device
    model = model.to(device)
    if device != "cpu":
        model_state = {
            k: {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            for k, v in model_state.items()
        }

    # Safe print for non-ASCII text on Windows
    try:
        print(f'Generating audio for: "{text}"')
    except UnicodeEncodeError:
        print(f"Generating audio for text ({len(text)} chars)")

    print(f"  Temperature: {model.temp}, LSD steps: {model.lsd_decode_steps}")
    audio = model.generate_audio(model_state, text)

    # audio shape: [samples] — add channel dim for torchaudio: [1, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    torchaudio.save(output_path, audio.cpu(), model.sample_rate)
    duration = audio.shape[-1] / model.sample_rate
    print(f"Saved {duration:.2f}s of audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pocket TTS inference with fine-tuned weights")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--checkpoint", default="checkpoints/pocket_tts_georgian/epoch_5.pt",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--voice", default="alba",
                        help="Voice name (alba, marius, javert, jean, fantine, cosette, eponine, azelma) or path to WAV")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-finetune", action="store_true", help="Use original pretrained weights only")
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature (paper default: 0.7)")
    parser.add_argument("--lsd-steps", type=int, default=1,
                        help="LSD decode steps / NFE (paper default: 1)")
    args = parser.parse_args()

    if args.no_finetune:
        print("Loading original pretrained model (no fine-tuning)...")
        model = TTSModel.load_model(
            config="b6369a24",
            temp=args.temp,
            lsd_decode_steps=args.lsd_steps,
        )
        model.eval()
    else:
        if not Path(args.checkpoint).exists():
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        model = load_finetuned_model(
            args.checkpoint,
            device=args.device,
            use_ema=not args.no_ema,
            temp=args.temp,
            lsd_steps=args.lsd_steps,
        )

    generate(model, args.text, voice=args.voice, output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
