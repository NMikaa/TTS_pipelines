"""
MagPIE TTS inference for Georgian.

Supports single utterance generation and batch generation for the full FLEURS
test set (used by evaluate.py).

Usage:
    # Single utterance
    python infer.py --checkpoint exp/magpie_tts_georgian/checkpoints/last.ckpt \
        --text "გამარჯობა, როგორ ხარ?" --output output.wav

    # Single utterance with voice cloning (reference audio)
    python infer.py --checkpoint exp/magpie_tts_georgian/checkpoints/last.ckpt \
        --text "გამარჯობა" --reference-audio ref_speaker.wav --output output.wav

    # Batch generation on FLEURS test set
    python infer.py --checkpoint exp/magpie_tts_georgian/checkpoints/last.ckpt \
        --fleurs --output-dir outputs/fleurs/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MagPIEConfig


def load_model(checkpoint: str, config: MagPIEConfig):
    """Load a fine-tuned MagPIE TTS model from checkpoint."""
    from nemo.collections.tts.models import MagpieTTSModel

    if checkpoint.endswith(".nemo"):
        model = MagpieTTSModel.restore_from(checkpoint)
    elif checkpoint.endswith(".ckpt"):
        model = MagpieTTSModel.load_from_checkpoint(checkpoint)
    else:
        # Assume it's a pretrained model name (e.g. for zero-shot testing)
        model = MagpieTTSModel.from_pretrained(checkpoint)

    model = model.eval().cuda()
    return model


def generate_single(
    model,
    text: str,
    output_path: str,
    reference_audio: str = None,
    config: MagPIEConfig = None,
    cfg_scale: float = 2.5,
    temperature: float = 0.6,
    top_k: int = 80,
):
    """Generate a single utterance."""
    if config is None:
        config = MagPIEConfig()

    kwargs = {
        "transcript": text,
        "apply_TN": False,  # no text normalization for Georgian
    }

    # Voice cloning with reference audio
    if reference_audio:
        ref_wav, sr = torchaudio.load(reference_audio)
        if sr != config.sample_rate:
            ref_wav = torchaudio.transforms.Resample(sr, config.sample_rate)(ref_wav)
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)
        kwargs["context_audio"] = ref_wav.cuda()
    else:
        kwargs["speaker_index"] = 1  # default speaker (Sofia)

    with torch.no_grad():
        audio, audio_len = model.do_tts(**kwargs)

    # Save output
    audio_out = audio[0, :audio_len[0]].cpu().unsqueeze(0)
    torchaudio.save(output_path, audio_out, config.sample_rate)
    print(f"  Saved: {output_path} ({audio_out.shape[-1] / config.sample_rate:.2f}s)")


def generate_fleurs(model, output_dir: str, config: MagPIEConfig = None, reference_audio: str = None):
    """Generate audio for all FLEURS Georgian test samples."""
    from datasets import load_dataset

    if config is None:
        config = MagPIEConfig()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS Georgian test set...")
    ds = load_dataset("google/fleurs", "ka_ge", split="test")
    print(f"  {len(ds)} test samples")

    # Load reference audio for voice cloning if provided
    ref_wav = None
    if reference_audio:
        ref_wav, sr = torchaudio.load(reference_audio)
        if sr != config.sample_rate:
            ref_wav = torchaudio.transforms.Resample(sr, config.sample_rate)(ref_wav)
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(dim=0, keepdim=True)
        ref_wav = ref_wav.cuda()

    metadata = []
    for i, sample in enumerate(ds):
        sample_id = sample.get("id", i)
        text = sample["transcription"]
        out_path = out_dir / f"fleurs_{sample_id}.wav"

        if out_path.exists():
            metadata.append({"id": sample_id, "text": text, "audio_path": str(out_path)})
            continue

        kwargs = {
            "transcript": text,
            "apply_TN": False,
        }
        if ref_wav is not None:
            kwargs["context_audio"] = ref_wav
        else:
            kwargs["speaker_index"] = 1

        try:
            with torch.no_grad():
                audio, audio_len = model.do_tts(**kwargs)
            audio_out = audio[0, :audio_len[0]].cpu().unsqueeze(0)
            torchaudio.save(str(out_path), audio_out, config.sample_rate)
            metadata.append({"id": sample_id, "text": text, "audio_path": str(out_path)})
        except Exception as e:
            print(f"  Error on sample {sample_id}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(ds)}...")

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  Done: {len(metadata)} samples generated in {out_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="MagPIE TTS inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt, .nemo, or pretrained model name")
    parser.add_argument("--text", type=str, default=None, help="Single text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path for single generation")
    parser.add_argument("--reference-audio", type=str, default=None,
                        help="Reference audio for voice cloning")
    parser.add_argument("--fleurs", action="store_true", help="Generate full FLEURS test set")
    parser.add_argument("--output-dir", type=str, default="outputs/fleurs/",
                        help="Output dir for FLEURS generation")
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    args = parser.parse_args()

    config = MagPIEConfig(data_dir=args.data_dir)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)

    if args.text:
        generate_single(model, args.text, args.output, args.reference_audio, config)
    elif args.fleurs:
        generate_fleurs(model, args.output_dir, config, args.reference_audio)
    else:
        parser.error("Provide either --text or --fleurs")


if __name__ == "__main__":
    main()
