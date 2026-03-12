"""
F5-TTS inference for Georgian.

Generates speech from Georgian text using a fine-tuned F5-TTS checkpoint
with extended vocab (pretrained + Georgian characters).

Usage:
    # Single utterance (uses best speaker ref by default)
    python infer.py --text "გამარჯობა, როგორ ხარ?"

    # With specific reference audio (for voice cloning)
    python infer.py --text "გამარჯობა" --ref-audio path/to/ref.wav --ref-text "რეფერენს ტექსტი"

    # Use a specific checkpoint
    python infer.py --text "გამარჯობა" --checkpoint ckpts/georgian_tts/model_30000.pt

    # Generate from FLEURS test set for evaluation
    python infer.py --eval --output-dir results/generated/ --checkpoint ckpts/georgian_tts/model_last.pt

    # Generate from eval manifest
    python infer.py --eval --manifest ../../data/clean/eval_manifest.json --output-dir results/generated/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Add repo root to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Defaults
DEFAULT_CHECKPOINT = "ckpts/georgian_tts/model_last.pt"
DEFAULT_VOCAB = "data/georgian_tts_char/vocab.txt"
DEFAULT_OUTPUT_DIR = "outputs"
SAMPLE_RATE = 24000


def load_georgian_model(checkpoint_path: str, vocab_path: str, device: str = "cuda"):
    """Load fine-tuned F5-TTS model with extended Georgian vocab."""
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import CFM, DiT

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ema_model = load_model(
        DiT,
        model_cfg,
        checkpoint_path,
        mel_spec_type="vocos",
        vocab_file=vocab_path,
        use_ema=False,
        device=device,
    )

    return ema_model, vocoder


def infer_single(
    model,
    vocoder,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    output_path: str,
    device: str = "cuda",
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
):
    """Generate a single utterance."""
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)

    audio_segment, final_sample_rate, _ = infer_process(
        ref_audio_processed,
        ref_text_processed,
        gen_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        speed=speed,
        device=device,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_segment, final_sample_rate)
    print(f"Saved: {output_path} ({len(audio_segment) / final_sample_rate:.2f}s)")
    return output_path


def get_default_ref(data_dir: str = "../../data/clean"):
    """Get the best speaker reference from speaker_refs_manifest.json."""
    manifest_path = Path(data_dir) / "speaker_refs_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Speaker refs manifest not found: {manifest_path}\n"
            "Provide --ref-audio and --ref-text explicitly."
        )

    repo_root = Path(__file__).resolve().parents[2]
    with open(manifest_path) as f:
        first_entry = json.loads(f.readline().strip())

    audio_path = Path(first_entry["audio_path"])
    if not audio_path.is_absolute():
        audio_path = repo_root / audio_path

    return str(audio_path), first_entry["text"]


def infer_eval(
    model,
    vocoder,
    manifest_path: str,
    output_dir: str,
    ref_audio: str,
    ref_text: str,
    device: str = "cuda",
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    max_samples: int = 0,
):
    """Generate audio for all entries in an evaluation manifest."""
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)

    with open(manifest_path) as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    if max_samples > 0:
        entries = entries[:max_samples]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, entry in enumerate(entries):
        gen_text = entry["text"]
        entry_id = entry.get("id", f"sample_{i:04d}")
        output_path = output_dir / f"{entry_id}.wav"

        if output_path.exists():
            print(f"[{i+1}/{len(entries)}] Skipping (exists): {entry_id}")
            results.append({"id": entry_id, "audio_path": str(output_path), "text": gen_text})
            continue

        try:
            audio_segment, final_sample_rate, _ = infer_process(
                ref_audio_processed,
                ref_text_processed,
                gen_text,
                model,
                vocoder,
                mel_spec_type="vocos",
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                speed=speed,
                device=device,
            )
            sf.write(str(output_path), audio_segment, final_sample_rate)
            results.append({"id": entry_id, "audio_path": str(output_path), "text": gen_text})
            print(f"[{i+1}/{len(entries)}] {entry_id}: {len(audio_segment)/final_sample_rate:.2f}s")
        except Exception as e:
            print(f"[{i+1}/{len(entries)}] FAILED {entry_id}: {e}")

    # Save generation manifest
    manifest_out = output_dir / "generated_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nGenerated {len(results)}/{len(entries)} samples -> {manifest_out}")
    return results


def main():
    parser = argparse.ArgumentParser(description="F5-TTS inference for Georgian")
    parser.add_argument("--text", type=str, help="Georgian text to synthesize")
    parser.add_argument("--ref-audio", type=str, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", type=str, help="Transcript of reference audio")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB, help="Vocab file path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-file", type=str, default=None, help="Output filename (single mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--nfe-step", type=int, default=32, help="Denoising steps (default: 32)")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="CFG strength (default: 2.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--data-dir", type=str, default="../../data/clean", help="Data dir for speaker refs")

    # Eval mode
    parser.add_argument("--eval", action="store_true", help="Generate from eval manifest")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest for eval mode")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples in eval mode (0=all)")

    args = parser.parse_args()

    if not args.text and not args.eval:
        parser.error("Provide --text for single inference or --eval for batch evaluation")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, vocoder = load_georgian_model(args.checkpoint, args.vocab, args.device)

    # Get reference audio
    if args.ref_audio and args.ref_text:
        ref_audio, ref_text = args.ref_audio, args.ref_text
    elif args.ref_audio:
        parser.error("--ref-audio requires --ref-text")
    else:
        ref_audio, ref_text = get_default_ref(args.data_dir)
        print(f"Using default ref: {Path(ref_audio).name}")

    if args.eval:
        # Batch evaluation mode
        manifest = args.manifest or str(Path(args.data_dir) / "eval_manifest.json")
        infer_eval(
            model, vocoder, manifest, args.output_dir,
            ref_audio, ref_text, args.device,
            args.nfe_step, args.cfg_strength, args.speed,
            args.max_samples,
        )
    else:
        # Single utterance mode
        output_file = args.output_file or "output.wav"
        output_path = str(Path(args.output_dir) / output_file)
        infer_single(
            model, vocoder, ref_audio, ref_text, args.text,
            output_path, args.device, args.nfe_step, args.cfg_strength, args.speed,
        )


if __name__ == "__main__":
    main()
