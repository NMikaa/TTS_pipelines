"""
MagPIE TTS fine-tuning for Georgian.

Four-step process:
  1. Resample audio from 24kHz to 22,050 Hz (NanoCodec requirement)
  2. Pre-compute NanoCodec tokens for all audio files
  3. Convert JSONL manifests to NeMo format with codec paths
  4. Fine-tune MagPIE TTS using NeMo's own training script (examples/tts/magpietts.py)

Usage:
    # Default (recommended for A6000 48GB)
    python train.py --data-dir ../../data/clean

    # Custom settings
    python train.py --data-dir ../../data/clean --lr 1e-5 --batch-size 8 --epochs 150

    # Prepare data only (resample + codec tokens + manifests)
    python train.py --data-dir ../../data/clean --prepare-only

    # Resume training (automatic — just re-run same command)
    python train.py --data-dir ../../data/clean
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# Add repo root to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MagPIEConfig

# Path to NeMo repo (cloned alongside pipelines)
NEMO_ROOT = Path(__file__).resolve().parents[2] / "NeMo"
NEMO_TRAIN_SCRIPT = NEMO_ROOT / "examples" / "tts" / "magpietts.py"
NEMO_CONFIG_DIR = NEMO_ROOT / "examples" / "tts" / "conf" / "magpietts"


def resample_audio(config: MagPIEConfig):
    """Step 1: Resample all audio from 24kHz to 22,050 Hz."""
    data_dir = Path(config.data_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]

    resampled_dir = data_dir / "audio_22khz"
    resampled_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio paths from both manifests
    audio_paths = set()
    for manifest_name in [config.train_manifest, config.eval_manifest, config.speaker_refs_manifest]:
        manifest_path = data_dir / manifest_name
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_paths.add(entry["audio_path"])

    print(f"Step 1: Resampling {len(audio_paths)} audio files to {config.sample_rate} Hz...")

    resampler = torchaudio.transforms.Resample(
        orig_freq=config.source_sample_rate,
        new_freq=config.sample_rate,
    )

    skipped = 0
    processed = 0
    sorted_paths = sorted(audio_paths)
    for rel_path in tqdm(sorted_paths, desc="  Resampling", unit="file"):
        src_path = repo_root / rel_path
        out_name = Path(rel_path).name
        out_path = resampled_dir / out_name

        if out_path.exists():
            skipped += 1
            continue

        if not src_path.exists():
            continue

        waveform, sr = torchaudio.load(str(src_path))
        if sr != config.source_sample_rate:
            resampler_alt = torchaudio.transforms.Resample(sr, config.sample_rate)
            waveform = resampler_alt(waveform)
        else:
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        torchaudio.save(str(out_path), waveform, config.sample_rate)
        processed += 1

    print(f"  Done: {processed} resampled, {skipped} already existed")
    return resampled_dir


def precompute_codec_tokens(config: MagPIEConfig, resampled_dir: Path):
    """Step 2: Pre-compute NanoCodec tokens for all resampled audio files."""
    from nemo.collections.tts.models import AudioCodecModel

    data_dir = Path(config.data_dir).resolve()
    codes_dir = data_dir / "codec_codes"
    codes_dir.mkdir(parents=True, exist_ok=True)

    # Load NanoCodec
    print(f"Step 2: Loading NanoCodec from {config.codec_model}...")
    codec = AudioCodecModel.from_pretrained(config.codec_model).eval().cuda()

    # Collect all audio filenames from manifests
    audio_filenames = set()
    for manifest_name in [config.train_manifest, config.eval_manifest, config.speaker_refs_manifest]:
        manifest_path = data_dir / manifest_name
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_filenames.add(Path(entry["audio_path"]).name)

    print(f"  Pre-computing codec tokens for {len(audio_filenames)} files...")

    skipped = 0
    processed = 0
    sorted_filenames = sorted(audio_filenames)
    for fname in tqdm(sorted_filenames, desc="  Encoding", unit="file"):
        stem = Path(fname).stem
        codes_path = codes_dir / f"{stem}.pt"

        if codes_path.exists():
            skipped += 1
            continue

        wav_path = resampled_dir / fname
        if not wav_path.exists():
            continue

        waveform, sr = torchaudio.load(str(wav_path))
        assert sr == config.sample_rate, f"Expected {config.sample_rate}Hz, got {sr}Hz"

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Squeeze channel dim: (1, time) -> (time,), then unsqueeze for batch: (1, time)
        # Codec expects (batch, time), NOT (batch, channels, time)
        waveform = waveform.squeeze(0).unsqueeze(0).cuda()
        wav_len = torch.tensor([waveform.shape[-1]], device="cuda")

        with torch.no_grad():
            codes, codes_len = codec.encode(audio=waveform, audio_len=wav_len)
            codes = codes.squeeze(0).cpu()  # [num_codebooks, num_frames]

        torch.save(codes, str(codes_path))
        processed += 1

    print(f"  Done: {processed} encoded, {skipped} already existed")

    # Free GPU memory
    del codec
    torch.cuda.empty_cache()

    return codes_dir


def convert_manifest_to_nemo(
    config: MagPIEConfig,
    manifest_name: str,
    resampled_dir: Path,
    codes_dir: Path,
    output_name: str,
) -> str:
    """Convert our JSONL manifest to NeMo MagPIE format."""
    data_dir = Path(config.data_dir).resolve()
    manifest_path = data_dir / manifest_name
    output_path = data_dir / output_name

    if output_path.exists():
        count = sum(1 for _ in open(output_path))
        print(f"  NeMo manifest already exists: {output_path} ({count} entries)")
        return str(output_path)

    # Load speaker references for context audio
    speaker_refs = {}
    refs_manifest = data_dir / config.speaker_refs_manifest
    if refs_manifest.exists():
        with open(refs_manifest) as f:
            for line in f:
                entry = json.loads(line.strip())
                spk = entry.get("speaker_id", "0")
                if spk not in speaker_refs:
                    speaker_refs[spk] = entry

    count = 0
    with open(manifest_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            entry = json.loads(line.strip())
            audio_name = Path(entry["audio_path"]).name
            stem = Path(audio_name).stem
            speaker_id = entry.get("speaker_id", "0")

            resampled_path = resampled_dir / audio_name
            codes_path = codes_dir / f"{stem}.pt"

            if not resampled_path.exists():
                continue

            nemo_entry = {
                "audio_filepath": str(resampled_path),
                "text": entry["text"],
                "duration": entry.get("duration", 0.0),
                "speaker": int(speaker_id) if speaker_id.isdigit() else 0,
            }

            if codes_path.exists():
                nemo_entry["target_audio_codes_path"] = str(codes_path)

            # Add context audio (for voice cloning conditioning)
            if speaker_id in speaker_refs:
                ref = speaker_refs[speaker_id]
                ref_name = Path(ref["audio_path"]).name
                ref_resampled = resampled_dir / ref_name
                ref_stem = Path(ref_name).stem
                ref_codes = codes_dir / f"{ref_stem}.pt"

                if ref_resampled.exists():
                    nemo_entry["context_audio_filepath"] = str(ref_resampled)
                    nemo_entry["context_audio_duration"] = ref.get("duration", 5.0)
                    if ref_codes.exists():
                        nemo_entry["context_audio_codes_path"] = str(ref_codes)

            f_out.write(json.dumps(nemo_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Wrote {count} entries to {output_path}")
    return str(output_path)


def prepare_data(config: MagPIEConfig):
    """Steps 1-3: Resample, compute codec tokens, convert manifests."""
    data_dir = Path(config.data_dir).resolve()

    for manifest_name in [config.train_manifest, config.eval_manifest]:
        if not (data_dir / manifest_name).exists():
            raise FileNotFoundError(f"Manifest not found: {data_dir / manifest_name}")

    # Step 1: Resample audio
    resampled_dir = resample_audio(config)

    # Step 2: Pre-compute codec tokens
    codes_dir = precompute_codec_tokens(config, resampled_dir)

    # Step 3: Convert manifests
    print("Step 3: Converting manifests to NeMo format...")
    train_manifest = convert_manifest_to_nemo(
        config, config.train_manifest, resampled_dir, codes_dir, "train_manifest_nemo.json"
    )
    eval_manifest = convert_manifest_to_nemo(
        config, config.eval_manifest, resampled_dir, codes_dir, "eval_manifest_nemo.json"
    )

    return train_manifest, eval_manifest


def train(config: MagPIEConfig, train_manifest: str, eval_manifest: str):
    """Step 4: Fine-tune MagPIE TTS using NeMo's training script with Hydra overrides."""
    if not NEMO_TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"NeMo training script not found at {NEMO_TRAIN_SCRIPT}. "
            f"Clone the NeMo repo: git clone --depth 1 https://github.com/NVIDIA/NeMo.git"
        )

    data_dir = Path(config.data_dir).resolve()
    resampled_dir = str(data_dir / "audio_22khz")
    codes_dir = str(data_dir / "codec_codes")

    # Build Hydra overrides for the NeMo training script
    # The base config is magpietts.yaml — we override what we need for fine-tuning
    overrides = [
        # Use Georgian config (includes all tokenizers from pretrained multilingual model)
        "--config-name=magpietts_georgian",
        # Epochs and batch size
        f"max_epochs={config.max_epochs}",
        f"batch_size={config.batch_size}",
        # Dataset meta — train
        f"+train_ds_meta.georgian_train.manifest_path={train_manifest}",
        f"+train_ds_meta.georgian_train.audio_dir={resampled_dir}",
        f"+train_ds_meta.georgian_train.feature_dir={codes_dir}",
        f"+train_ds_meta.georgian_train.sample_weight=1.0",
        # Use ByT5 tokenizer for Georgian (byte-level, language-agnostic)
        # Without this, NeMo defaults to english_phoneme (IPA) which fails on Georgian
        "+train_ds_meta.georgian_train.tokenizer_names=[text_ce_tokenizer]",
        # Dataset meta — val
        f"+val_ds_meta.georgian_eval.manifest_path={eval_manifest}",
        f"+val_ds_meta.georgian_eval.audio_dir={resampled_dir}",
        f"+val_ds_meta.georgian_eval.feature_dir={codes_dir}",
        f"+val_ds_meta.georgian_eval.sample_weight=1.0",
        # Use ByT5 tokenizer for Georgian eval too
        "+val_ds_meta.georgian_eval.tokenizer_names=[text_ce_tokenizer]",
        # Codec model
        f"model.codecmodel_path={config.codec_model}",
        # Optimizer — lower LR for fine-tuning
        f"model.optim.lr={config.learning_rate}",
        # Trainer
        f"trainer.devices={config.devices}",
        f"trainer.precision={config.precision}",
        f"trainer.gradient_clip_val={config.grad_clip_val}",
        f"trainer.log_every_n_steps=10",
        f"trainer.check_val_every_n_epoch=1",
        # Use single GPU strategy
        "trainer.strategy=auto" if config.devices == 1 else "trainer.strategy=ddp_find_unused_parameters_true",
        # Experiment manager
        f"exp_manager.exp_dir={config.exp_dir}",
        f"exp_manager.name={config.exp_name}",
        f"exp_manager.resume_if_exists={str(config.resume_if_exists).lower()}",
        "exp_manager.resume_ignore_no_checkpoint=true",
        # Initialize from pretrained model
        f"+init_from_pretrained_model={config.pretrained_model}",
    ]

    # W&B logging
    if config.wandb_project:
        overrides.extend([
            "exp_manager.create_wandb_logger=true",
            f"exp_manager.wandb_logger_kwargs.project={config.wandb_project}",
            f"exp_manager.wandb_logger_kwargs.name={config.wandb_run_name}",
        ])

    cmd = [sys.executable, str(NEMO_TRAIN_SCRIPT)] + overrides

    print(f"\nStarting MagPIE TTS fine-tuning:")
    print(f"  Pretrained model: {config.pretrained_model}")
    print(f"  Codec: {config.codec_model}")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Eval manifest: {eval_manifest}")
    print(f"  Epochs: {config.max_epochs}")
    print(f"  LR: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Precision: {config.precision}")
    print(f"  Grad clip: {config.grad_clip_val}")
    print(f"  NeMo script: {NEMO_TRAIN_SCRIPT}")
    print(f"\n  Command:")
    print(f"  {' '.join(cmd[:3])} \\")
    for ov in cmd[3:]:
        print(f"    {ov} \\")
    print()

    # Run NeMo training script
    result = subprocess.run(cmd, cwd=str(NEMO_ROOT))
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining complete. Checkpoints saved to {config.exp_dir}/{config.exp_name}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MagPIE TTS on Georgian")
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["32", "bf16-mixed", "16-mixed"])
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, don't train")
    args = parser.parse_args()

    config = MagPIEConfig(data_dir=args.data_dir)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.max_epochs = args.epochs
    if args.devices:
        config.devices = args.devices
    if args.precision:
        config.precision = args.precision
    if args.no_wandb:
        config.wandb_project = None

    # Steps 1-3: Prepare data
    train_manifest, eval_manifest = prepare_data(config)

    if args.prepare_only:
        print("\nData preparation complete. Exiting (--prepare-only).")
        return

    # Step 4: Train using NeMo's script directly
    train(config, train_manifest, eval_manifest)


if __name__ == "__main__":
    main()
