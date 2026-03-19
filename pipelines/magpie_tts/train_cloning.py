"""
Voice cloning training for MagPIE TTS.

Trains the context encoder to learn zero-shot voice cloning from audio
references, using the Georgian Common Voice dataset (12 speakers).

Strategy:
  - Start from pretrained NVIDIA multilingual model + Georgian fine-tuned weights
  - Strip baked speaker embeddings so the context encoder path activates
  - Create context-paired training data: each sample gets a different clip
    from the same speaker as reference audio
  - Train with speaker holdout: N speakers for training, 2 held out for
    zero-shot evaluation

Why this might work:
  - Context encoder is a small 1-layer transformer (768d, 12 heads)
  - The decoder already understands Georgian from prior fine-tuning
  - We're teaching the context encoder to map reference audio → speaker
    conditioning, while the rest of the model adapts its cross-attention
  - Context encoder starts from random init (pretrained .nemo excludes it
    when baked embeddings are present) but 12 speakers should give enough
    signal for a 1-layer module

Prerequisites:
  - Data downloaded: python -m shared.data.download --output-dir ./data
  - Base training data prepared: python train.py --data-dir ../../data/clean --prepare-only
  - (Optional) Georgian checkpoint from prior fine-tuning

Usage:
    # List available speakers and their stats
    python train_cloning.py --data-dir ../../data/clean --list-speakers

    # Full pipeline (pretrained + Georgian checkpoint)
    python train_cloning.py --data-dir ../../data/clean \\
        --georgian-ckpt /path/to/georgian.ckpt \\
        --holdout-speakers 1 5

    # Without Georgian checkpoint (learn Georgian + cloning together)
    python train_cloning.py --data-dir ../../data/clean --holdout-speakers 1 5

    # Prepare model + data only, don't train
    python train_cloning.py --data-dir ../../data/clean --prepare-only

    # Resume training (automatic via NeMo exp_manager)
    python train_cloning.py --data-dir ../../data/clean
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Prevent OpenBLAS from spawning 36 threads per dataloader worker — causes
# "Resource temporarily unavailable" when workers * threads > system limit.
for _env_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MagPIEConfig

NEMO_ROOT = Path(__file__).resolve().parents[2] / "NeMo"
NEMO_TRAIN_SCRIPT = NEMO_ROOT / "examples" / "tts" / "magpietts.py"


# ---------------------------------------------------------------------------
# Step 1: Model preparation — strip baked embeddings
# ---------------------------------------------------------------------------

def prepare_cloning_checkpoint(
    pretrained_model: str,
    georgian_ckpt: Optional[str],
    output_path: str,
) -> str:
    """Create a checkpoint without baked embeddings for voice cloning training.

    Loads the pretrained multilingual model, optionally overlays Georgian
    fine-tuned weights (excluding baked embeddings), strips the baked speaker
    embeddings, and saves a .ckpt with just the state_dict.

    The resulting checkpoint has:
      - Georgian-adapted encoder/decoder (if georgian_ckpt provided)
      - Random-initialized context encoder (will be trained)
      - No baked embeddings (forces context encoder path)

    Args:
        pretrained_model: HuggingFace model ID or path to .nemo file.
        georgian_ckpt: Path to Georgian fine-tuning .ckpt (optional).
        output_path: Where to save the cloning-ready checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    output_path = str(Path(output_path).resolve())
    if Path(output_path).exists():
        print(f"  Cloning checkpoint already exists: {output_path}")
        return output_path

    from nemo.collections.tts.models import MagpieTTSModel

    print(f"  Loading pretrained model: {pretrained_model}")
    model = MagpieTTSModel.from_pretrained(pretrained_model, map_location="cpu")

    if georgian_ckpt:
        print(f"  Overlaying Georgian weights from: {georgian_ckpt}")
        ckpt = torch.load(str(georgian_ckpt), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # Keep Georgian encoder/decoder weights, skip baked embeddings and
        # context_encoder (we want the random init, since Georgian ckpt
        # wouldn't have trained context_encoder weights anyway)
        filtered = {
            k: v for k, v in state.items()
            if "baked" not in k and "context_encoder" not in k
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        skipped = len(state) - len(filtered)
        print(f"  Loaded {len(filtered)} Georgian params, skipped {skipped} baked/context keys")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}...")

    # Strip baked speaker embeddings so has_baked_context_embedding → False
    had_baked = model.has_baked_context_embedding
    model.baked_context_embedding = None
    model._baked_embedding_T = None
    model._baked_embedding_D = None
    model.baked_context_embedding_len = None
    print(f"  Stripped baked embeddings (was present: {had_baked})")
    print(f"  has_baked_context_embedding is now: {model.has_baked_context_embedding}")

    # Save as PyTorch checkpoint (simpler than .nemo, avoids tokenizer/config issues)
    # NeMo loads this via init_from_ptl_ckpt
    state_dict = model.state_dict()  # Custom state_dict excludes codec/SV models
    torch.save({"state_dict": state_dict}, output_path)

    n_ctx = sum(1 for k in state_dict if "context_encoder" in k)
    print(f"  Saved cloning checkpoint: {output_path}")
    print(f"  Total params: {len(state_dict)}, context_encoder params: {n_ctx}")

    del model
    torch.cuda.empty_cache()
    return output_path


# ---------------------------------------------------------------------------
# Step 2: Data preparation — context-paired manifests
# ---------------------------------------------------------------------------

def load_all_samples(data_dir: Path) -> Dict[str, list]:
    """Load all samples from train + eval manifests, grouped by speaker_id."""
    samples_by_speaker: Dict[str, list] = defaultdict(list)

    for manifest_name in ["train_manifest.json", "eval_manifest.json"]:
        manifest_path = data_dir / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run: python -m shared.data.download --output-dir ./data"
            )
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                spk = entry.get("speaker_id", "0")
                samples_by_speaker[spk].append(entry)

    return dict(samples_by_speaker)


def list_speakers(data_dir: str):
    """Print speaker statistics from the dataset."""
    samples_by_speaker = load_all_samples(Path(data_dir).resolve())

    print(f"\n{'Speaker':>8} {'Samples':>8} {'Hours':>8} {'Avg MOS':>8} {'Avg CER':>8}")
    print("-" * 48)

    total_samples = 0
    total_hours = 0
    for spk in sorted(samples_by_speaker.keys(), key=lambda s: int(s) if s.isdigit() else s):
        samples = samples_by_speaker[spk]
        n = len(samples)
        hours = sum(s.get("duration", 0) for s in samples) / 3600
        avg_mos = sum(s.get("nisqa_mos", 0) for s in samples) / n if n else 0
        avg_cer = sum(s.get("asr_cer", 0) for s in samples) / n if n else 0
        print(f"{spk:>8} {n:>8} {hours:>8.2f} {avg_mos:>8.2f} {avg_cer:>8.3f}")
        total_samples += n
        total_hours += hours

    print("-" * 48)
    print(f"{'Total':>8} {total_samples:>8} {total_hours:>8.2f}")
    print(f"\n{len(samples_by_speaker)} speakers available for holdout selection.\n")


def create_cloning_manifests(
    data_dir: str,
    holdout_speakers: List[str],
    resampled_dir: Path,
    codes_dir: Path,
    seed: int = 42,
) -> Tuple[str, str]:
    """Create NeMo manifests for voice cloning training.

    For each sample, assigns a randomly selected DIFFERENT clip from the same
    speaker as context audio. Splits data into training (non-holdout speakers)
    and evaluation (holdout speakers + 5% of training speakers for loss
    monitoring).

    Args:
        data_dir: Path to data/clean directory.
        holdout_speakers: Speaker IDs to hold out for zero-shot evaluation.
        resampled_dir: Path to 22kHz resampled audio.
        codes_dir: Path to pre-computed codec codes.
        seed: Random seed for reproducible context assignment.

    Returns:
        (train_manifest_path, eval_manifest_path)
    """
    data_dir = Path(data_dir).resolve()
    rng = random.Random(seed)

    samples_by_speaker = load_all_samples(data_dir)
    holdout_set = set(str(s) for s in holdout_speakers)

    # Verify holdout speakers exist
    for spk in holdout_set:
        if spk not in samples_by_speaker:
            available = sorted(samples_by_speaker.keys())
            raise ValueError(
                f"Holdout speaker '{spk}' not found. Available: {available}"
            )

    train_entries = []
    holdout_entries = []

    for speaker, samples in samples_by_speaker.items():
        is_holdout = speaker in holdout_set

        if len(samples) < 2:
            print(f"  Warning: speaker {speaker} has only {len(samples)} sample(s), skipping")
            continue

        for i, entry in enumerate(samples):
            # Pick a random DIFFERENT clip from the same speaker as context
            ctx_idx = rng.randint(0, len(samples) - 2)
            if ctx_idx >= i:
                ctx_idx += 1
            context_entry = samples[ctx_idx]

            # Build NeMo manifest entry
            audio_name = Path(entry["audio_path"]).name
            stem = Path(audio_name).stem
            ctx_name = Path(context_entry["audio_path"]).name
            ctx_stem = Path(ctx_name).stem

            resampled_path = resampled_dir / audio_name
            codes_path = codes_dir / f"{stem}.pt"
            ctx_resampled = resampled_dir / ctx_name
            ctx_codes = codes_dir / f"{ctx_stem}.pt"

            # Skip if files don't exist
            if not resampled_path.exists():
                continue

            nemo_entry = {
                "audio_filepath": str(resampled_path),
                "text": entry["text"],
                "duration": entry.get("duration", 0.0),
                "speaker": int(speaker) if speaker.isdigit() else 0,
            }
            if codes_path.exists():
                nemo_entry["target_audio_codes_path"] = str(codes_path)
            if ctx_resampled.exists():
                nemo_entry["context_audio_filepath"] = str(ctx_resampled)
                nemo_entry["context_audio_duration"] = context_entry.get("duration", 5.0)
            if ctx_codes.exists():
                nemo_entry["context_audio_codes_path"] = str(ctx_codes)
            elif ctx_resampled.exists():
                print(f"  Warning: codec codes missing for context {ctx_stem}, will encode on-the-fly")

            if is_holdout:
                holdout_entries.append(nemo_entry)
            else:
                train_entries.append(nemo_entry)

    # Split 5% of training data for validation loss monitoring
    rng.shuffle(train_entries)
    split_idx = max(1, len(train_entries) // 20)
    eval_from_train = train_entries[:split_idx]
    train_entries = train_entries[split_idx:]

    # Eval = holdout speakers (zero-shot) + 5% training speakers (loss monitoring)
    eval_entries = holdout_entries + eval_from_train

    # Write manifests
    train_path = data_dir / "cloning_train_manifest_nemo.json"
    eval_path = data_dir / "cloning_eval_manifest_nemo.json"

    for path, entries, label in [
        (train_path, train_entries, "train"),
        (eval_path, eval_entries, "eval"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {label}: {len(entries)} entries → {path}")

    # Report split stats
    holdout_str = ", ".join(sorted(holdout_set))
    print(f"\n  Holdout speakers: [{holdout_str}]")
    print(f"  Training speakers: {len(samples_by_speaker) - len(holdout_set)}")
    print(f"  Holdout eval samples: {len(holdout_entries)}")
    print(f"  Train-split eval samples: {len(eval_from_train)}")

    return str(train_path), str(eval_path)


# ---------------------------------------------------------------------------
# Step 3: Training — launch NeMo
# ---------------------------------------------------------------------------

def train_cloning(
    cloning_ckpt: str,
    train_manifest: str,
    eval_manifest: str,
    config: MagPIEConfig,
    learning_rate: float = 5e-5,
    max_epochs: int = 150,
    batch_size: int = 32,
    context_duration_min: float = 3.0,
    context_duration_max: float = 8.0,
):
    """Launch NeMo training for voice cloning.

    Uses the cloning-ready checkpoint (no baked embeddings) and context-paired
    manifests. The context encoder learns to map reference audio to speaker
    conditioning while the decoder adapts its cross-attention.
    """
    if not NEMO_TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"NeMo training script not found at {NEMO_TRAIN_SCRIPT}. "
            f"Clone NeMo: git clone --depth 1 https://github.com/NVIDIA/NeMo.git"
        )

    data_dir = Path(config.data_dir).resolve()
    resampled_dir = str(data_dir / "audio_22khz")
    codes_dir = str(data_dir / "codec_codes")

    overrides = [
        # Georgian config (tokenizers match pretrained model)
        "--config-name=magpietts_georgian",
        # Epochs and batch
        f"max_epochs={max_epochs}",
        f"batch_size={batch_size}",
        # Training data
        f"+train_ds_meta.cloning_train.manifest_path={train_manifest}",
        f"+train_ds_meta.cloning_train.audio_dir={resampled_dir}",
        f"+train_ds_meta.cloning_train.feature_dir={codes_dir}",
        "+train_ds_meta.cloning_train.sample_weight=1.0",
        "+train_ds_meta.cloning_train.tokenizer_names=[text_ce_tokenizer]",
        # Validation data
        f"+val_ds_meta.cloning_eval.manifest_path={eval_manifest}",
        f"+val_ds_meta.cloning_eval.audio_dir={resampled_dir}",
        f"+val_ds_meta.cloning_eval.feature_dir={codes_dir}",
        "+val_ds_meta.cloning_eval.sample_weight=1.0",
        "+val_ds_meta.cloning_eval.tokenizer_names=[text_ce_tokenizer]",
        # Context encoder settings
        f"model.context_duration_min={context_duration_min}",
        f"model.context_duration_max={context_duration_max}",
        "model.load_cached_codes_if_available=true",
        # Codec
        f"model.codecmodel_path={config.codec_model}",
        # Optimizer
        f"model.optim.lr={learning_rate}",
        # Trainer
        f"trainer.devices={config.devices}",
        f"trainer.precision={config.precision}",
        f"trainer.gradient_clip_val={config.grad_clip_val}",
        "trainer.log_every_n_steps=10",
        "trainer.check_val_every_n_epoch=1",
        # Reduce dataloader workers to avoid OpenBLAS thread exhaustion
        "model.train_ds.dataloader_params.num_workers=2",
        "model.validation_ds.dataloader_params.num_workers=1",
        "trainer.strategy=auto" if config.devices == 1
        else "trainer.strategy=ddp_find_unused_parameters_true",
        # Experiment manager
        f"exp_manager.exp_dir={config.exp_dir}",
        "exp_manager.name=magpie_tts_cloning",
        f"exp_manager.resume_if_exists={str(config.resume_if_exists).lower()}",
        "exp_manager.resume_ignore_no_checkpoint=true",
        # Initialize from cloning-ready checkpoint (string form — simpler, matches NeMo docs)
        f"+init_from_ptl_ckpt={cloning_ckpt}",
        # Freeze everything except context_encoder + decoder cross-attention
        # to preserve Georgian quality while training voice cloning
        "+freeze_for_cloning=true",
    ]

    # W&B logging
    if config.wandb_project:
        overrides.extend([
            "exp_manager.create_wandb_logger=true",
            f"exp_manager.wandb_logger_kwargs.project={config.wandb_project}",
            "exp_manager.wandb_logger_kwargs.name=magpie-tts-cloning",
        ])

    cmd = [sys.executable, str(NEMO_TRAIN_SCRIPT)] + overrides

    print(f"\nStarting voice cloning training:")
    print(f"  Cloning checkpoint: {cloning_ckpt}")
    print(f"  Train manifest: {train_manifest}")
    print(f"  Eval manifest: {eval_manifest}")
    print(f"  Epochs: {max_epochs}")
    print(f"  LR: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Context duration: {context_duration_min}-{context_duration_max}s")
    print(f"  Precision: {config.precision}")
    print(f"\n  Command:")
    print(f"  {' '.join(cmd[:3])} \\")
    for ov in cmd[3:]:
        print(f"    {ov} \\")
    print()

    result = subprocess.run(cmd, cwd=str(NEMO_ROOT))
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining complete. Checkpoints in {config.exp_dir}/magpie_tts_cloning")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voice cloning training for MagPIE TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See available speakers
  python train_cloning.py --data-dir ../../data/clean --list-speakers

  # Train with Georgian checkpoint, hold out speakers 1 and 5
  python train_cloning.py --data-dir ../../data/clean \\
      --georgian-ckpt /path/to/checkpoint.ckpt \\
      --holdout-speakers 1 5

  # Prepare everything but don't start training
  python train_cloning.py --data-dir ../../data/clean --prepare-only
        """,
    )
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument(
        "--georgian-ckpt", type=str, default=None,
        help="Path to Georgian fine-tuning .ckpt (optional, improves quality)",
    )
    parser.add_argument(
        "--pretrained-model", type=str, default="nvidia/magpie_tts_multilingual_357m",
        help="HuggingFace pretrained model ID",
    )
    parser.add_argument(
        "--holdout-speakers", type=str, nargs="+", default=["1", "5"],
        help="Speaker IDs to hold out for zero-shot evaluation (default: 1 5)",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=150, help="Max epochs (default: 150)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--context-min", type=float, default=3.0, help="Min context duration (s)")
    parser.add_argument("--context-max", type=float, default=8.0, help="Max context duration (s)")
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["32", "bf16-mixed", "16-mixed"])
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-speakers", action="store_true", help="Print speaker stats and exit")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare model + data only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()

    # --- List speakers ---
    if args.list_speakers:
        list_speakers(args.data_dir)
        return

    # --- Validate prerequisites ---
    resampled_dir = data_dir / "audio_22khz"
    codes_dir = data_dir / "codec_codes"

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        print("Run: python -m shared.data.download --output-dir ./data")
        sys.exit(1)

    if not resampled_dir.exists() or not codes_dir.exists():
        print(f"Error: resampled audio or codec codes not found in {data_dir}")
        print("Run: python train.py --data-dir ../../data/clean --prepare-only")
        sys.exit(1)

    config = MagPIEConfig(data_dir=args.data_dir)
    if args.devices:
        config.devices = args.devices
    if args.precision:
        config.precision = args.precision
    if args.no_wandb:
        config.wandb_project = None

    # --- Step 1: Prepare cloning checkpoint ---
    cloning_ckpt_dir = data_dir / "cloning"
    cloning_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cloning_ckpt_path = str(cloning_ckpt_dir / "cloning_ready.ckpt")

    print("\nStep 1: Preparing cloning-ready checkpoint...")
    prepare_cloning_checkpoint(
        pretrained_model=args.pretrained_model,
        georgian_ckpt=args.georgian_ckpt,
        output_path=cloning_ckpt_path,
    )

    # --- Step 2: Create context-paired manifests ---
    print("\nStep 2: Creating context-paired manifests...")
    train_manifest, eval_manifest = create_cloning_manifests(
        data_dir=args.data_dir,
        holdout_speakers=args.holdout_speakers,
        resampled_dir=resampled_dir,
        codes_dir=codes_dir,
        seed=args.seed,
    )

    if args.prepare_only:
        print("\nPreparation complete (--prepare-only). Ready to train with:")
        print(f"  python train_cloning.py --data-dir {args.data_dir}")
        return

    # --- Step 3: Train ---
    print("\nStep 3: Launching voice cloning training...")
    train_cloning(
        cloning_ckpt=cloning_ckpt_path,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        config=config,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        context_duration_min=args.context_min,
        context_duration_max=args.context_max,
    )


if __name__ == "__main__":
    main()
