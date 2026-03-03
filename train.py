#!/usr/bin/env python3
"""
Train Pocket TTS on Georgian data.

Usage:
    # Step 1: Pre-compute latents (run once)
    python scripts/precompute_latents.py --manifest alignment/voice_actor_manifest.json

    # Step 2: Train
    python train.py --batch-size 16 --num-epochs 10 --lr 1e-4

    # Resume from checkpoint
    python train.py --resume checkpoints/pocket_tts_georgian/epoch_3.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "pocket-tts"))

from pocket_tts_training.config import TrainingConfig
from pocket_tts_training.trainer import PocketTTSTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Pocket TTS on Georgian data")

    # Data
    parser.add_argument("--latents-dir", default="latents_cache", help="Pre-computed latents directory")
    parser.add_argument("--manifest", default="alignment/voice_actor_manifest.json")
    parser.add_argument("--georgian-spm", default="", help="Path to Georgian SentencePiece model")

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Loss
    parser.add_argument("--fm-ratio", type=float, default=0.75, help="Flow matching ratio (vs LSD)")
    parser.add_argument("--hbm", type=int, default=4, help="Head batch multiplier")
    parser.add_argument("--eos-weight", type=float, default=0.1, help="EOS loss weight")

    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/pocket_tts_georgian")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()

    # Validate
    latents_dir = Path(args.latents_dir)
    if not (latents_dir / "metadata.json").exists():
        print(f"Pre-computed latents not found at {latents_dir}/")
        print("Run this first:")
        print(f"  python scripts/precompute_latents.py --manifest {args.manifest}")
        sys.exit(1)

    # Build config
    config = TrainingConfig(
        manifest_path=args.manifest,
        latents_dir=args.latents_dir,
        georgian_spm_path=args.georgian_spm,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        fm_ratio=args.fm_ratio,
        head_batch_multiplier=args.hbm,
        eos_loss_weight=args.eos_weight,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        mixed_precision=not args.no_amp,
    )

    # Print info
    print("=" * 60)
    print("Pocket TTS Training - Georgian Language")
    print("=" * 60)
    print(f"  Latents dir:  {config.latents_dir}")
    print(f"  Batch size:   {config.batch_size} (effective: {config.effective_batch_size})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs:       {config.num_epochs}")
    print(f"  Mixed prec:   {config.mixed_precision}")
    print(f"  Output:       {config.output_dir}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:          {gpu} ({mem:.1f} GB)")
    print("=" * 60)

    # Train
    trainer = PocketTTSTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
