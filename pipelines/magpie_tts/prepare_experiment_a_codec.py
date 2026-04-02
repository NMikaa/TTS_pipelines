#!/usr/bin/env python3
"""
Prepare Experiment A data: resample audio and extract NanoCodec tokens.

This runs the data preparation steps from train.py for the punctuated-only
Experiment A dataset (393h, 24 speakers, hold-out for zero-shot testing).

Usage:
    python prepare_experiment_a_codec.py

Timeline:
    - Audio resampling: ~10-15 minutes (single GPU)
    - NanoCodec token extraction: ~2 hours (on A100/H100, VRAM-intensive)
    - Total: ~2-3 hours
"""

import sys
from pathlib import Path

# Add magpie_tts to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from config import MagPIEConfig
from train import prepare_data


def main():
    # Configure for Experiment A data
    config = MagPIEConfig(
        data_dir="../../data/saba_experiment_a",
        train_manifest="train_manifest.json",
        eval_manifest="val_manifest.json",
        speaker_refs_manifest="holdout_manifest.json",  # Can be used for voice cloning context
        max_epochs=100,
        batch_size=48,
        learning_rate=2e-5,
        devices=2,  # DDP on 2 GPUs
    )

    print("=" * 70)
    print("EXPERIMENT A: DATA PREPARATION")
    print("=" * 70)
    print(f"Data directory: {config.data_dir}")
    print(f"Train manifest: {config.train_manifest}")
    print(f"Eval manifest: {config.eval_manifest}")
    print()

    # Run data preparation (steps 1-3: resample, codec, convert)
    print("Starting data preparation...")
    print("This will take 2-3 hours.")
    print()

    try:
        train_manifest, eval_manifest = prepare_data(config)
        print()
        print("=" * 70)
        print("DATA PREPARATION COMPLETE")
        print("=" * 70)
        print(f"Train manifest (NeMo): {train_manifest}")
        print(f"Eval manifest (NeMo): {eval_manifest}")
        print()
        print("Next steps:")
        print("  1. Check codec token extraction (ls data/saba_experiment_a/codec_codes/ | wc -l)")
        print("  2. Run training: python run_experiment_a.py")
    except Exception as e:
        print(f"Error during data preparation: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()