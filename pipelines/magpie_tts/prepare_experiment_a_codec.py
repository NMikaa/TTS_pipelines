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
import logging
from pathlib import Path
from datetime import datetime

# Add magpie_tts to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from config import MagPIEConfig
from train import prepare_data

# Setup logging to both console and file
log_dir = Path("../../data/saba_experiment_a/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"codec_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


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

    logger.info("=" * 70)
    logger.info("EXPERIMENT A: DATA PREPARATION")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Train manifest: {config.train_manifest}")
    logger.info(f"Eval manifest: {config.eval_manifest}")
    logger.info(f"Hold-out manifest: {config.speaker_refs_manifest}")
    logger.info("")
    logger.info("Timeline: ~2-3 hours total")
    logger.info("  - Resampling: ~10-15 minutes")
    logger.info("  - Codec extraction: ~1-2 hours (GPU-intensive)")
    logger.info("  - Manifest conversion: ~5-10 minutes")
    logger.info("")

    # Run data preparation (steps 1-3: resample, codec, convert)
    logger.info("Starting data preparation...")
    logger.info("")

    try:
        train_manifest, eval_manifest = prepare_data(config)
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ DATA PREPARATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Train manifest (NeMo): {train_manifest}")
        logger.info(f"Eval manifest (NeMo): {eval_manifest}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Verify codec extraction:")
        logger.info("     ls data/saba_experiment_a/codec_codes/ | wc -l")
        logger.info("  2. Launch training:")
        logger.info("     python run_experiment_a.py")
    except Exception as e:
        logger.error(f"ERROR during data preparation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()