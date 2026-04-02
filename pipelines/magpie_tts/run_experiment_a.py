#!/usr/bin/env python3
"""
Run Experiment A: MagPIE TTS fine-tuning on 393h punctuated Georgian speech.

This script:
1. Verifies data preparation is complete (codec tokens extracted)
2. Launches distributed training on 2 GPUs using DDP
3. Uses Hydra config overrides for Experiment A settings
4. Saves checkpoints and logs to exp/experiment_a_punctuated_joint_training/

Usage:
    # Run training (automatic checkpoint resume if interrupted)
    python run_experiment_a.py

    # Custom learning rate
    python run_experiment_a.py model.optim.lr=1e-5

    # Resume from specific checkpoint
    python run_experiment_a.py trainer.ckpt_path=exp/experiment_a_punctuated_joint_training/.../checkpoint.ckpt

Timeline:
    - Training: ~2-3 weeks on 2 GPUs (100 epochs, likely ~80 with early stopping)
    - Early stopping: patience=5, stops if val loss doesn't improve for 5 epochs
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path("../../data/saba_experiment_a/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


def verify_data_preparation():
    """Check that codec tokens were extracted successfully."""
    data_dir = Path("../../data/saba_experiment_a")

    checks = {
        "train_manifest.json": data_dir / "train_manifest.json",
        "val_manifest.json": data_dir / "val_manifest.json",
        "train_manifest_nemo.json": data_dir / "train_manifest_nemo.json",
        "val_manifest_nemo.json": data_dir / "val_manifest_nemo.json",
        "audio_22khz/": data_dir / "audio_22khz",
        "codec_codes/": data_dir / "codec_codes",
    }

    logger.info("Verifying data preparation...")
    missing = []
    for name, path in checks.items():
        if not path.exists():
            missing.append(name)
        else:
            if path.is_dir():
                count = len(list(path.glob("*")))
                logger.info(f"  ✓ {name:30} ({count:,} files)")
            else:
                size_mb = path.stat().st_size / 1024 / 1024
                logger.info(f"  ✓ {name:30} ({size_mb:,.1f} MB)")

    if missing:
        logger.error(f"❌ Missing files/directories:")
        for name in missing:
            logger.error(f"    - {name}")
        logger.error("\nRun data preparation first:")
        logger.error("  python prepare_experiment_a_codec.py")
        return False

    logger.info("✓ Data preparation verified\n")
    return True


def launch_training():
    """Launch distributed training using NeMo's training script."""
    data_dir = Path("../../data/saba_experiment_a").resolve()
    nemo_root = Path(__file__).resolve().parents[2] / "NeMo"
    nemo_script = nemo_root / "examples" / "tts" / "magpietts.py"

    if not nemo_script.exists():
        logger.error(f"❌ NeMo training script not found at {nemo_script}")
        logger.error("Clone NeMo: git clone --depth 1 https://github.com/NVIDIA/NeMo.git")
        return False

    # Get NeMo manifests
    train_manifest = data_dir / "train_manifest_nemo.json"
    val_manifest = data_dir / "val_manifest_nemo.json"
    audio_dir = data_dir / "audio_22khz"
    codes_dir = data_dir / "codec_codes"

    # Build Hydra overrides
    overrides = [
        # Config: use Experiment A config
        "--config-name=magpietts_experiment_a",
        # Data paths
        f"+train_ds_meta.georgian_train.manifest_path={train_manifest}",
        f"+train_ds_meta.georgian_train.audio_dir={audio_dir}",
        f"+train_ds_meta.georgian_train.feature_dir={codes_dir}",
        f"+val_ds_meta.georgian_eval.manifest_path={val_manifest}",
        f"+val_ds_meta.georgian_eval.audio_dir={audio_dir}",
        f"+val_ds_meta.georgian_eval.feature_dir={codes_dir}",
        # Trainer: DDP on 2 GPUs
        "trainer.devices=2",
        "trainer.strategy=ddp",
    ]

    logger.info("=" * 70)
    logger.info("EXPERIMENT A: TRAINING")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Config: magpietts_experiment_a.yaml")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"NeMo script: {nemo_script}")
    logger.info("")
    logger.info("Training configuration:")
    logger.info("  - Model: nvidia/magpie_tts_multilingual_357m (357M params)")
    logger.info("  - Data: 182k samples (~393h) from 24 speakers")
    logger.info("  - Epochs: 100 (target ~80 with early stopping)")
    logger.info("  - Batch size: 48 per GPU × 2 = 96 effective")
    logger.info("  - Learning rate: 2e-5 (fine-tuning rate)")
    logger.info("  - Precision: bf16-mixed (saves ~50% VRAM)")
    logger.info("  - Early stopping: patience=5")
    logger.info(f"  - Distributed: DDP on 2 GPUs")
    logger.info("")
    logger.info("W&B Integration:")
    logger.info("  - Project: georgian-tts")
    logger.info("  - Run: magpie-experiment-a")
    logger.info("  - Monitor at: https://wandb.ai/")
    logger.info("")

    # Set environment variables for distributed training
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    logger.info("Launching training...")
    logger.info("")

    cmd = [
        sys.executable,
        str(nemo_script),
    ] + overrides

    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent), env={**dict(__import__('os').environ), **env})
        return result.returncode == 0
    except Exception as e:
        logger.error(f"❌ Error launching training: {e}", exc_info=True)
        return False


def main():
    logger.info("")
    logger.info("Log file: " + str(log_file))

    # Verify data
    if not verify_data_preparation():
        sys.exit(1)

    # Launch training
    success = launch_training()

    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Training launched successfully")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Monitor progress:")
        logger.info("  - TensorBoard: tensorboard --logdir=exp/")
        logger.info("  - WandB: https://wandb.ai/")
        logger.info("  - Training log: " + str(log_file))
        logger.info("")
        logger.info("Training will resume automatically from last checkpoint if interrupted.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
