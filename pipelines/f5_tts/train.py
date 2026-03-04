"""
F5-TTS fine-tuning for Georgian.

Usage:
    python train.py --data-dir ./data --run-name georgian_f5_v1
    python train.py --resume checkpoints/georgian_f5_v1/epoch_5.pt
"""

import argparse
import sys
from pathlib import Path

from config import F5TTSConfig

# Add repo root to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def main():
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_f5_v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = F5TTSConfig(
        data_dir=args.data_dir,
        run_name=args.run_name,
    )
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    # Load data with shared splits
    print("Loading dataset...")
    entries = prepare_dataset(config.data_dir)
    train_ids, val_ids, test_ids = get_splits(config.data_dir)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # TODO: Implement F5-TTS fine-tuning
    # 1. Load pre-trained F5-TTS model
    # 2. Prepare dataset in F5-TTS format
    # 3. Set up training loop (or use F5-TTS's built-in trainer)
    # 4. Train with checkpointing
    raise NotImplementedError(
        "F5-TTS fine-tuning not yet implemented. "
        "See https://github.com/SWivid/F5-TTS for the upstream training code."
    )


if __name__ == "__main__":
    main()
