"""
CosyVoice 3 fine-tuning for Georgian.

Usage:
    python train.py --data-dir ./data --run-name georgian_cosyvoice_v1
    python train.py --resume checkpoints/georgian_cosyvoice_v1/epoch_5.pt
"""

import argparse
import sys
from pathlib import Path

from config import CosyVoiceConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CosyVoice 3 on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_cosyvoice_v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = CosyVoiceConfig(data_dir=args.data_dir, run_name=args.run_name)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    entries = prepare_dataset(config.data_dir)
    train_ids, val_ids, test_ids = get_splits(config.data_dir)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # TODO: Implement CosyVoice 3 fine-tuning
    # 1. Clone/install CosyVoice from GitHub
    # 2. Load pre-trained CosyVoice 3 model
    # 3. Prepare dataset in CosyVoice SFT format
    # 4. Run SFT training script
    raise NotImplementedError(
        "CosyVoice 3 fine-tuning not yet implemented. "
        "See https://github.com/FunAudioLLM/CosyVoice for the upstream training code."
    )


if __name__ == "__main__":
    main()
