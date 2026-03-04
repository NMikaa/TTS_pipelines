"""
CSM-1B fine-tuning for Georgian.

Usage:
    python train.py --data-dir ./data --run-name georgian_csm_v1
    python train.py --data-dir ./data --full-finetune --run-name georgian_csm_full
"""

import argparse
import sys
from pathlib import Path

from config import CSMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_csm_v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--full-finetune", action="store_true", help="Full fine-tuning instead of LoRA")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = CSMConfig(data_dir=args.data_dir, run_name=args.run_name)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.full_finetune:
        config.use_lora = False

    entries = prepare_dataset(config.data_dir)
    train_ids, val_ids, test_ids = get_splits(config.data_dir)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # TODO: Implement CSM-1B fine-tuning
    # 1. Load sesame/csm-1b from HuggingFace
    # 2. Apply LoRA (or full fine-tune)
    # 3. Build dataset with word-level alignment
    # 4. Train with HuggingFace Trainer
    raise NotImplementedError(
        "CSM-1B fine-tuning not yet implemented. "
        "See https://github.com/SesameAILabs/csm for the upstream model."
    )


if __name__ == "__main__":
    main()
