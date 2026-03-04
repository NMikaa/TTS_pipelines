"""
Orpheus TTS fine-tuning for Georgian.

Usage:
    python train.py --data-dir ./data --run-name georgian_orpheus_v1
    python train.py --data-dir ./data --model-size 1b --run-name georgian_orpheus_1b
"""

import argparse
import sys
from pathlib import Path

from config import OrpheusConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Orpheus TTS on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_orpheus_v1")
    parser.add_argument("--model-size", type=str, default="400m", choices=["150m", "400m", "1b", "3b"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--full-finetune", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = OrpheusConfig(data_dir=args.data_dir, run_name=args.run_name, model_size=args.model_size)
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

    # TODO: Implement Orpheus fine-tuning
    # 1. Load pre-trained Orpheus model (selected size)
    # 2. Apply LoRA if configured
    # 3. Tokenize audio data into Orpheus audio tokens
    # 4. Fine-tune with HuggingFace Trainer or Unsloth
    raise NotImplementedError(
        "Orpheus TTS fine-tuning not yet implemented. "
        "See https://github.com/canopyai/Orpheus-TTS for upstream training code."
    )


if __name__ == "__main__":
    main()
