"""
Qwen3-TTS fine-tuning for Georgian.

Usage:
    python train.py --data-dir ./data --run-name georgian_qwen3tts_v1
"""

import argparse
import sys
from pathlib import Path

from config import Qwen3TTSConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_qwen3tts_v1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = Qwen3TTSConfig(data_dir=args.data_dir, run_name=args.run_name)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.no_lora:
        config.use_lora = False

    entries = prepare_dataset(config.data_dir)
    train_ids, val_ids, test_ids = get_splits(config.data_dir)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # TODO: Implement Qwen3-TTS fine-tuning
    # 1. Load pre-trained Qwen3-TTS from HuggingFace
    # 2. Prepare dataset in Qwen TTS format
    # 3. Apply LoRA or full fine-tune
    # 4. Train with standard transformers Trainer or custom loop
    raise NotImplementedError(
        "Qwen3-TTS fine-tuning not yet implemented. "
        "See https://huggingface.co/Qwen/Qwen3-TTS for model details."
    )


if __name__ == "__main__":
    main()
