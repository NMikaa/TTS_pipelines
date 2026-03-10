"""
F5-TTS fine-tuning for Georgian.

Two-step process:
  1. Convert JSONL manifest → pipe-delimited CSV → Arrow dataset (prepare_csv_wavs)
  2. Fine-tune F5-TTS v1 Base with extended vocab (pretrained + Georgian chars)

Usage:
    # Default (recommended for A6000 48GB)
    python train.py --data-dir ../../data/clean

    # Custom settings
    python train.py --data-dir ../../data/clean --lr 5e-6 --batch-size 4000 --epochs 150

    # Resume from checkpoint (automatic — just re-run same command)
    python train.py --data-dir ../../data/clean
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import F5TTSConfig


def _ensure_symlink(pkg_data_dir: Path, local_dir: Path, name: str):
    """Create symlink in F5-TTS package data dir pointing to our local data."""
    pkg_data_dir.mkdir(parents=True, exist_ok=True)
    link_path = pkg_data_dir / name
    if link_path.is_symlink() or link_path.exists():
        return
    link_path.symlink_to(local_dir)
    print(f"  Symlinked {link_path} -> {local_dir}")


def manifest_to_csv(manifest_path: str, csv_path: str, audio_base_dir: str) -> int:
    """Convert JSONL manifest to F5-TTS pipe-delimited CSV format."""
    audio_base = Path(audio_base_dir).resolve()
    count = 0

    with open(manifest_path) as f_in, open(csv_path, "w", encoding="utf-8") as f_out:
        f_out.write("audio_file|text\n")
        for line in f_in:
            entry = json.loads(line.strip())
            audio_path = Path(entry["audio_path"])
            if not audio_path.is_absolute():
                audio_path = audio_base / audio_path
            audio_path = audio_path.resolve()

            if not audio_path.exists():
                print(f"Warning: audio not found, skipping: {audio_path}")
                continue

            text = entry["text"].strip()
            if not text:
                continue

            f_out.write(f"{audio_path}|{text}\n")
            count += 1

    return count


def prepare_dataset(config: F5TTSConfig):
    """Step 1: Convert manifest to CSV and prepare Arrow dataset."""
    data_dir = Path(config.data_dir).resolve()
    manifest_path = data_dir / "train_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {manifest_path}")

    from importlib.resources import files as pkg_files

    pkg_data_dir = Path(str(pkg_files("f5_tts").joinpath("../../data"))).resolve()
    dataset_dir_name = f"{config.dataset_name}_{config.tokenizer}"
    local_data_dir = Path("data")
    csv_path = local_data_dir / "train.csv"
    arrow_dir = local_data_dir / dataset_dir_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if Arrow dataset already exists
    if (arrow_dir / "raw.arrow").exists():
        print(f"Arrow dataset already exists at {arrow_dir}, skipping preparation.")
        _ensure_symlink(pkg_data_dir, arrow_dir.resolve(), dataset_dir_name)
        return

    # Step 1a: Convert JSONL manifest to CSV
    print("Step 1a: Converting JSONL manifest to CSV...")
    repo_root = Path(__file__).resolve().parents[2]
    count = manifest_to_csv(str(manifest_path), str(csv_path), str(repo_root))
    print(f"  Wrote {count} entries to {csv_path}")

    # Step 1b: Prepare Arrow dataset
    print("Step 1b: Preparing Arrow dataset (this may take a few minutes)...")
    from f5_tts.train.datasets.prepare_csv_wavs import prepare_and_save_set

    prepare_and_save_set(
        inp_dir=str(csv_path),
        out_dir=str(arrow_dir),
        is_finetune=False,
    )
    print(f"  Arrow dataset saved to {arrow_dir}")

    # Step 1c: Replace vocab with extended vocab (pretrained + Georgian chars)
    extended_vocab_path = local_data_dir / "extended_vocab.txt"
    if extended_vocab_path.exists():
        import shutil

        shutil.copy2(str(extended_vocab_path), str(arrow_dir / "vocab.txt"))
        print("  Replaced vocab.txt with extended vocab (pretrained + Georgian)")

    # Symlink into F5-TTS package data dir
    _ensure_symlink(pkg_data_dir, arrow_dir.resolve(), dataset_dir_name)


def _prepare_extended_checkpoint(checkpoint_path: str, extended_vocab_size: int):
    """Resize pretrained checkpoint's text_embed to match extended vocab.

    The pretrained model has 2546 text embeddings (2545 pinyin + 1 filler).
    We extend to extended_vocab_size + 1 by appending rows initialized with
    the mean of existing embeddings.
    """
    extended_ckpt = os.path.join(checkpoint_path, "pretrained_model_1250000_extended.safetensors")
    if os.path.isfile(extended_ckpt):
        return extended_ckpt

    import torch
    from cached_path import cached_path
    from safetensors.torch import load_file, save_file

    print("Downloading pretrained F5-TTS v1 Base checkpoint...")
    ckpt_path = str(
        cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
    )
    print(f"  Source: {ckpt_path}")

    state_dict = load_file(ckpt_path)
    key = "ema_model.transformer.text_embed.text_embed.weight"
    old_embed = state_dict[key]
    new_size = extended_vocab_size + 1  # +1 for filler token at index 0
    print(f"  Resizing text_embed: {old_embed.shape[0]} -> {new_size}")

    new_embed = torch.zeros(new_size, old_embed.shape[1])
    new_embed[: old_embed.shape[0]] = old_embed
    new_embed[old_embed.shape[0] :] = old_embed.mean(dim=0)
    state_dict[key] = new_embed

    save_file(state_dict, extended_ckpt)
    print(f"  Saved: {extended_ckpt}")
    return extended_ckpt


def train(config: F5TTSConfig):
    """Step 2: Fine-tune F5-TTS."""
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import load_dataset
    from f5_tts.model.utils import get_tokenizer

    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=config.sample_rate,
        mel_spec_type="vocos",
    )

    # Checkpoint path
    from importlib.resources import files as pkg_files

    checkpoint_path = os.path.join("ckpts", config.dataset_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    pkg_ckpts_dir = Path(str(pkg_files("f5_tts").joinpath("../../ckpts"))).resolve()
    _ensure_symlink(pkg_ckpts_dir, Path(checkpoint_path).resolve(), config.dataset_name)

    # Tokenizer — use "custom" with extended vocab path
    vocab_path = str(Path("data") / f"{config.dataset_name}_{config.tokenizer}" / "vocab.txt")
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    print(f"\nVocab size: {vocab_size}")

    # Prepare extended checkpoint (resize text_embed for new vocab)
    if config.finetune:
        extended_ckpt = _prepare_extended_checkpoint(checkpoint_path, vocab_size)
        # The trainer picks up pretrained_* files automatically

    # Build model
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Trainer
    trainer = Trainer(
        model,
        config.epochs,
        config.learning_rate,
        num_warmup_updates=config.num_warmup_updates,
        save_per_updates=config.save_per_updates,
        keep_last_n_checkpoints=config.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=config.batch_size_per_gpu,
        batch_size_type=config.batch_size_type,
        max_samples=config.max_samples,
        grad_accumulation_steps=config.grad_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        logger=config.logger,
        wandb_project="georgian-tts",
        wandb_run_name="f5-tts",
        wandb_resume_id=None,
        log_samples=config.log_samples,
        last_per_updates=config.last_per_updates,
        bnb_optimizer=config.bnb_optimizer,
    )

    # Load dataset — use "custom" tokenizer with our vocab path
    train_dataset = load_dataset(
        config.dataset_name, config.tokenizer, mel_spec_kwargs=mel_spec_kwargs
    )

    print(f"\nStarting training:")
    print(f"  Epochs: {config.epochs}")
    print(f"  LR: {config.learning_rate}")
    print(f"  Batch size (frames): {config.batch_size_per_gpu}")
    print(f"  Max samples/batch: {config.max_samples}")
    print(f"  Warmup steps: {config.num_warmup_updates}")
    print(f"  8-bit optimizer: {config.bnb_optimizer}")
    print(f"  Checkpoints: {checkpoint_path}")
    print()

    trainer.train(
        train_dataset,
        resumable_with_seed=config.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS on Georgian")
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument("--dataset-name", type=str, default="georgian_tts")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size in frames (default: 3200)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset, don't train")
    args = parser.parse_args()

    config = F5TTSConfig(data_dir=args.data_dir, dataset_name=args.dataset_name)
    if args.batch_size:
        config.batch_size_per_gpu = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.epochs = args.epochs
    if args.no_wandb:
        config.logger = None

    # Step 1: Prepare dataset
    prepare_dataset(config)

    if args.prepare_only:
        print("\nDataset preparation complete. Exiting (--prepare-only).")
        return

    # Step 2: Train
    train(config)


if __name__ == "__main__":
    main()
