"""
CSM-1B fine-tuning for Georgian TTS using Unsloth.

Usage:
    python train.py --data-dir ../../data/clean
    python train.py --data-dir ../../data/clean --lr 2e-4 --num-epochs 20
    python train.py --data-dir ../../data/clean --max-steps 5   # quick test
"""

import argparse
import time
from pathlib import Path

from unsloth import FastModel, is_bfloat16_supported
from transformers import (
    AutoProcessor,
    CsmForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from config import CSMConfig
from dataset import load_manifest, preprocess_example
from callbacks import WandbValSampleLogger


def build_dataset(manifest_path, processor, config, desc=""):
    """Load manifest -> HF Dataset -> preprocess via processor."""
    entries = load_manifest(str(manifest_path))
    ds = Dataset.from_list(entries)

    def preprocess_fn(example):
        return preprocess_example(
            example, processor, "speaker_id",
            config.max_text_tokens, config.max_audio_samples,
        )

    processed = ds.map(
        preprocess_fn,
        remove_columns=ds.column_names,
        desc=desc,
        num_proc=1,
    )
    return processed


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B on Georgian (Unsloth)")
    parser.add_argument("--data-dir", type=str, default="../../data/clean")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = CSMConfig()
    config.data_dir = args.data_dir
    if args.run_name:
        config.run_name = args.run_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    data_dir = Path(config.data_dir)
    train_manifest = data_dir / config.train_manifest
    eval_manifest = data_dir / config.eval_manifest

    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")

    # --- Model ---
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] Loading model: {config.model_name}")
    model, _ = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=2048,
        dtype=None,
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=False,
    )
    processor = AutoProcessor.from_pretrained(config.model_name)
    print(f"[{time.time()-t0:.1f}s] Model + processor loaded")

    # --- LoRA ---
    print(f"[{time.time()-t0:.1f}s] Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
    model = FastModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    print(f"[{time.time()-t0:.1f}s] LoRA applied")

    # --- Data ---
    print(f"[{time.time()-t0:.1f}s] Preprocessing train dataset...")
    processed_train = build_dataset(train_manifest, processor, config, desc="Preprocessing train")
    print(f"[{time.time()-t0:.1f}s] Train: {len(processed_train)} samples")

    processed_eval = None
    if eval_manifest.exists():
        print(f"[{time.time()-t0:.1f}s] Preprocessing eval dataset...")
        processed_eval = build_dataset(eval_manifest, processor, config, desc="Preprocessing eval")
        print(f"[{time.time()-t0:.1f}s] Eval: {len(processed_eval)} samples")

    # --- Training args ---
    steps_per_epoch = len(processed_train) // (config.batch_size * config.gradient_accumulation_steps)
    logging_steps = max(1, int(steps_per_epoch * 0.05))
    eval_steps = max(1, int(steps_per_epoch * 0.5))

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        eval_strategy="steps" if processed_eval else "no",
        eval_steps=eval_steps,
        save_strategy=config.save_strategy,
        save_steps=eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True if processed_eval else False,
        metric_for_best_model="eval_loss" if processed_eval else None,
        greater_is_better=False if processed_eval else None,
        optim="adamw_8bit",
        seed=config.seed,
        report_to="wandb",
        remove_unused_columns=False,
    )

    if args.max_steps:
        training_args.max_steps = args.max_steps

    # --- Callbacks ---
    callbacks = []
    if processed_eval:
        callbacks.append(WandbValSampleLogger(
            processor=processor,
            eval_dataset=processed_eval,
            sample_rate=config.sample_rate,
            n_examples=8,
        ))

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        callbacks=callbacks,
    )

    print(f"\n[{time.time()-t0:.1f}s] Starting training: {config.run_name}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  LR: {config.lr}")
    print(f"  LoRA: r={config.lora_r}")
    print(f"  Logging every {logging_steps} steps (~0.05 epoch)")
    print(f"  Eval every {eval_steps} steps (~0.5 epoch)")

    trainer.train(resume_from_checkpoint=args.resume)

    # --- Save ---
    final_dir = Path(config.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nModel saved to {final_dir}")


if __name__ == "__main__":
    main()
