#!/usr/bin/env python3
"""
Train CSM-1B on Georgian data.

Usage:
    # With voice actor manifest:
    python train_csm.py --voice-actor-path alignment/voice_actor_manifest.json \
                        --alignment-json alignments.json \
                        --run-name georgian_csm_v1

    # With HF shards:
    python train_csm.py --dataset-path /path/to/shard_ \
                        --alignment-json alignments.json \
                        --run-name georgian_csm_v1

    # Full fine-tuning (needs more VRAM):
    python train_csm.py --full-finetuning --voice-actor-path ... --run-name ...

    # Resume from fine-tuned checkpoint:
    python train_csm.py --base-model checkpoints/csm_georgian/.../checkpoint-XXX \
                        --voice-actor-path ... --run-name georgian_csm_v2

Environment variables:
    HF_TOKEN / HUGGINGFACE_TOKEN  - HuggingFace authentication
    WANDB_API_KEY                 - Weights & Biases API key (if using wandb)
"""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Train CSM-1B on Georgian data")

    # Model
    parser.add_argument("--model-id", default="sesame/csm-1b")
    parser.add_argument("--full-finetuning", action="store_true",
                        help="Full fine-tuning instead of LoRA")
    parser.add_argument("--base-model", default=None,
                        help="Path to fine-tuned model to resume from")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Data
    parser.add_argument("--dataset-path", default=None,
                        help="HF shards path prefix (e.g. /path/to/shard_)")
    parser.add_argument("--voice-actor-path", default=None,
                        help="JSONL manifest path")
    parser.add_argument("--alignment-json", default="alignments.json")
    parser.add_argument("--speakers", nargs="*", default=[],
                        help="Speaker IDs to filter")
    parser.add_argument("--test-size", type=float, default=0.05)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="checkpoints/csm_georgian")
    parser.add_argument("--report-to", default="tensorboard",
                        choices=["tensorboard", "wandb"])

    # Hub
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)

    args = parser.parse_args()

    # Validate data source
    if not args.dataset_path and not args.voice_actor_path:
        print("Error: Must provide --dataset-path or --voice-actor-path")
        sys.exit(1)

    # Build config
    from csm_training.config import CSMTrainingConfig
    config = CSMTrainingConfig(
        model_id=args.model_id,
        full_finetuning=args.full_finetuning,
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dataset_path=args.dataset_path,
        voice_actor_path=args.voice_actor_path,
        alignment_json_path=args.alignment_json,
        list_of_speakers=args.speakers,
        test_size=args.test_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        run_name=args.run_name,
        output_dir=args.output_dir,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Print info
    print("=" * 60)
    print("CSM-1B Training - Georgian Language")
    print("=" * 60)
    print(f"  Model:        {config.model_id}")
    print(f"  Mode:         {'Full' if config.full_finetuning else f'LoRA r={config.lora_r}'}")
    print(f"  Batch size:   {config.batch_size} (effective: {config.effective_batch_size})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs:       {config.num_epochs}")
    print(f"  Output:       {config.output_dir}")
    print(f"  Logging:      {config.report_to}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU:          {gpu} ({mem:.1f} GB)")
    print("=" * 60)

    # Train
    from csm_training.trainer import CSMTrainer
    trainer = CSMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
