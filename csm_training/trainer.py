"""
CSM-1B Trainer using HuggingFace Trainer + Unsloth for efficient LoRA fine-tuning.

Supports:
- LoRA fine-tuning via Unsloth FastModel (efficient, lower VRAM)
- Full fine-tuning for larger GPU setups
- Gradient checkpointing, 8-bit AdamW, cosine LR
- WandB or TensorBoard logging
- Automatic dataset validation and error filtering
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import CSMTrainingConfig
from .dataset import (
    CSMDataLoader,
    TTSTrainingDataset,
    filter_errored_samples,
)


class CSMTrainer:
    """End-to-end CSM-1B training manager."""

    def __init__(self, config: CSMTrainingConfig):
        self.config = config
        self._setup_environment()

        # Load model + processor
        self.model, self.processor = self._load_model()

        # Load data
        self.train_ds, self.valid_ds = self._load_datasets()

    def _setup_environment(self):
        """Configure environment before torch import side effects."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # CUDA memory optimization
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8",
        )

        # HuggingFace auth
        hf_token = self.config.hf_token
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        # Logging backend
        if self.config.report_to == "wandb":
            wandb_key = self.config.wandb_api_key
            if wandb_key:
                os.environ["WANDB_API_KEY"] = wandb_key
            os.environ["WANDB_PROJECT"] = self.config.wandb_project
            os.environ["WANDB_DIR"] = self.config.wandb_dir

    def _load_model(self) -> Tuple:
        """Load CSM-1B with optional LoRA via Unsloth."""
        from transformers import AutoProcessor, CsmForConditionalGeneration
        from unsloth import FastModel

        cfg = self.config
        processor = AutoProcessor.from_pretrained(cfg.model_id)

        model_name = cfg.base_model or "unsloth/csm-1b"

        model, _ = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_text,
            dtype=None,
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=False,
            full_finetuning=cfg.full_finetuning,
        )

        if not cfg.full_finetuning and cfg.base_model is None:
            model = FastModel.get_peft_model(
                model,
                r=cfg.lora_r,
                target_modules=cfg.lora_target_modules,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=cfg.seed,
                use_rslora=False,
                loftq_config=None,
            )

        print(f"Model loaded: {model_name}")
        if not cfg.full_finetuning:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        return model, processor

    def _load_datasets(self) -> Tuple:
        """Load and prepare train/valid datasets."""
        cfg = self.config

        data_loader = CSMDataLoader(
            shards_path=cfg.dataset_path,
            voice_actor_path=cfg.voice_actor_path,
            list_of_speakers=cfg.list_of_speakers,
        )
        data_loader.load_alignments(cfg.alignment_json_path)

        if cfg.list_of_speakers:
            data_loader.ds = data_loader._filter_speakers(
                data_loader.ds, cfg.list_of_speakers
            )

        train_raw, valid_raw = data_loader.split_train_test(
            cfg.test_size, cfg.seed
        )

        # Wrap with training dataset
        train_ds = TTSTrainingDataset(train_raw, data_loader.alignment_dict, self.processor)
        valid_ds = TTSTrainingDataset(valid_raw, data_loader.alignment_dict, self.processor)

        # Filter errored samples
        print("Validating train dataset...")
        train_ds = filter_errored_samples(train_ds)
        print("Validating valid dataset...")
        valid_ds = filter_errored_samples(valid_ds)

        print(f"Final dataset sizes - Train: {len(train_ds)}, Valid: {len(valid_ds)}")
        return train_ds, valid_ds

    def train(self):
        """Run training with HuggingFace Trainer."""
        from transformers import Trainer, TrainingArguments
        from unsloth import is_bfloat16_supported

        cfg = self.config

        # Freeze codec model (only train the language model)
        self.model.train()
        self.model.codec_model.eval()

        # Calculate step frequencies
        steps_per_epoch = max(1, len(self.train_ds) // cfg.effective_batch_size)
        logging_steps = max(1, steps_per_epoch // cfg.logging_steps_per_epoch)
        eval_steps = max(1, steps_per_epoch // cfg.eval_saves_per_epoch)

        print(f"\nTraining plan:")
        print(f"  Steps/epoch:   {steps_per_epoch}")
        print(f"  Total steps:   {steps_per_epoch * cfg.num_epochs}")
        print(f"  Log every:     {logging_steps} steps")
        print(f"  Eval/save:     {eval_steps} steps")

        checkpoint_dir = str(Path(cfg.output_dir) / cfg.run_name / "checkpoints")

        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            remove_unused_columns=False,
            gradient_checkpointing=cfg.gradient_checkpointing,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            dataloader_num_workers=cfg.num_workers,
            no_cuda=False,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            num_train_epochs=cfg.num_epochs,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            logging_steps=logging_steps,
            dataloader_prefetch_factor=4,
            logging_dir=f"{checkpoint_dir}/logs",
            logging_strategy="steps",
            report_to=[cfg.report_to],
            save_steps=eval_steps,
            save_strategy="steps",
            greater_is_better=False,
            lr_scheduler_type=cfg.lr_scheduler_type,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            run_name=cfg.run_name,
            eval_steps=eval_steps,
            eval_strategy="steps",
            seed=cfg.seed,
        )

        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.valid_ds,
        )

        print(f"\nStarting training:")
        print(f"  Epochs:     {cfg.num_epochs}")
        print(f"  LR:         {cfg.learning_rate}")
        print(f"  Batch size: {cfg.batch_size} (effective: {cfg.effective_batch_size})")
        print(f"  Output:     {checkpoint_dir}")
        print(f"  Train size: {len(self.train_ds)}")
        print(f"  Valid size: {len(self.valid_ds)}")

        trainer.train()

        # Optional: push to hub
        if cfg.push_to_hub and cfg.hub_model_id:
            hf_token = cfg.hf_token
            self.model.push_to_hub(
                cfg.hub_model_id, private=True, token=hf_token
            )
            print(f"Model pushed to {cfg.hub_model_id}")
