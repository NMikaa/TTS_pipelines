"""Training configuration for CSM-1B fine-tuning."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CSMTrainingConfig:
    """All hyperparameters for CSM-1B training."""

    # Model
    model_id: str = "sesame/csm-1b"
    full_finetuning: bool = False
    base_model: Optional[str] = None  # Path to resume from a fine-tuned model

    # LoRA (only used when full_finetuning=False)
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Data
    dataset_path: Optional[str] = None  # HF shards path (shard_XXXX format)
    voice_actor_path: Optional[str] = None  # JSONL manifest path
    alignment_json_path: str = "alignments.json"
    list_of_speakers: List[str] = field(default_factory=list)
    test_size: float = 0.05

    # Audio constants
    frame_size: int = 1920  # 24kHz * 0.08s
    target_sr: int = 24_000
    max_audio: int = 24_000 * 120  # 120s max
    max_text: int = 4096
    cutoffs_len: int = 128

    # Training
    learning_rate: float = 1e-4
    num_epochs: int = 6
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    num_workers: int = 12
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    gradient_checkpointing: bool = True
    seed: int = 42

    # Logging & Saving
    logging_steps_per_epoch: int = 20
    eval_saves_per_epoch: int = 4
    run_name: str = "csm_georgian"
    output_dir: str = "checkpoints/csm_georgian"

    # Logging backend
    wandb_project: str = "CSM Finetuning"
    wandb_dir: str = "./wandb"
    report_to: str = "tensorboard"  # "wandb" or "tensorboard"

    # Push to hub (optional)
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def hf_token(self) -> Optional[str]:
        """Read HuggingFace token from environment variable."""
        return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    @property
    def wandb_api_key(self) -> Optional[str]:
        """Read WandB API key from environment variable."""
        return os.environ.get("WANDB_API_KEY")
