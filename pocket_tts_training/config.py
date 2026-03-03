"""Training configuration for Pocket TTS fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    # --- Data ---
    manifest_path: str = "alignment/voice_actor_manifest.json"
    latents_dir: str = "latents_cache"
    max_audio_seconds: float = 15.0

    # --- Model ---
    input_proj_weights_path: str = "quantizer_input_proj_weight.safetensors"
    pretrained_model_variant: str = "b6369a24"

    # --- Optimization ---
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = -1  # -1 = use num_epochs
    num_epochs: int = 10
    gradient_clip: float = 1.0
    betas: tuple = (0.9, 0.95)

    # --- LSD Loss ---
    fm_ratio: float = 0.75  # 75% flow matching, 25% LSD
    head_batch_multiplier: int = 4

    # --- EMA ---
    ema_decay: float = 0.9999
    ema_start_step: int = 1000

    # --- Precision ---
    mixed_precision: bool = True
    dtype: str = "float16"  # float16 or bfloat16

    # --- Logging ---
    log_every: int = 50
    save_every_epoch: bool = True
    output_dir: str = "checkpoints/pocket_tts_georgian"
    experiment_name: str = "pocket_tts_georgian"

    # --- System ---
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def amp_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
