"""Qwen3-TTS training configuration."""

from dataclasses import dataclass


@dataclass
class Qwen3TTSConfig:
    # Model
    model_name: str = "Qwen/Qwen3-TTS"

    # Data
    data_dir: str = "./data"
    manifest: str = "alignment/voice_actor_manifest.json"
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training
    batch_size: int = 8
    lr: float = 5e-5
    num_epochs: int = 20
    warmup_steps: int = 200
    grad_accum: int = 2
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Fine-tuning
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32

    # Checkpointing
    output_dir: str = "checkpoints"
    run_name: str = "georgian_qwen3tts_v1"
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500

    # Hardware
    num_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bf16"
