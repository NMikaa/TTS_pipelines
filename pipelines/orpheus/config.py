"""Orpheus TTS training configuration."""

from dataclasses import dataclass


@dataclass
class OrpheusConfig:
    # Model
    model_name: str = "canopylabs/orpheus-tts-0.4b"
    model_size: str = "400m"  # 150m, 400m, 1b, 3b

    # Data
    data_dir: str = "./data"
    manifest: str = "alignment/voice_actor_manifest.json"
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training
    batch_size: int = 8
    lr: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 200
    grad_accum: int = 4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16

    # Checkpointing
    output_dir: str = "checkpoints"
    run_name: str = "georgian_orpheus_v1"
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500

    # Hardware
    num_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bf16"
