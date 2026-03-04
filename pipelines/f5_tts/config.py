"""F5-TTS training configuration."""

from dataclasses import dataclass, field


@dataclass
class F5TTSConfig:
    # Model
    model_name: str = "SWivid/F5-TTS"

    # Data
    data_dir: str = "./data"
    manifest: str = "alignment/voice_actor_manifest.json"
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training
    batch_size: int = 16
    lr: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 500
    grad_accum: int = 1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Fine-tuning
    freeze_text_encoder: bool = False

    # Checkpointing
    output_dir: str = "checkpoints"
    run_name: str = "georgian_f5_v1"
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500

    # Hardware
    num_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bf16"
