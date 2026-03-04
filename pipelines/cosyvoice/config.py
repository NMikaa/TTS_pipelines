"""CosyVoice 3 training configuration."""

from dataclasses import dataclass


@dataclass
class CosyVoiceConfig:
    # CosyVoice 3 (December 2025, 1M+ hours) — check HuggingFace for latest checkpoint
    model_name: str = "FunAudioLLM/CosyVoice2-0.5B"  # TODO: update to v3 checkpoint when available

    # Data
    data_dir: str = "./data"
    manifest: str = "alignment/voice_actor_manifest.json"
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training
    batch_size: int = 8
    lr: float = 1e-4
    num_epochs: int = 20
    warmup_steps: int = 500
    grad_accum: int = 2
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Checkpointing
    output_dir: str = "checkpoints"
    run_name: str = "georgian_cosyvoice_v1"
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500

    # Hardware
    num_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bf16"
