"""CSM-1B training configuration."""

from dataclasses import dataclass


@dataclass
class CSMConfig:
    # Model
    model_name: str = "sesame/csm-1b"

    # Data
    data_dir: str = "./data"
    manifest: str = "alignment/voice_actor_manifest.json"
    alignments: str = "alignments.json"
    sample_rate: int = 24000
    max_duration_s: float = 120.0
    max_text_length: int = 4096

    # Training
    batch_size: int = 64
    lr: float = 1e-4
    num_epochs: int = 6
    warmup_steps: int = 5
    grad_accum: int = 2
    weight_decay: float = 0.01

    # LoRA
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 32

    # Checkpointing
    output_dir: str = "checkpoints"
    run_name: str = "georgian_csm_v1"
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500

    # Hardware
    num_workers: int = 12
    seed: int = 42
    gradient_checkpointing: bool = True
