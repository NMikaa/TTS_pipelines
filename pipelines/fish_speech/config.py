"""OpenAudio S1-mini (Fish Speech) training configuration."""

from dataclasses import dataclass


@dataclass
class FishSpeechConfig:
    # Model — OpenAudio S1-mini (rebranded from Fish Speech)
    # Download: huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
    model_name: str = "fishaudio/openaudio-s1-mini"
    checkpoint_dir: str = "checkpoints/openaudio-s1-mini"

    # Data — expects speaker subdirs with .wav + .lab (plain text) files
    data_dir: str = "./data"
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training (Fish Speech uses PyTorch Lightning + Hydra)
    # These are passed to fish_speech/train.py via Hydra config overrides
    batch_size: int = 4  # Fish Speech default
    max_steps: int = 10000
    gradient_clip: float = 1.0

    # LoRA config — fine-tunes only the LLAMA component
    # Default: r_8_alpha_16 (rank=8, alpha=16)
    lora_config: str = "r_8_alpha_16"

    # Checkpointing
    output_dir: str = "results"
    run_name: str = "georgian_fish_v1"
    save_every_n_steps: int = 100
    val_every_n_steps: int = 100

    # Hardware
    num_workers: int = 1  # Fish Speech default for VQ extraction
    vq_batch_size: int = 16
    seed: int = 42
    mixed_precision: str = "bf16"
