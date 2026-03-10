"""F5-TTS training configuration."""

from dataclasses import dataclass


@dataclass
class F5TTSConfig:
    # Model
    exp_name: str = "F5TTS_v1_Base"  # "F5TTS_v1_Base" or "F5TTS_Base"

    # Data
    data_dir: str = "../../data/clean"
    dataset_name: str = "georgian_tts"  # used as output dir name under ckpts/
    sample_rate: int = 24000
    max_duration_s: float = 15.0

    # Training — F5-TTS uses frame-based batch sizing
    batch_size_per_gpu: int = 9600  # in frames (not samples). ~40GB VRAM on A6000 48GB
    batch_size_type: str = "frame"  # "frame" or "sample"
    max_samples: int = 64  # max sequences per batch
    learning_rate: float = 1e-5  # 1e-5 recommended for fine-tuning
    epochs: int = 100
    num_warmup_updates: int = 500  # 500 for fine-tuning (not 20000 as in pretraining)
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Fine-tuning
    finetune: bool = True
    pretrain_ckpt: str = None  # auto-downloads from HF if None
    tokenizer: str = "char"  # MUST be "char" for Georgian (not "pinyin")

    # Checkpointing
    save_per_updates: int = 10000  # ~3.2GB per checkpoint, save less often
    last_per_updates: int = 5000
    keep_last_n_checkpoints: int = 2  # 2 checkpoints + model_last.pt = ~10GB max

    # Logging
    logger: str = "wandb"  # None, "wandb", or "tensorboard"
    log_samples: bool = True

    # Hardware
    bnb_optimizer: bool = True  # 8-bit Adam to save VRAM
    seed: int = 42
