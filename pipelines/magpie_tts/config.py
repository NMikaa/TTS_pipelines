"""MagPIE TTS fine-tuning configuration for Georgian."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MagPIEConfig:
    # Model
    pretrained_model: str = "nvidia/magpie_tts_multilingual_357m"
    codec_model: str = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"

    # Data
    data_dir: str = "../../data/clean"
    train_manifest: str = "train_manifest.json"
    eval_manifest: str = "eval_manifest.json"
    speaker_refs_manifest: str = "speaker_refs_manifest.json"
    sample_rate: int = 22050  # NanoCodec requires 22,050 Hz (NOT 24kHz)
    source_sample_rate: int = 24000  # Our audio is 24kHz, needs resampling
    max_duration_s: float = 15.0

    # NanoCodec
    codec_num_codebooks: int = 8
    codec_fps: float = 21.5  # NanoCodec frame rate

    # Training
    batch_size: int = 48  # ~18GB at bs=16, scale up for 49GB GPU
    learning_rate: float = 2e-5  # lower than default 2e-4 for fine-tuning
    max_epochs: int = 100
    warmup_steps: int = 500
    grad_clip_val: float = 2.5
    precision: str = "bf16-mixed"  # saves VRAM vs FP32
    lr_scheduler_gamma: float = 0.998  # ExponentialLR decay

    # Context / voice cloning
    context_duration_min: float = 3.0
    context_duration_max: float = 8.0

    # Classifier-free guidance
    cfg_unconditional_prob: float = 0.1

    # Checkpointing
    exp_dir: str = "exp"
    exp_name: str = "magpie_tts_georgian"
    save_top_k: int = 5
    save_last: bool = True
    resume_if_exists: bool = True  # auto-resume from last checkpoint

    # Logging
    wandb_project: str = "georgian-tts"
    wandb_run_name: str = "magpie-tts"

    # Hardware
    devices: int = 1
    num_workers: int = 4
    seed: int = 42

    # Georgian test sentences for audio logging
    eval_sentences: List[str] = field(default_factory=lambda: [
        "გამარჯობა, როგორ ხარ?",
        "საქართველო არის უძველესი ქვეყანა კავკასიაში.",
        "თბილისი საქართველოს დედაქალაქია.",
        "ქართული ენა ერთ ერთი უძველესი ენაა მსოფლიოში.",
        "დღეს კარგი ამინდია და მზე ანათებს.",
    ])
