"""CSM-1B training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CSMConfig:
    # Model
    model_name: str = "sesame/csm-1b"

    # Data
    data_dir: str = "../../data/clean"
    train_manifest: str = "train_manifest.json"
    eval_manifest: str = "eval_manifest.json"
    sample_rate: int = 24000
    max_audio_seconds: float = 10.0
    max_text_tokens: int = 512

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 2  # effective batch = 128
    lr: float = 5e-5
    num_epochs: int = 15
    warmup_steps: int = 10
    weight_decay: float = 0.002
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "n_embed",
    ])

    # Checkpointing & logging
    output_dir: str = "checkpoints"
    run_name: str = "georgian_csm_v1"
    save_strategy: str = "steps"
    save_total_limit: int = 5

    # W&B
    wandb_project: str = "georgian-tts"

    # Hardware
    seed: int = 42

    @property
    def max_audio_samples(self):
        return int(self.max_audio_seconds * self.sample_rate)

    # Georgian sentences for audio logging during evaluation
    eval_sentences: List[str] = field(default_factory=lambda: [
        "გამარჯობა, როგორ ხარ?",
        "საქართველო არის უძველესი ქვეყანა კავკასიაში.",
        "თბილისი საქართველოს დედაქალაქია.",
        "ქართული ენა ერთ ერთი უძველესი ენაა მსოფლიოში.",
        "დღეს კარგი ამინდია და მზე ანათებს.",
    ])
