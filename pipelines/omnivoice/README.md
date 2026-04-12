# OmniVoice Georgian — Fine-Tuning + Benchmark

A fine-tuned [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) variant for Georgian (ქართული) text-to-speech with zero-shot voice cloning, plus a systematic benchmark comparing it against the pretrained baseline and MagPIE TTS Georgian.

🔗 **Released model**: [`NMikka/omnivoice-finetuned`](https://huggingface.co/NMikka/omnivoice-finetuned)
📊 **Benchmark report**: [`BENCHMARK.md`](BENCHMARK.md)

## TL;DR

- **First systematic Georgian TTS benchmark** comparing 3 architectures on FLEURS-KA with 5 metrics (CER, UTMOS, TTSDS Pitch, TTSDS Speaking Rate, cross-lingual leakage)
- **Open fine-tuning recipe** for OmniVoice on Georgian (lr=2e-5, ~480 steps, 0.99 quality threshold)
- **Released model `099v2_ckpt480`** — fine-tuned OmniVoice for Georgian with robust voice cloning, no language leakage, and clearly better subjective Georgian quality vs pretrained per native listener feedback
- **Honest discussion of metric limitations** — automatic metrics fail to capture phonetic native-likeness for low-resource languages

## Key Result

| Model | FL-CER ↓ | FL-MOS ↑ | Voice cloning | Cross-lingual |
|-------|----------|----------|---------------|---------------|
| OmniVoice pretrained | 1.64% | 2.749 | Speaker collapse on male voices | 0% leakage but English degrades |
| **OmniVoice 099v2_ckpt480** ⭐ | 1.61% | 2.920 | Robust both genders | 0% leakage, English preserved |
| MagPIE TTS Georgian | 1.80% | 3.140 | Baked speakers only | N/A (no English) |

See [`BENCHMARK.md`](BENCHMARK.md) for the full report including TTSDS prosody analysis and per-checkpoint comparisons.

## Quick Start

### Inference

```bash
# Install
pip install omnivoice torchaudio soundfile

# Or use this repo's environment
python3.12 -m venv venv_omnivoice
source venv_omnivoice/bin/activate
pip install -e ./OmniVoice
```

```python
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

model = OmniVoice.from_pretrained(
    "NMikka/omnivoice-finetuned",
    device_map="cuda:0",
    dtype=torch.float16,
    load_asr=True,  # for auto-transcription of reference
)

# Voice cloning with reference
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text=None,  # auto-transcribed if None
)

result = model.generate(
    text="გამარჯობა, ეს არის ქართული ტექსტი.",
    language="Georgian",
    voice_clone_prompt=prompt,
    generation_config=OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0),
)

import torchaudio
torchaudio.save("output.wav", result[0].cpu(), 24000)
```

### Gradio UI

```bash
python pipelines/omnivoice/app.py --port 7860 --device cuda:0 --share
```

Includes Voice Clone and Voice Design tabs, with checkpoint switching and instruct support.

## Fine-Tuning Recipe

The full training config that produced ckpt-480 is in [`config/train_config_v2.json`](config/train_config_v2.json):

| Parameter | Value | Note |
|-----------|-------|------|
| `learning_rate` | `2e-5` | **Critical:** lower than the official 5e-5 to preserve pretrained capabilities |
| `warmup_ratio` | `0.01` | Short warmup |
| `steps` | 1500 | But best checkpoint is at step 480 (~2 epochs) |
| `batch_tokens` | 4096 | Per GPU |
| `grad_accum` | 4 | Effective batch ~32k tokens |
| `prompt_ratio_range` | `[0.0, 0.3]` | Default |
| `mask_ratio_range` | `[0.0, 1.0]` | Default |
| `drop_cond_ratio` | `0.1` | Required for CFG at inference |
| `mixed_precision` | `bf16` | |
| `num_speakers` | 29 | In-house Georgian corpus |
| `total_hours` | 345 | At 0.99 text-audio match ratio |

**Critical findings:**
- **lr=2e-5 is the right choice** — the official 5e-5 caused too much catastrophic forgetting of cross-lingual capability
- **Best checkpoint is around 1-2 epochs** (ckpt-240 to ckpt-480). Beyond epoch 4, English regression starts to appear
- **0.99 quality threshold for training data** — at 0.95 threshold, the noise causes Georgian phonemes to leak into English output (3% leakage rate observed on 095_ckpt1606)
- **All cleanly fine-tuned checkpoints have 0% language leakage** in cross-lingual cloning (verified on FLEURS English)

See [`BENCHMARK.md`](BENCHMARK.md) for the full ablation across 11 checkpoints.

## Evaluation Pipeline

```bash
# Stage 1: Generate audio for all test sets and checkpoints
python pipelines/omnivoice/evaluate.py --stage generate --checkpoint all --gpu 0

# Stage 2: Round-trip ASR transcription via omniASR-CTC-1B
python pipelines/omnivoice/evaluate.py --stage transcribe --gpu 0

# Stage 3: Compute CER, WER, and language leakage metrics
python pipelines/omnivoice/evaluate.py --stage metrics

# Stage 4 (optional): TTSDS prosody on FLEURS
python pipelines/omnivoice/run_ttsds_fleurs.py
```

The eval pipeline uses:
- **FLEURS Georgian** test set (979 samples) — held out from all training data
- **FLEURS English** test sets for cross-lingual and English regression checks
- **Common Voice Georgian** for voice cloning intelligibility (note: data leakage caveat — pretrained OmniVoice was trained on Common Voice)

## Text Normalization

Two rule-based normalizers for verbalizing digits/dates/numbers in benchmark text:

- **Georgian** ([`text_normalizer.py`](text_normalizer.py)) — uses `num2geotext` + custom regex. 93% coverage of FLEURS Georgian digit patterns
- **English** ([`text_normalizer_en.py`](text_normalizer_en.py)) — uses `num2words` + custom regex. 100% coverage of FLEURS English digit patterns

Both apply the same normalization to TTS input AND CER reference text, so the comparison is fair.

## Files

```
pipelines/omnivoice/
├── README.md                    # This file
├── BENCHMARK.md                 # Full benchmark report
├── config/
│   ├── train_config_v2.json     # The winning fine-tuning config
│   └── data_config_099.json     # Data manifest paths
├── evaluate.py                  # Multi-stage eval (generate, transcribe, metrics)
├── evaluate_magpie.py           # MagPIE TTS eval (NeMo-based)
├── run_ttsds_fleurs.py          # TTSDS prosody on FLEURS-KA
├── text_normalizer.py           # Georgian number→words normalizer
├── text_normalizer_en.py        # English number→words normalizer
├── app.py                       # Gradio inference UI
└── eval_results/
    ├── comparison.json          # CER/WER metrics per checkpoint
    ├── sim_utmos.json           # WavLM speaker sim + UTMOS metrics
    └── ttsds_fleurs/            # TTSDS prosody results on FLEURS
```

## Limitations

1. **No native human MOS** — the gold standard for TTS quality. We rely on automatic metrics + native listener feedback (anecdotal). A formal MOS study would be the next step.
2. **No phoneme error rate** — would need a Georgian phoneme aligner. The TTS metric we'd actually want.
3. **TTSDS components are partially English-biased** — we excluded MPM/Allosaurus/WeSpeaker components and reported only the most language-agnostic ones (Pitch, mHuBERT-147 SR). Caveats apply.
4. **No voice quality measurement** — what listeners notice (vowel quality, ejective consonants, native phonation) is not captured by any metric in this benchmark.

## Citation

If you use the model, fine-tuning recipe, or benchmark, please cite:

```
@misc{omnivoice-georgian-2026,
  author = {Mikaberidze, Nika},
  title = {OmniVoice Georgian: Fine-tuning and Benchmark for Georgian Text-to-Speech},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/NMikaa/TTS_pipelines}
}
```

## License

- Code: Apache 2.0
- Model weights: Apache 2.0 (inherited from base [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice))
- Benchmark methodology and report: CC-BY 4.0
