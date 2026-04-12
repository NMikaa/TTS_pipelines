# Georgian TTS Pipelines & Benchmark

A systematic benchmark of open-source TTS architectures for Georgian (ქართული), a low-resource Caucasian language with no prior TTS benchmarks. Includes a fine-tuned [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) model with zero-shot voice cloning support, and comparisons against MagPIE TTS, F5-TTS, and CSM-1B.

🤗 **Released model**: [`NMikka/omnivoice-finetuned`](https://huggingface.co/NMikka/omnivoice-finetuned)
📊 **Benchmark report**: [`pipelines/omnivoice/BENCHMARK.md`](pipelines/omnivoice/BENCHMARK.md)
🛠️ **Fine-tuning recipe**: [`pipelines/omnivoice/README.md`](pipelines/omnivoice/README.md)

## Highlights

- **First open Georgian TTS benchmark** comparing 4 architectures on the same FLEURS-KA test set
- **5 evaluation metrics** including CER, UTMOS, TTSDS Pitch, TTSDS Speaking Rate, cross-lingual leakage
- **Open fine-tuning recipe** for OmniVoice on Georgian (lr=2e-5, 2 epochs, 0.99 quality threshold)
- **Released model** with robust zero-shot voice cloning across both genders, no language leakage
- **Honest discussion** of metric limitations for low-resource languages — automatic metrics fail to capture phonetic native-likeness

## Results (FLEURS Georgian, 979 samples)

Round-trip evaluation: each model generates audio, [Meta Omnilingual ASR-CTC-1B](https://huggingface.co/facebook/omniASR-CTC-1B) transcribes it, then CER is computed against the normalized reference text.

| Model | Params | FL-CER ↓ | FL-MOS ↑ | TTSDS Pitch ↑ | TTSDS SR ↑ | Voice Cloning |
|-------|--------|----------|----------|---------------|------------|---------------|
| **OmniVoice 099v2_ckpt480** ⭐ | 600M | 1.61% | 2.920 | 82.07 | 75.51 | ✅ Robust both genders |
| OmniVoice pretrained | 600M | 1.64% | 2.749 | **85.64** | 76.34 | ⚠️ Speaker collapse |
| MagPIE TTS Georgian | 357M | 1.80% | **3.140** | 77.95 | **85.09** | ❌ Baked speakers |
| F5-TTS Georgian | 335M | 5.09% | - | - | - | ✅ |
| CSM-1B Georgian | 1B | 10.81% | - | - | - | ❌ |

**Key takeaway**: OmniVoice 099v2_ckpt480 (released model) matches pretrained intelligibility while sounding noticeably more native to Georgian listeners and producing reliable voice cloning across genders. MagPIE wins on naturalness and speaking rate by a wide margin (CTC-based duration modeling). See [`pipelines/omnivoice/BENCHMARK.md`](pipelines/omnivoice/BENCHMARK.md) for full analysis including the metrics-vs-listening gap.

## Why FLEURS, not Common Voice?

We deliberately exclude Common Voice from this benchmark because OmniVoice's pretrained model was trained on Common Voice — using it for evaluation introduces data leakage that artificially favors the pretrained baseline. **FLEURS Georgian is held out from all models tested here**, providing a clean comparison.

## Released Model: OmniVoice 099v2_ckpt480

```python
import torch
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

model = OmniVoice.from_pretrained(
    "NMikka/omnivoice-finetuned",
    device_map="cuda:0",
    dtype=torch.float16,
    load_asr=True,  # auto-transcribe reference audio
)

prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text=None,  # auto-transcribed
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

## Pipelines

| Pipeline | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| [omnivoice](pipelines/omnivoice/) ⭐ | Qwen3-0.6B + HiggsAudioV2 codec, diffusion LM | Released | Best voice cloning robustness, fine-tuning recipe + benchmark |
| [magpie_tts](pipelines/magpie_tts/) | Encoder-decoder + CTC alignment (NeMo) | Done | Best naturalness + speaking rate, 5 baked speakers |
| [f5_tts](pipelines/f5_tts/) | Non-AR flow matching (DiT) | Done | Voice cloning, weaker on Georgian |
| [csm_1b](pipelines/csm_1b/) | Llama + Mimi codec | Done | Multi-speaker, weaker on Georgian |

## Quick Start

### Use the released OmniVoice Georgian model

```bash
pip install omnivoice torchaudio soundfile
# See pipelines/omnivoice/README.md for inference examples
```

### Run the OmniVoice Georgian benchmark

```bash
git clone https://github.com/NMikaa/TTS_pipelines.git
cd TTS_pipelines

# 1. Set up venv
python3.12 -m venv venv_omnivoice
source venv_omnivoice/bin/activate
git clone https://github.com/k2-fsa/OmniVoice.git
pip install -e ./OmniVoice
pip install num2geotext num2words Levenshtein

# 2. Prepare test sets (FLEURS + Common Voice Georgian)
python pipelines/omnivoice/prepare_eval_data.py

# 3. Run evaluation pipeline
python pipelines/omnivoice/evaluate.py --stage generate --checkpoint pretrained --gpu 0
python pipelines/omnivoice/evaluate.py --stage transcribe --gpu 0
python pipelines/omnivoice/evaluate.py --stage metrics
```

See [`pipelines/omnivoice/README.md`](pipelines/omnivoice/README.md) for full reproduction steps.

## Evaluation Metrics

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **CER / WER** | Intelligibility (round-trip ASR) | [Meta omniASR-CTC-1B](https://huggingface.co/facebook/omniASR-CTC-1B) |
| **UTMOS** | Predicted naturalness | [UTMOS22Strong](https://github.com/sarulab-speech/UTMOS22) |
| **Speaker Sim (SIM-o)** | Voice cloning fidelity | [WavLM-ECAPA-TDNN](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) |
| **TTSDS Pitch** | Pitch distribution distance to real speech | [TTSDS](https://github.com/ttsds/ttsds) (WORLD F0) |
| **TTSDS Speaking Rate** | Rhythm distribution distance to real speech | TTSDS (mHuBERT-147 token rate) |
| **Language Leakage** | % of cross-lingual outputs with target script bleed | Custom Unicode regex check |

## Limitations

This benchmark uses only **automatic metrics**. The most important limitation is the absence of **native human MOS evaluation**, which would be the gold standard for capturing perceived Georgian quality. Native Georgian listener feedback (informal) suggests that fine-tuned OmniVoice produces clearly more authentic Georgian than pretrained, but no automatic metric in this benchmark captures this dimension. We discuss this gap in detail in the [benchmark report](pipelines/omnivoice/BENCHMARK.md#why-subjective-listening-tells-a-different-story).

## Citation

```bibtex
@misc{omnivoice-georgian-2026,
  author = {Mikaberidze, Nika},
  title = {OmniVoice Georgian: Fine-tuning and Benchmark for Georgian Text-to-Speech},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/NMikaa/TTS_pipelines}
}
```

## License

- **Code**: Apache 2.0
- **OmniVoice fine-tuned weights**: Apache 2.0 (inherited from base model)
- **MagPIE TTS weights**: NVIDIA Open Model License
- **F5-TTS weights**: CC-BY-NC-4.0
- **Benchmark methodology and report**: CC-BY 4.0

## Acknowledgments

- [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — base TTS model
- [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) — MagPIE TTS framework
- [TTSDS](https://github.com/ttsds/ttsds) — distribution-based evaluation suite
- [Meta Omnilingual ASR](https://huggingface.co/facebook/omniASR-CTC-1B) — Georgian ASR for round-trip evaluation
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — Georgian speech data
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) — multilingual evaluation benchmark
