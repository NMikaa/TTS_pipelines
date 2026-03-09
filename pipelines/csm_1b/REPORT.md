# CSM-1B Georgian TTS — Evaluation Report

## Model Overview

| | |
|---|---|
| **Base model** | [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) |
| **Architecture** | 1B Llama backbone + 100M decoder + Mimi codec (frozen) |
| **Fine-tuning** | LoRA via Unsloth (rank=64, alpha=64) |
| **Trainable params** | 58M / 1.69B (3.44%) |
| **Target modules** | q/k/v/o_proj, gate/up/down_proj, n_embed |
| **Sample rate** | 24kHz |
| **GPU** | NVIDIA RTX A6000 (48GB) |

## Training

| | |
|---|---|
| **Dataset** | [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) — 21,421 samples, 12 speakers, 35 hours |
| **Epochs** | ~14 |
| **Batch size** | 64 (effective 128 with gradient accumulation=2) |
| **Learning rate** | 5e-5, cosine schedule, 10 warmup steps |
| **Final eval loss** | 5.553 (plateaued from epoch ~10 onward) |
| **Training time** | ~25 hours |
| **Optimizer** | AdamW, weight decay 0.002 |

## In-Domain Evaluation

Evaluated on 120 samples (12 speakers x 10 samples each) from the training data's speaker reference manifest.

| Metric | Value |
|---|---|
| **CER** | **0.0281** |
| WER | 0.1363 |
| MCD | 5.43 dB |
| ECAPA-TDNN | 0.5609 |

## FLEURS Georgian Benchmark

Evaluated on the FLEURS Georgian test set — 979 samples from unseen speakers and unseen text. The model generates all output in a single speaker voice.

### Results

| Metric | Value |
|---|---|
| **CER mean** | **0.1081** |
| CER median | 0.0541 |
| CER std | 0.1541 |
| CER p90 | 0.2507 |
| WER mean | 0.2494 |
| WER median | 0.2000 |

### CER Distribution

| Range | Count | % |
|---|---|---|
| [0.00, 0.05) | 475 | 48.5% |
| [0.05, 0.10) | 170 | 17.4% |
| [0.10, 0.20) | 190 | 19.4% |
| [0.20, 0.50) | 104 | 10.6% |
| [0.50, 1.00] | 40 | 4.1% |

### Interpretation

- **48.5% of samples have near-perfect intelligibility** (CER < 5%), and **65.9% are below 10% CER**.
- The **median CER of 5.4%** shows the model produces intelligible Georgian speech for the majority of inputs.
- The **mean (10.8%) is pulled up by the long tail** — 4.1% of samples have CER > 50%, likely caused by very long or complex sentences where the model hallucinates or truncates.
- **Important context**: This is a zero-shot generalization test. The model was trained on 12 Common Voice speakers and evaluated on entirely different FLEURS speakers and text.

### ASR Baseline

The ASR model used for round-trip evaluation (Meta Omnilingual ASR 7B) has a reported baseline CER of ~1.9% on Georgian. This means a perfect TTS system evaluated with this pipeline would show ~2% CER, not 0%. The model's 5.4% median CER suggests the TTS itself introduces roughly 3-4% additional character errors on typical sentences.

## Evaluation Methodology

| | |
|---|---|
| **ASR model** | Meta Omnilingual ASR 7B (`omniASR_LLM_7B`) — 1.9% CER on Georgian (SOTA) |
| **Speaker similarity** | ECAPA-TDNN via SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`). Language-agnostic — trained on VoxCeleb (multi-language), extracts speaker identity from acoustic features, not linguistic content. |
| **MCD** | pymcd with DTW alignment and silence removal |
| **Generation** | batch_size=8, max_new_tokens=1250 (10s), temperature=default |
| **Benchmark** | FLEURS Georgian test set — 979 samples, professional recordings |

**Why CER over WER**: Georgian is an agglutinative language with long compound words. WER is disproportionately harsh — a single character error in a 20-character word counts as a full word error. CER better captures partial recognition quality.

**Why not UTMOS**: UTMOS is trained on English data only. Absolute scores are not calibrated for Georgian. It was excluded to avoid misleading quality claims.

### Dual Python Environment

The evaluation pipeline requires **two separate Python environments** due to dependency conflicts:

1. **Main environment** (torch 2.10 + torchaudio 2.10) — used for model inference, MCD computation, and ECAPA-TDNN speaker similarity via SpeechBrain. Note: torchaudio 2.10 removed `list_audio_backends()`, so SpeechBrain requires a monkey-patch (`torchaudio.list_audio_backends = lambda: ['soundfile']`).

2. **ASR environment** (`/root/asr_env/`, torch 2.8 + fairseq2 0.6) — used exclusively for Meta Omnilingual ASR 7B transcription. fairseq2 0.6 requires torch 2.8 and is binary-incompatible with torch 2.10 (segfaults). The eval scripts shell out to this environment via `subprocess.run(["/root/asr_env/bin/python3", ...])`.

```bash
# Create ASR environment
python3 -m venv /root/asr_env
/root/asr_env/bin/pip install torch==2.8.0 torchvision==0.23.0 fairseq2==0.6 omnilingual-asr
```

## Summary

CSM-1B fine-tuned with LoRA on 35 hours of cleaned Georgian Common Voice data achieves **5.4% median CER** on the FLEURS Georgian benchmark, with 48.5% of samples near-perfect and 65.9% below 10% CER. The model generalizes to unseen speakers and text, producing intelligible Georgian speech from a 1B parameter model trained on a single GPU in ~25 hours.
