# TTS Pipelines — Georgian TTS Benchmark

The first Georgian TTS benchmark — a comparative study of 5 open-source TTS architectures for Georgian, a low-resource language with no existing TTS benchmarks.

## Goal

Fine-tune 5 different TTS models on the **same dataset** ([Common Voice Georgian](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned)), evaluate them on the **same benchmark** ([FLEURS Georgian](https://huggingface.co/datasets/google/fleurs), 979 test samples) with the **same metrics**, and determine which architecture works best for Georgian.

Each pipeline is **self-contained** — pick one, follow its README, and you can train + evaluate independently.

## Results

Evaluated on the full [FLEURS Georgian](https://huggingface.co/datasets/google/fleurs) test set (979 samples). Round-trip evaluation: TTS generates audio, [Meta Omnilingual ASR 7B](https://huggingface.co/facebook/omniASR-LLM-7B) transcribes it, then CER/WER is computed against the original text.

| Model | Params | CER | WER | Trained Model | Status |
|-------|--------|-----|-----|---------------|--------|
| [MagPIE TTS](pipelines/magpie_tts/) | 357M | **2.16%** | **7.08%** | [NMikka/Magpie-TTS-Geo-357m](https://huggingface.co/NMikka/Magpie-TTS-Geo-357m) | Done |
| [F5-TTS](pipelines/f5_tts/) | 335M | 5.09% | 18.66% | [NMikka/F5-TTS-Georgian](https://huggingface.co/NMikka/F5-TTS-Georgian) | Done |
| [CSM-1B](pipelines/csm_1b/) | 1B | 10.81% | 24.94% | [NMikka/CSM-1B-Georgian](https://huggingface.co/NMikka/CSM-1B-Georgian) | Done |
| [Orpheus](pipelines/orpheus/) | 3B | — | — | — | Training |
| [Qwen3-TTS](pipelines/qwen3_tts/) | 0.6B | — | — | — | Training |

> The ASR model itself has ~1.9% CER on Georgian, so MagPIE TTS is near the measurement floor — effectively near-perfect intelligibility.

## Models

| Pipeline | Architecture | Fine-tuning | Voice Cloning | License (Weights) |
|----------|-------------|-------------|---------------|-------------------|
| [F5-TTS](pipelines/f5_tts/) | Non-AR flow matching (DiT) | Full fine-tune | Yes (reference audio) | CC-BY-NC-4.0 |
| [Orpheus](pipelines/orpheus/) | Pure LLM (Llama 3.2 + SNAC) | LoRA via Unsloth | Yes (prompt) | Apache 2.0 |
| [Qwen3-TTS](pipelines/qwen3_tts/) | Multi-codebook LM | Full SFT | Yes (reference audio) | Apache 2.0 |
| [CSM-1B](pipelines/csm_1b/) | Llama + Mimi codec | LoRA via Unsloth | No (multi-speaker) | Apache 2.0 |
| [MagPIE TTS](pipelines/magpie_tts/) | Encoder-decoder transformer + CTC alignment | Full SFT via NeMo | Yes (reference audio) | NVIDIA Open Model |

Different models use different fine-tuning methods (LoRA vs full SFT). This is intentional — a model that achieves great results with LoRA on a single GPU is more practical than one requiring full fine-tuning on 8 GPUs. The comparison reflects real-world usability.

## Evaluation Metrics

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **CER** | Intelligibility (round-trip: TTS → ASR → compare text) | [Meta Omnilingual ASR 7B](https://huggingface.co/facebook/omniASR-LLM-7B) |
| **WER** | Word-level intelligibility | Same ASR model |
| **UTMOS** | Naturalness (MOS prediction) | [SpeechMOS](https://github.com/tarepan/SpeechMOS) |
| **FAD** | Distribution quality (generated vs real speech) | VGGish embeddings |
| **Speaker Similarity** | Voice identity preservation (cloning models only) | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) cosine similarity |

**Important caveats:**
- UTMOS is trained on English — use for relative ranking only, not absolute quality
- Whisper is catastrophically bad on Georgian (78-88% WER) — never use it for Georgian ASR
- Speaker similarity only applies to voice-cloning models, not CSM-1B

## Data

- **Training:** [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) — ~71k clips from Mozilla Common Voice Georgian (24kHz)
- **Evaluation:** FLEURS Georgian test set — 979 samples, different speakers and text from training data (no data leakage)
- **Splits:** Deterministic hash-based (90% train, 5% val, 5% test). All pipelines use the same splits via `shared.data.get_splits()`

## Quick Start

### 1. Install shared dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

```bash
python -m shared.data.download --output-dir ./data
```

### 3. Pick a pipeline and follow its README

```bash
cd pipelines/magpie_tts        # or csm_1b, f5_tts, etc.
pip install -r requirements.txt
python train.py --data-dir ../../data/clean
python evaluate.py --checkpoint /path/to/best.ckpt --output-dir results_final
```

Each pipeline has its own README with detailed setup, training, and evaluation instructions.

## Project Structure

```
TTS_pipelines/
├── README.md
├── requirements.txt
├── shared/
│   ├── data/
│   │   ├── download.py                 # Download from S3
│   │   ├── prepare.py                  # Unified manifest format
│   │   └── splits.py                   # Fixed train/val/test splits
│   └── evaluation/
│       ├── evaluate.py                 # Run all metrics
│       ├── intelligibility.py          # CER/WER via Meta Omnilingual ASR
│       └── speaker_similarity.py       # ECAPA-TDNN cosine sim
├── pipelines/
│   ├── f5_tts/                         # F5-TTS (335M, DiT flow matching) ✓
│   ├── orpheus/                        # Orpheus (3B, Llama + SNAC)
│   ├── qwen3_tts/                      # Qwen3-TTS (0.6B, multi-codebook LM)
│   ├── csm_1b/                         # CSM-1B (1B, Llama + Mimi) ✓
│   └── magpie_tts/                     # MagPIE TTS (357M, enc-dec + CTC) ✓
└── report/                             # Comparative report (coming soon)
```

## Adding a New Pipeline

1. Create a new directory under `pipelines/`
2. Add: `README.md`, `config.py`, `train.py`, `infer.py`, `evaluate.py`, `requirements.txt`
3. Use `shared.data.get_splits()` for fair comparison
4. Use `shared.evaluation` for consistent metrics

## References

- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [FLEURS](https://huggingface.co/datasets/google/fleurs)
- [Meta Omnilingual ASR](https://huggingface.co/facebook/omniASR-LLM-7B)
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
