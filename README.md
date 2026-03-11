# TTS Pipelines

The first Georgian TTS benchmark — a comparative study of open-source TTS architectures for Georgian, a low-resource language with no existing TTS benchmarks.

## Goal

Train 4 different TTS models on the **same dataset** (Common Voice Georgian), evaluate them on the **same benchmark** (FLEURS Georgian) with the **same metrics**, and determine which architecture works best for Georgian. The result is an open-source framework that anyone can use to benchmark TTS models on their own language.

## Models

| Pipeline | Architecture | Params | Fine-tuning | License |
|----------|-------------|--------|-------------|---------|
| [F5-TTS](pipelines/f5_tts/) | Non-AR flow matching (DiT) | 335M | Full fine-tune | CC-BY-NC-4.0 |
| [Orpheus](pipelines/orpheus/) | Pure LLM (Llama backbone) | 150M-3B | LoRA (Unsloth) | Apache 2.0 |
| [Qwen3-TTS](pipelines/qwen3_tts/) | Multi-codebook LM | 0.6B | Full SFT | Apache 2.0 |
| [CSM-1B](pipelines/csm_1b/) | Llama + Mimi codec | 1B | LoRA (Unsloth) | Apache 2.0 |

## Evaluation Metrics

All models are evaluated with the same metrics on FLEURS Georgian (979 test samples):

| Metric | What it measures | Tool |
|--------|-----------------|------|
| **CER** | Intelligibility (round-trip TTS -> ASR -> text) | Meta Omnilingual ASR 7B |
| **MCD with DTW** | Spectral closeness to reference audio |
| **Speaker similarity** | Voice identity preservation | ECAPA-TDNN cosine similarity (language-agnostic) |

## Data

- **Training:** ~21k WAV files at 24kHz from Mozilla Common Voice Georgian, stored on S3 (`s3://ttsopensource/`)
- **Evaluation:** FLEURS Georgian via `datasets.load_dataset("google/fleurs", "ka_ge", split="test")`

A fixed train/val/test split is shared across all pipelines. See [shared/data/](shared/data/).

## Quick start

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
cd pipelines/f5_tts
pip install -r requirements.txt
python train.py --data-dir ../../data --run-name my_experiment
python evaluate.py --checkpoint checkpoints/best.pt --data-dir ../../data --output-dir results/
python generate_report.py --results-dir results/ --output report.md
```

Each pipeline is **self-contained** — it has its own README, training script, inference script, evaluation script, and report generator. Pick one, follow the README, and you're done.

## Adding a new pipeline

1. Create a new directory under `pipelines/`
2. Add these files (follow the pattern of existing pipelines):
   - `README.md` — setup, training, inference, evaluation instructions
   - `config.py` — training configuration dataclass
   - `train.py` — training script (imports from `shared.data`)
   - `infer.py` — inference script (single utterance + test set generation)
   - `evaluate.py` — evaluation script (imports from `shared.evaluation`)
   - `generate_report.py` — markdown report from evaluation results
   - `requirements.txt` — pipeline-specific dependencies
3. Use the shared data splits (`shared.data.get_splits`) for fair comparison
4. Use the shared evaluation metrics (`shared.evaluation.run_full_evaluation`)

## Project structure

```
TTS_pipelines/
├── README.md
├── CLAUDE.md                           # Full project context and decisions
├── requirements.txt
├── shared/
│   ├── data/
│   │   ├── download.py                 # Download from S3
│   │   ├── prepare.py                  # Unified manifest format
│   │   └── splits.py                   # Fixed train/val/test splits
│   └── evaluation/
│       ├── evaluate.py                 # Run all metrics
│       ├── intelligibility.py          # CER via Meta Omnilingual ASR
│       ├── naturalness.py              # UTMOS
│       ├── speaker_similarity.py       # ECAPA-TDNN cosine sim
│       └── fad.py                      # Frechet Audio Distance
├── pipelines/
│   ├── f5_tts/
│   ├── orpheus/
│   ├── qwen3_tts/
│   └── csm_1b/
└── report/                             # Comparative report/paper
```

## References

- Mozilla Common Voice: https://commonvoice.mozilla.org/
- Meta Omnilingual ASR: https://huggingface.co/facebook/omniASR_LLM_7B
- SpeechBrain ECAPA-TDNN: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- FLEURS: https://huggingface.co/datasets/google/fleurs
