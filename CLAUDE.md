# CLAUDE.md — Project Guide for AI Assistants

## What is this project?

The first Georgian TTS benchmark — comparing open-source TTS architectures fine-tuned on [Common Voice Georgian](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) (12 speakers, 35 hours, 21k clips).

Each pipeline under `pipelines/` is self-contained: train, infer, evaluate. All share the same data (`shared/data/`) and evaluation (`shared/evaluation/`).

## Project Structure

```
TTS_pipelines/
├── shared/                    # Shared data download, splits, evaluation
│   ├── data/download.py       # Downloads from HuggingFace (NMikka/Common-Voice-Geo-Cleaned)
│   ├── data/splits.py         # Deterministic hash-based train/val/test splits
│   └── evaluation/            # CER/WER via Meta Omnilingual ASR, speaker similarity
├── pipelines/
│   ├── magpie_tts/            # NVIDIA MagPIE TTS 357M — best results (2.16% CER)
│   ├── f5_tts/                # F5-TTS 335M — flow matching DiT
│   ├── csm_1b/                # CSM-1B — Llama + Mimi codec
│   ├── orpheus/               # Orpheus TTS (scaffold)
│   └── qwen3_tts/             # Qwen3 TTS (scaffold)
├── NeMo/                      # NVIDIA NeMo (git submodule/clone, not committed)
└── data/                      # Downloaded data (gitignored)
```

## MagPIE TTS Pipeline (primary focus)

### Architecture
```
Text → ByT5-small → 6-layer Encoder → CTC Alignment → 12-layer Decoder → NanoCodec → Waveform (22kHz)
```

357M params. `model_type: "decoder_ce"`. NanoCodec: 8 codebooks, 21.5 fps.

### Key Files (pipelines/magpie_tts/)

| File | Purpose |
|------|---------|
| `config.py` | `MagPIEConfig` dataclass with all hyperparameters |
| `train.py` | 4-stage pipeline: resample → codec tokens → manifests → NeMo training |
| `train_cloning.py` | Voice cloning training: strip baked embeddings, context-pair data, freeze approach |
| `generate.py` | Inference via `generate_speech` (chunked, handles voice cloning via `--ref-audio`) |
| `stream.py` | Streaming inference with cross-fade at chunk boundaries |
| `test_cloning.py` | Quick voice cloning test script (loads checkpoint, generates from reference audio) |
| `evaluate.py` | FLEURS evaluation (CER/WER via round-trip ASR) |
| `conf/magpietts_georgian.yaml` | Hydra config — extends base with all pretrained tokenizers |
| `conf/pretrained_dicts/` | Phoneme dictionaries extracted from pretrained .nemo |

### Baked Speakers vs Voice Cloning

The pretrained model has **5 baked speaker embeddings** (`nn.Embedding(5, T*D)`) that bypass the context encoder entirely. When `has_baked_context_embedding` is True, the model uses the lookup table; when False, it uses the **context encoder** (1-layer transformer, 18.1M params) to process reference audio.

**Voice cloning training** (`train_cloning.py`):
1. `prepare_cloning_checkpoint()` — loads pretrained + Georgian weights, strips baked embeddings
2. `create_cloning_manifests()` — pairs each sample with a different clip from the same speaker as context
3. `train_cloning()` — launches NeMo with `+freeze_for_cloning=true` (freezes all except context_encoder + decoder cross-attention, ~22.8M trainable / 289M frozen)

### Critical Tokenizer Bug

MagPIE uses a combined vocabulary from 10+ tokenizers. The offset of `text_ce_tokenizer` (Georgian) must match the pretrained model exactly. If tokenizer order differs in the Hydra config, offsets shift and inference produces garbage even though training looks fine (teacher-forced validation masks the bug).

**Fix**: Always load tokenizer from the training checkpoint at inference time:
```python
training_cfg = ckpt['hyper_parameters']['cfg']
model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)
```

### NeMo Local Modifications

We patched two files in the local NeMo clone:

1. **`nemo/utils/callbacks/nemo_model_checkpoint.py:240`** — added `weights_only=False` to `torch.load()` (PyTorch 2.6 compatibility)
2. **`nemo/core/classes/modelPT.py:1414,1428`** — same `weights_only=False` fix
3. **`examples/tts/magpietts.py:75-92`** — added `freeze_for_cloning` support (freezes all params except context_encoder + decoder cross-attention patterns)

### Voice Cloning Inference

```python
# Key: strip baked embeddings BEFORE loading cloning checkpoint
model.baked_context_embedding = None
model._baked_embedding_T = None
model._baked_embedding_D = None
model.baked_context_embedding_len = None

# Then load checkpoint
model.load_state_dict(ckpt['state_dict'], strict=False)
```

Pass reference audio in the batch:
```python
batch = {
    'text': tokens,
    'text_lens': token_lens,
    'context_audio': ref_waveform,       # (1, time) at 22050 Hz
    'context_audio_lens': ref_len,
    'context_sample_rate': 22050,
    'context_text_tokens': torch.zeros(1, 1, dtype=torch.long),
    'context_text_tokens_lens': torch.zeros(1, dtype=torch.long),
    'has_text_context': torch.tensor([False]),
}
```

## Data

- **Source**: `NMikka/Common-Voice-Geo-Cleaned` on HuggingFace
- **12 speakers**, dominant speaker 1 (8.79h, 5673 samples), smallest speaker 14 (0.66h, 321 samples)
- **Manifest fields**: `audio_path`, `text`, `speaker_id`, `duration`, `id`
- **NeMo manifest fields**: `audio_filepath`, `text`, `duration`, `speaker`, `target_audio_codes_path`
- Download: `python -m shared.data.download --output-dir ./data`
- Data prep (resample + codec): `python train.py --data-dir ../../data/clean --prepare-only`

## Environment

- **Python**: Use a virtualenv with NeMo, torch, etc installed
- **GPU**: CUDA GPU recommended (48GB VRAM for training at batch_size=32)
- **NeMo**: Local clone at `./NeMo/`, installed in editable mode (`pip install -e ./NeMo`)
- **Key packages**: torch, torchaudio, nemo_toolkit, transformers, datasets, wandb, soundfile

## Common Commands

```bash
# Download data
python -m shared.data.download --output-dir ./data

# Train (baked speaker fine-tuning)
cd pipelines/magpie_tts
python train.py --data-dir ../../data/clean

# Train voice cloning (frozen approach)
python train_cloning.py --data-dir ../../data/clean --holdout-speakers 14 10 --lr 1e-4 --epochs 150

# Generate speech (baked speaker)
python generate.py --text "გამარჯობა" --speakers 1

# Generate speech (voice cloning)
python generate.py --text "გამარჯობა" --ref-audio reference.wav

# Test cloning from checkpoint
python test_cloning.py --checkpoint /path/to/ckpt --ref-audio ref.wav --text "გამარჯობა"

# Evaluate on FLEURS
python evaluate.py --checkpoint /path/to/best.ckpt --speaker 1

# Streaming inference
python stream.py --text "გრძელი ტექსტი..." --speaker 1
```

## Build & Training Notes

- Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before training to prevent thread exhaustion in dataloader workers
- Use `num_workers=2` for train, `num_workers=1` for val to stay within system limits
- Voice cloning training uses ~45GB VRAM at batch_size=32 (mostly activation memory)
- Wandb validation audio is teacher-forced (not autoregressive) — it's an upper bound on quality, not realistic inference
