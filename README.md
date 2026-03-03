# TTS Pipelines

Fine-tuning pipelines for Georgian text-to-speech. Two model architectures: **Pocket TTS** (~100M params, flow matching) and **CSM-1B** (~1B params, LoRA fine-tuning).

## Project Structure

```
TTS_pipelines/
├── train.py                          # Pocket TTS entry point
├── train_csm.py                      # CSM-1B entry point
├── requirements.txt                  # Python dependencies
├── pocket-tts/                       # Kyutai Pocket TTS (git submodule)
├── pocket_tts_training/
│   ├── config.py                     # TrainingConfig dataclass
│   ├── trainer.py                    # PocketTTSTrainer (training loop, EMA, checkpointing)
│   ├── dataset.py                    # LatentDataset, BucketSampler, collation
│   └── lsd_loss.py                   # Flow Matching + LSD loss
├── csm_training/
│   ├── config.py                     # CSMTrainingConfig dataclass
│   ├── trainer.py                    # CSMTrainer (HuggingFace Trainer wrapper)
│   ├── dataset.py                    # CSMDataLoader, TTSTrainingDataset
│   └── alignment.py                  # AlignmentProcessor (CTM parsing, gap distribution)
└── scripts/
    ├── precompute_latents.py         # Encode audio -> Mimi latents (run before Pocket TTS training)
    ├── generate_alignments.py        # Run NeMo Forced Aligner -> CTM files
    ├── process_alignments.py         # CTM files -> alignments.json
    ├── monitor_progress.py           # Monitor NFA progress in real-time
    └── renumber_speakers.py          # Renumber speaker IDs by sample count
```

## Requirements

```
# Common
torch>=2.1.0
torchaudio>=2.1.0
tqdm>=4.60
tensorboard>=2.15
boto3>=1.34

# Pocket TTS
sentencepiece>=0.2.0
safetensors>=0.4.0
pydantic>=2.0
beartype>=0.18.0
PyYAML>=6.0

# CSM-1B
transformers>=4.40.0
datasets>=2.18.0
unsloth
pandas>=2.0
soundfile>=0.12
```

```bash
pip install -r requirements.txt
```

---

## Data

All training data is stored on S3 (`s3://ttsopensource/`) and sourced from Common Voice Georgian.

| File | Size | Used By | Description |
|------|------|---------|-------------|
| `audio.rar` | 12.1 GB | Both | ~71k WAV files at 24kHz |
| `latents_cache.rar` | 374 MB | Pocket TTS | Pre-computed Mimi latents (~40k samples) |
| `alignment/voice_actor_manifest.json` | 11 MB | Both | JSONL manifest with `audio_filepath`, `text`, `source` (speaker ID) |
| `alignments.json` | 43 MB | CSM-1B | Word-level timestamps per audio file |
| `quantizer_input_proj_weight.safetensors` | 0.1 MB | Pocket TTS | Mimi input projection weights (512->32 dim) |

### Manifest Format (`voice_actor_manifest.json`)

One JSON object per line:

```json
{"audio_filepath": "/path/to/clip.wav", "text": "გამარჯობა", "source": "1"}
```

### Alignments Format (`alignments.json`)

Maps audio paths to word-level timestamps:

```json
{
  "/path/to/clip.wav": [
    {"word": "გამარჯობა", "start": 0.0, "end": 0.52},
    {"word": "მსოფლიო", "start": 0.52, "end": 1.14}
  ]
}
```

---

## Pipeline 1: Pocket TTS

Fine-tunes Kyutai's FlowLM on pre-computed Mimi latents using LSD (Lagrangian Self-Distillation) + Flow Matching loss.

### Architecture

- **Base model:** Kyutai Pocket TTS FlowLM (variant `b6369a24`)
- **Transformer:** d_model=1024, 6 layers, 16 attention heads
- **Flow network:** flow_dim=512
- **Text tokenizer:** SentencePiece (4000 tokens)
- **Audio codec:** Mimi (24kHz, 12.5 Hz frame rate, 32-dim latents)
- **Only FlowLM is trained** — Mimi encoder/decoder stays frozen

### Data Needed

1. `alignment/voice_actor_manifest.json` — text + audio paths
2. `latents_cache/` — pre-computed latents directory containing:
   - `metadata.json` — list of `{latent_path, text, speaker_id, num_frames}`
   - `000000.pt`, `000001.pt`, ... — latent tensors `[S, 32]`
   - `normalization_stats.pt` — `emb_mean` and `emb_std` for flow-space normalization
3. `quantizer_input_proj_weight.safetensors` — only needed if re-computing latents

### Step 1: Pre-compute Latents (run once)

Encodes all audio through Mimi encoder + input_proj (512->32 dims), saving `.pt` files to disk. This removes Mimi from the training loop entirely.

```bash
python scripts/precompute_latents.py \
    --manifest alignment/voice_actor_manifest.json \
    --output latents_cache \
    --weights quantizer_input_proj_weight.safetensors \
    --max-seconds 15.0
```

If you already have `latents_cache.rar` from S3, just extract it — no need to re-run this.

### Step 2: Train

```bash
python train.py \
    --latents-dir latents_cache \
    --manifest alignment/voice_actor_manifest.json \
    --batch-size 16 \
    --lr 3e-4 \
    --num-epochs 10
```

### Resume from Checkpoint

```bash
python train.py --resume checkpoints/pocket_tts_georgian/epoch_3.pt
```

### Training Flow

```
Pre-computed latents [B, S, 32]
    → Normalize: (latent - emb_mean) / emb_std
    → Teacher forcing: input [BOS, x_0, ..., x_{S-2}], target [x_0, ..., x_{S-1}]
    → input_linear: [S, 32] -> [S, 1024]
    → Prepend text embeddings: [T+S, 1024]
    → Causal transformer -> conditioning [S, d_model]
    → LSD loss (75% flow matching + 25% self-distillation)
    → AdamW + cosine LR + EMA update
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--latents-dir` | `latents_cache` | Pre-computed latents directory |
| `--manifest` | `alignment/voice_actor_manifest.json` | JSONL manifest |
| `--batch-size` | `16` | Batch size per GPU |
| `--lr` | `3e-4` | Peak learning rate |
| `--num-epochs` | `10` | Number of epochs |
| `--max-steps` | `-1` | Max steps (-1 = use epochs) |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--warmup-steps` | `500` | LR warmup steps |
| `--gradient-clip` | `1.0` | Max gradient norm |
| `--fm-ratio` | `0.75` | Flow matching ratio (vs LSD) |
| `--hbm` | `4` | Head batch multiplier (noise samples per conditioning) |
| `--num-workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--output-dir` | `checkpoints/pocket_tts_georgian` | Checkpoint output dir |
| `--resume` | `None` | Checkpoint path to resume from |
| `--no-amp` | `False` | Disable mixed precision |

### Hyperparameters (config.py)

| Parameter | Value |
|-----------|-------|
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| Min LR | 1e-5 |
| EMA decay | 0.9999 |
| EMA start step | 1000 |
| Mixed precision | float16 / bfloat16 |
| Log every | 50 steps |
| Max audio | 15 seconds |

### Checkpoints

Saved every epoch to `checkpoints/pocket_tts_georgian/`:

```
epoch_0.pt
epoch_1.pt
...
final.pt
```

Each checkpoint contains: `flow_lm_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `ema_state_dict`, `scaler_state_dict`, `global_step`, `epoch`, `config`.

---

## Pipeline 2: CSM-1B

Fine-tunes Sesame's CSM-1B using LoRA (via Unsloth) with a 2-word look-ahead streaming training scheme and word-level alignment.

### Architecture

- **Base model:** `sesame/csm-1b` (~1B params)
- **Fine-tuning:** LoRA r=128, alpha=32 (targets q/k/v/o/gate/up/down projections)
- **Trainer:** HuggingFace Trainer + Unsloth FastModel
- **Audio:** 24kHz, frame size 1920 samples (80ms)
- **Codec model is frozen** — only the language model is trained

### Data Needed

1. `alignment/voice_actor_manifest.json` — text + audio paths
2. `alignments.json` — word-level timestamps (from NeMo Forced Aligner)
3. Audio WAV files — referenced by the manifest

### Training Scheme

Uses a **2-word look-ahead** approach:

1. **Context turn** (masked from loss): shifted audio from a nearby speaker, provides voice conditioning
2. **First message**: up to 3 words of text + first word's audio
3. **Subsequent messages**: one look-ahead word of text + that word's audio

The context turn loss is masked so the model only learns to *generate* audio, not reconstruct context.

### Step 1: Generate Alignments (if needed)

If you don't have `alignments.json`, generate it from scratch:

```bash
# 1. Run NeMo Forced Aligner (produces CTM files, ~4-5 hours on GPU)
python scripts/generate_alignments.py

# 2. Monitor progress (in a separate terminal)
python scripts/monitor_progress.py

# 3. Process CTM files into alignments.json
python scripts/process_alignments.py \
    --voice_actor_manifest alignment/voice_actor_manifest.json \
    --alignment_base_path alignment/ \
    --output_json alignments.json
```

### Step 2: Train

```bash
python train_csm.py \
    --voice-actor-path alignment/voice_actor_manifest.json \
    --alignment-json alignments.json \
    --run-name georgian_csm_v1
```

### Full Fine-tuning (more VRAM)

```bash
python train_csm.py \
    --full-finetuning \
    --voice-actor-path alignment/voice_actor_manifest.json \
    --alignment-json alignments.json \
    --run-name georgian_csm_full
```

### Resume from Checkpoint

```bash
python train_csm.py \
    --base-model checkpoints/csm_georgian/georgian_csm_v1/checkpoints/checkpoint-XXX \
    --voice-actor-path alignment/voice_actor_manifest.json \
    --alignment-json alignments.json \
    --run-name georgian_csm_v2
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | `sesame/csm-1b` | Base model ID |
| `--full-finetuning` | `False` | Full fine-tuning instead of LoRA |
| `--base-model` | `None` | Path to resume from a fine-tuned checkpoint |
| `--lora-r` | `128` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--dataset-path` | `None` | HF shards path prefix (alternative to manifest) |
| `--voice-actor-path` | `None` | JSONL manifest path |
| `--alignment-json` | `alignments.json` | Word-level alignment file |
| `--speakers` | `[]` | Filter to specific speaker IDs |
| `--test-size` | `0.05` | Validation split fraction |
| `--batch-size` | `64` | Batch size per GPU |
| `--lr` | `1e-4` | Learning rate |
| `--num-epochs` | `6` | Number of epochs |
| `--grad-accum` | `2` | Gradient accumulation steps |
| `--warmup-steps` | `5` | LR warmup steps |
| `--weight-decay` | `0.01` | Weight decay |
| `--num-workers` | `12` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--run-name` | *required* | Name for this training run |
| `--output-dir` | `checkpoints/csm_georgian` | Checkpoint output dir |
| `--report-to` | `tensorboard` | Logging backend (`tensorboard` or `wandb`) |
| `--push-to-hub` | `False` | Push final model to HuggingFace Hub |
| `--hub-model-id` | `None` | HuggingFace repo ID |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Yes | HuggingFace authentication (to download `sesame/csm-1b`) |
| `WANDB_API_KEY` | Only if `--report-to wandb` | Weights & Biases API key |

### Hyperparameters (config.py)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW 8-bit |
| LR scheduler | Cosine |
| Gradient checkpointing | Enabled |
| Max audio length | 120 seconds |
| Max text length | 4096 tokens |
| Frame size | 1920 samples (80ms) |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Eval/save frequency | 4x per epoch |
| Logging frequency | 20x per epoch |

---

## Utility Scripts

### `scripts/precompute_latents.py`
Encodes all audio through Mimi encoder + input_proj once. Saves `[S, 32]` latent tensors and `normalization_stats.pt`. Required before Pocket TTS training.

### `scripts/generate_alignments.py`
Creates a NeMo manifest from `data.parquet` and runs NeMo Forced Aligner to produce CTM (word-level timing) files. Requires a Georgian ASR model (`.nemo` file) and NeMo toolkit.

### `scripts/process_alignments.py`
Reads CTM files from NFA output, applies weighted gap distribution and short word merging (< 80ms), and writes `alignments.json`. Required before CSM-1B training.

### `scripts/monitor_progress.py`
Real-time progress monitor for NFA alignment generation. Run in a separate terminal. Shows files/sec, ETA, and stall detection.

### `scripts/renumber_speakers.py`
Renumbers speaker IDs in `voice_actor_manifest.json` so the speaker with the most samples becomes `1`, next becomes `2`, etc. Creates `.backup` files before modifying.

---

## Quick Start

### Pocket TTS (faster, smaller model)

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from S3
aws s3 cp s3://ttsopensource/latents_cache.rar .
aws s3 cp s3://ttsopensource/alignment/ alignment/ --recursive

# Extract latents
unrar x latents_cache.rar

# Train
python train.py --batch-size 16 --num-epochs 10 --lr 3e-4
```

### CSM-1B (larger, higher quality)

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from S3
aws s3 cp s3://ttsopensource/audio.rar .
aws s3 cp s3://ttsopensource/alignment/ alignment/ --recursive
aws s3 cp s3://ttsopensource/alignments.json .

# Extract audio
unrar x audio.rar

# Set HuggingFace token
export HF_TOKEN=your_token_here

# Train
python train_csm.py \
    --voice-actor-path alignment/voice_actor_manifest.json \
    --alignment-json alignments.json \
    --run-name georgian_csm_v1
```

---

## Monitoring

Both pipelines log to TensorBoard:

```bash
# Pocket TTS
tensorboard --logdir checkpoints/pocket_tts_georgian/runs/

# CSM-1B
tensorboard --logdir checkpoints/csm_georgian/<run-name>/checkpoints/logs/
```

For WandB logging with CSM-1B, pass `--report-to wandb` and set `WANDB_API_KEY`.
