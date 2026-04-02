# Experiment A: Setup and Execution Guide

**Status:** Data preparation complete, ready for codec extraction and training

**Timeline:** 
- Codec extraction: 2-3 hours (GPU-intensive)
- Training: 2-3 weeks on 2 GPUs (100 epochs, early stopping at ~80)
- Evaluation: 2-3 days

---

## Prerequisites

### 1. Environment Setup

You need a Python environment with NeMo and PyTorch. Use the MagPIE requirements:

```bash
# Create virtualenv
python3 -m venv ~/tts_env
source ~/tts_env/bin/activate

# Install shared dependencies
pip install -r requirements.txt

# Install MagPIE-specific dependencies
pip install -r pipelines/magpie_tts/requirements.txt

# Install NeMo in editable mode (if not already done)
pip install -e ./NeMo
```

### 2. GPU Requirements

- **Codec extraction:** 1 GPU with 24GB+ VRAM (NanoCodec inference)
- **Training:** 2 GPUs with 90GB+ total VRAM (batch_size=48 per GPU, bf16-mixed precision)
- Environment variables:
  ```bash
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ```

---

## Data Preparation Status

### ✅ Completed

- **Punctuated filtering:** Created `prepare_experiment_a_data.py`
  - Input: 639,943 concatenated segments (train + eval from saba_clean)
  - Output: 182,406 training + 3,770 validation + 145 hold-out samples
  - Punctuated segments only: ~394 hours from 24 speakers
  - Hold-out speaker: ზაალ სამადაშვილი (0.25h, 145 samples) for zero-shot testing

### ⏳ Next: Codec Token Extraction

**Files prepared:**
- `prepare_experiment_a_codec.py` — Codec extraction script
- `conf/magpietts_experiment_a.yaml` — Hydra configuration
- `run_experiment_a.py` — Training launch script

**Data locations:**
- Input manifests: `data/saba_experiment_a/`
  - `train_manifest.json` (182,406 samples)
  - `val_manifest.json` (3,770 samples)
  - `holdout_manifest.json` (145 samples)
- Output (to be created):
  - `audio_22khz/` — Resampled audio at 22.05kHz
  - `codec_codes/` — Pre-computed NanoCodec tokens
  - `train_manifest_nemo.json` — NeMo training manifest
  - `val_manifest_nemo.json` — NeMo validation manifest

---

## Step-by-Step Execution

### Step 1: Extract Codec Tokens (2-3 hours)

```bash
cd pipelines/magpie_tts
python prepare_experiment_a_codec.py
```

**What it does:**
1. Resamples audio from original rate → 22.05kHz (NanoCodec requirement)
2. Loads NanoCodec model and extracts tokens for each audio file
3. Converts manifests to NeMo format with codec token paths

**Expected output:**
```
Step 1: Resampling ... audio files
  Done: ~186k resampled, 0 already existed

Step 2: Loading NanoCodec ...
  Pre-computing codec tokens for ~186k files
  Done: ~186k encoded, 0 already existed

Step 3: Converting manifests to NeMo format
  Wrote 182406 entries to train_manifest_nemo.json
  Wrote 3770 entries to val_manifest_nemo.json
```

### Step 2: Verify Codec Extraction

```bash
# Check resampled audio
ls data/saba_experiment_a/audio_22khz/ | wc -l
# Should show ~186k files

# Check codec tokens
ls data/saba_experiment_a/codec_codes/ | wc -l
# Should show ~186k files

# Check NeMo manifests
wc -l data/saba_experiment_a/*nemo.json
# Should show 182406 train + 3770 val = 186176 total
```

### Step 3: Launch Training (2-3 weeks)

```bash
cd pipelines/magpie_tts
python run_experiment_a.py
```

**What it does:**
1. Verifies data preparation is complete
2. Launches DDP training on 2 GPUs with Hydra overrides
3. Monitors validation loss and applies early stopping (patience=5)
4. Saves top 5 checkpoints + last checkpoint

**Monitoring:**
```bash
# TensorBoard
tensorboard --logdir=pipelines/magpie_tts/exp/

# WandB
# Check project: georgian-tts, run: magpie-experiment-a
```

**Output:** 
- Checkpoints: `exp/experiment_a_punctuated_joint_training/version_*/checkpoints/`
- Logs: `exp/experiment_a_punctuated_joint_training/version_*/`

### Step 4: Evaluate Checkpoints (2-3 days)

After training completes, evaluate all 5 best checkpoints:

```bash
cd pipelines/magpie_tts

# Test in-distribution performance (40 training speakers)
python evaluate.py \
  --checkpoint exp/experiment_a_punctuated_joint_training/*/checkpoints/epoch=*.ckpt \
  --test-set fleurs  # FLEURS Georgian test set

# Test zero-shot voice cloning (held-out speaker)
python test_zero_shot_cloning.py \
  --checkpoint exp/experiment_a_punctuated_joint_training/*/checkpoints/epoch=*.ckpt \
  --ref-audio data/saba_experiment_a/holdout_manifest.json
```

---

## Configuration Details

### Experiment A Hyperparameters

**Data:**
- Training: 182,406 samples (~393.4h) from 24 speakers
- Validation: 3,770 samples (~8.1h) from same 24 speakers
- Hold-out: 145 samples (0.25h) from ზაალ სამადაშვილი
- Split: Deterministic hash-based (98/2) on segment ID

**Model:**
- Architecture: 357M parameters, ByT5-small encoder, 12-layer decoder
- Speaker conditioning: Baked embeddings (41 speakers) + context encoder (joint training)
- Pretrained: nvidia/magpie_tts_multilingual_357m

**Training:**
- Batch size: 48 per GPU × 2 = 96 effective
- Learning rate: 2e-5 (fine-tuning, 10x lower than pretraining)
- Max epochs: 100 (expect ~80 with early stopping)
- Optimizer: AdamW (β1=0.9, β2=0.999)
- LR scheduler: Exponential decay (γ=0.998/step)
- Warmup: 500 steps
- Gradient clipping: 2.5
- Precision: bf16-mixed (saves ~50% VRAM)

**Early Stopping:**
- Monitor: Validation loss
- Patience: 5 epochs (stop if no improvement for 5 checkpoints)
- Save strategy: Top 5 by validation loss + last checkpoint

---

## Success Criteria

### In-Distribution Performance (24 Training Speakers)

✅ **Target:** CER < 3% on FLEURS test set
- Baseline (35h, 12 speakers): 2.16% CER
- Expected (393h, 24 speakers): 1.5-2.5% CER
- Minimum acceptable: < 5% CER

### Zero-Shot Voice Cloning (Hold-Out Speaker)

✅ **Target:** CER < 6% on FLEURS test set using context encoder
- Interpretation: If this succeeds, context encoder learned general speaker representation across 24 speakers
- Minimum acceptable: Context encoder produces usable embeddings (subjective listening test)

### Training Stability

✅ All baked embeddings and context encoder receive non-zero gradients
✅ No NaN/Inf losses during training
✅ Validation loss curves smoothly
✅ DDP synchronization works properly

---

## Decision Tree (After Experiment A Completes)

### If CER < 3% AND zero-shot < 6%:
→ ✅ Hypothesis validated: High-quality data + joint training works
→ Proceed to **Phase 2: DPO/GRPO preference optimization** (expected 40% CER reduction)

### If CER 3-5% AND zero-shot works:
→ ⚠️ Acceptable results, but headroom for improvement
→ Evaluate **Experiment B** (all 1,384h unpunctuated data) vs Phase 2 trade-off

### If CER > 5% OR zero-shot fails:
→ ❌ Debug before proceeding
→ Check: Tokenizer offsets, DDP synchronization, learning rate scale, codec paths

---

## Troubleshooting

### ModuleNotFoundError: torch / nemo

**Solution:** Activate your virtualenv with NeMo installed
```bash
source ~/tts_env/bin/activate
pip install -r pipelines/magpie_tts/requirements.txt
pip install -e ./NeMo
```

### CUDA out of memory during codec extraction

**Solution:** The codec extraction is VRAM-intensive. If you have < 24GB:
- Use a smaller GPU batch size (but it's not configurable in the current script)
- Run on a machine with more VRAM (A100, H100, multi-GPU)

### CUDA out of memory during training

**Solution:** Reduce batch size (currently 48 per GPU):
```bash
python run_experiment_a.py trainer.batch_size=32
# or 24 for very tight VRAM
```

### Training stuck / slow progress

**Solution:** Check GPU utilization and data loading:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check dataloader workers (target: 100% GPU util)
# If low, increase num_workers in config (currently 2)
```

---

## Files Created

```
pipelines/magpie_tts/
├── prepare_experiment_a_data.py      # ✅ Data filtering script
├── prepare_experiment_a_codec.py      # ⏳ Codec extraction (requires env)
├── run_experiment_a.py                # ⏳ Training launcher
├── conf/
│   └── magpietts_experiment_a.yaml    # ⏳ Hydra config
└── evaluate_experiment_a.py           # TODO: Evaluation script

data/saba_experiment_a/
├── train_manifest.json                # ✅ Input (182k samples)
├── val_manifest.json                  # ✅ Input (3.7k samples)
├── holdout_manifest.json              # ✅ Input (145 samples)
├── audio_22khz/                       # ⏳ To be created (resampled audio)
├── codec_codes/                       # ⏳ To be created (NanoCodec tokens)
├── train_manifest_nemo.json           # ⏳ To be created (NeMo format)
└── val_manifest_nemo.json             # ⏳ To be created (NeMo format)
```

---

## Next Steps

1. **Activate environment:** `source ~/tts_env/bin/activate`
2. **Run codec extraction:** `python pipelines/magpie_tts/prepare_experiment_a_codec.py`
3. **Wait 2-3 hours** for codec extraction to complete
4. **Launch training:** `python pipelines/magpie_tts/run_experiment_a.py`
5. **Wait 2-3 weeks** for training to complete
6. **Evaluate results** against success criteria
