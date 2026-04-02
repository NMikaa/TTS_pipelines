# Experiment A: Logging & Monitoring Setup

## Disk Space Requirements

### Estimate: ~65-70 GB additional space needed

**Breakdown:**

| Component | Size | Notes |
|-----------|------|-------|
| Original saba_clean audio | ~49 GB | Already extracted, 24kHz |
| Resampled audio (22.05kHz) | ~62 GB | Mono, 16-bit, ~394h total |
| Codec tokens | ~1-2 GB | 8 codebooks × 30.5M frames |
| NeMo manifests | ~50 MB | JSON files, negligible |
| **Total additional** | **~63-64 GB** | |
| **Available space** | **~200 GB** | ✓ **Sufficient** |
| **Safe margin** | **~130-135 GB** | Comfortable headroom |

**Warnings:**
- Do NOT delete original audio during resampling — codec extraction needs both versions
- After codec extraction completes, you can safely delete original saba_data parquets (already done)
- Resampled audio can be deleted after successful training starts, but keep for reproducibility

---

## W&B (Weights & Biases) Setup

### Step 1: Create Account

1. Go to https://wandb.ai/
2. Sign up (free account available)
3. Navigate to: https://wandb.ai/authorize
4. Copy your **API key**

### Step 2: Login

```bash
cd pipelines/magpie_tts

# Run W&B login script
bash setup_wandb.sh

# OR manually login
wandb login YOUR_API_KEY
```

### Step 3: Verify Setup

```bash
wandb status
# Should show: You are logged in to wandb
```

---

## Logging Structure

### Log Files Location

Both codec extraction and training create timestamped log files in:
```
data/saba_experiment_a/logs/
├── codec_preparation_20240402_150530.log    # Codec extraction
└── training_20240402_200000.log             # Training runs
```

### Log File Format

Each log entry includes timestamp, level, and message:
```
14:32:15 | INFO     | Step 1: Resampling 186234 audio files to 22050 Hz...
14:35:42 | INFO     | Done: 186234 resampled, 0 already existed
14:35:43 | INFO     | Step 2: Loading NanoCodec from nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps...
14:35:50 | INFO     | Loading: ████████████████████ [100%]
```

---

## Monitoring During Execution

### Real-Time Console Output

Both scripts log to console AND file simultaneously:
```bash
# Console shows progress with timestamps
# File captures complete history
tail -f data/saba_experiment_a/logs/codec_preparation_*.log
```

### Command for Watching Logs

```bash
# Watch codec extraction (replace timestamp)
tail -f data/saba_experiment_a/logs/codec_preparation_20240402_150530.log

# Watch training (after it starts)
tail -f data/saba_experiment_a/logs/training_20240402_200000.log
```

---

## W&B Dashboard Monitoring

### During Training

**Project:** georgian-tts
**Run name:** magpie-experiment-a

**Navigate to:** https://wandb.ai/your-username/georgian-tts/runs/magpie-experiment-a

### Tracked Metrics

**Training:**
- `train_loss` — Training loss (should decrease)
- `train_lr` — Learning rate (exponential decay)
- `train_epoch` — Current epoch

**Validation:**
- `val_loss` — Validation loss (monitor for early stopping)
- `val_cer` — Character Error Rate (if computed during validation)

**System:**
- GPU memory usage
- Training speed (samples/sec)
- Epoch duration

### Example Dashboard View

```
Epoch 1-10:   Loss trends down
Epoch 20-30:  Validation loss plateaus
Epoch 50-70:  Early stopping patience decreases
Epoch 75:     Early stopping triggers (patience=5 reached)
              Training stops, saves final checkpoint
```

---

## Quick Reference: Monitoring Commands

### Check codec preparation progress

```bash
# Watch real-time log
tail -f data/saba_experiment_a/logs/codec_preparation_*.log

# Count extracted codec files (should reach ~186k)
watch -n 5 'ls data/saba_experiment_a/codec_codes/ | wc -l'
```

### Check training progress

```bash
# Watch training log
tail -f data/saba_experiment_a/logs/training_*.log

# Check GPU usage (in another terminal)
nvidia-smi -l 1  # Updates every second

# Check checkpoint saving
watch -n 10 'ls -lah exp/experiment_a_punctuated_joint_training/*/checkpoints/'
```

### W&B Monitoring

```bash
# Open dashboard in browser (automatically)
wandb launch https://wandb.ai/your-username/georgian-tts

# Or view in terminal
w wandb sync exp/experiment_a_punctuated_joint_training/
```

---

## Setup & Run Commands

### Full Execution Sequence

```bash
# 1. Navigate to MagPIE directory
cd pipelines/magpie_tts

# 2. Setup W&B (one-time)
bash setup_wandb.sh
# Follow prompts to enter API key

# 3. Extract codec tokens (2-3 hours)
# Logs: data/saba_experiment_a/logs/codec_preparation_*.log
python prepare_experiment_a_codec.py

# 4. Verify codec extraction
ls data/saba_experiment_a/codec_codes/ | wc -l  # Should be ~186k

# 5. Launch training (2-3 weeks)
# Logs: data/saba_experiment_a/logs/training_*.log
# W&B: https://wandb.ai/your-username/georgian-tts
python run_experiment_a.py

# 6. Monitor (in another terminal)
tail -f ../../../data/saba_experiment_a/logs/training_*.log
# or
tensorboard --logdir=exp/
```

---

## Expected Output Examples

### Codec Extraction Log

```
15:30:00 | INFO     | ========================================================================
15:30:00 | INFO     | EXPERIMENT A: DATA PREPARATION
15:30:00 | INFO     | ========================================================================
15:30:00 | INFO     | Log file: /root/TTS_pipelines/data/saba_experiment_a/logs/codec_prep...
15:30:00 | INFO     | Data directory: ../../data/saba_experiment_a
15:30:00 | INFO     | Train manifest: train_manifest.json
15:30:00 | INFO     | 
15:30:00 | INFO     | Timeline: ~2-3 hours total
15:30:00 | INFO     |   - Resampling: ~10-15 minutes
15:30:00 | INFO     |   - Codec extraction: ~1-2 hours (GPU-intensive)
15:30:00 | INFO     |   - Manifest conversion: ~5-10 minutes
15:30:00 | INFO     | 
15:30:01 | INFO     | Step 1: Resampling 186234 audio files to 22050 Hz...
15:30:01 | INFO     |   Resampling: ████████████████████ [100%] 186234/186234
15:35:42 | INFO     |   Done: 186234 resampled, 0 already existed
15:35:43 | INFO     | Step 2: Loading NanoCodec...
15:35:50 | INFO     |   Pre-computing codec tokens for 186234 files
15:35:50 | INFO     |   Encoding: ████████████████████ [100%] 186234/186234
17:38:15 | INFO     | Step 3: Converting manifests to NeMo format
17:38:25 | INFO     |   Wrote 182406 entries to train_manifest_nemo.json
17:38:25 | INFO     |   Wrote 3770 entries to val_manifest_nemo.json
17:38:25 | INFO     | 
17:38:25 | INFO     | ✓ DATA PREPARATION COMPLETE
```

### Training Log (excerpt)

```
21:00:00 | INFO     | ========================================================================
21:00:00 | INFO     | EXPERIMENT A: TRAINING
21:00:00 | INFO     | ========================================================================
21:00:00 | INFO     | Log file: /root/TTS_pipelines/data/saba_experiment_a/logs/training_...
21:00:00 | INFO     | 
21:00:00 | INFO     | Training configuration:
21:00:00 | INFO     |   - Model: nvidia/magpie_tts_multilingual_357m (357M params)
21:00:00 | INFO     |   - Data: 182k samples (~393h) from 24 speakers
21:00:00 | INFO     |   - Epochs: 100 (target ~80 with early stopping)
21:00:00 | INFO     |   - Batch size: 48 per GPU × 2 = 96 effective
21:00:00 | INFO     |   - Learning rate: 2e-5 (fine-tuning rate)
21:00:00 | INFO     |   - Precision: bf16-mixed (saves ~50% VRAM)
21:00:00 | INFO     |   - Early stopping: patience=5
21:00:00 | INFO     |   - Distributed: DDP on 2 GPUs
21:00:00 | INFO     | 
21:00:00 | INFO     | W&B Integration:
21:00:00 | INFO     |   - Project: georgian-tts
21:00:00 | INFO     |   - Run: magpie-experiment-a
21:00:00 | INFO     |   - Monitor at: https://wandb.ai/
21:00:00 | INFO     | Launching training...
21:00:15 | INFO     | [NeMo] Rank 0 initialized DistributedDataParallel with strategy: ddp
21:00:20 | INFO     | [NeMo] Epoch 1/100 | Train Loss: 4.234 | Val Loss: 3.891
21:00:45 | INFO     | [NeMo] Epoch 2/100 | Train Loss: 3.892 | Val Loss: 3.567
...
```

---

## Troubleshooting

### "wandb: command not found"
```bash
pip install wandb
wandb login YOUR_API_KEY
```

### "No such file or directory: codec_preparation_*.log"
```bash
# Logs are created in:
ls data/saba_experiment_a/logs/
```

### "Permission denied" on logs
```bash
chmod -R 755 data/saba_experiment_a/logs/
```

### W&B login stuck
```bash
# Use headless mode
wandb login --relogin YOUR_API_KEY
```

---

## Notes

- **Log files are permanent** — Keep them for reproducibility and debugging
- **W&B is optional** — If login fails, training still runs (logs to local files)
- **TensorBoard alternative** — View training curves with:
  ```bash
  tensorboard --logdir=exp/experiment_a_punctuated_joint_training/
  ```
- **Disk space** — Monitor `data/saba_experiment_a/` to ensure you don't run out
