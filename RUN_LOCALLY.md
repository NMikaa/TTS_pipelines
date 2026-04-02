# How to Run Experiment A Locally

Your W&B API key has been embedded in the setup script. Everything is ready to execute on your machine.

---

## 🚀 One-Command Execution

```bash
cd /root/TTS_pipelines
bash RUN_EXPERIMENT_A.sh
```

That's it! This script will:

1. ✅ Verify your environment (PyTorch, NeMo, CUDA, GPU)
2. ✅ Auto-configure W&B (no manual login needed)
3. ✅ Extract codec tokens (2-3 hours)
4. ✅ Verify extraction completed
5. ✅ Launch training (2-3 weeks)
6. ✅ Generate logs with timestamps
7. ✅ Track metrics on W&B dashboard

---

## 📋 Prerequisites (One-Time Setup)

Before running the script, ensure you have:

### 1. Python 3.8+ with virtualenv

```bash
python3 --version  # Should be 3.8 or higher
```

### 2. Create and activate virtualenv

```bash
python3 -m venv ~/tts_env
source ~/tts_env/bin/activate
```

### 3. Install dependencies

```bash
# Core dependencies
pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1 — adjust for your CUDA version)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# NeMo framework
pip install nemo_toolkit[tts]

# Additional dependencies
pip install pytorch-lightning omegaconf hydra-core librosa kaldialign wandb
```

**Note:** If you have a different CUDA version, replace `cu121` with your version:
- CUDA 11.8: `cu118`
- CUDA 12.1: `cu121`
- CPU only: `cpu`

### 4. Clone/install NeMo

```bash
cd /root/TTS_pipelines
git clone --depth 1 https://github.com/NVIDIA/NeMo.git
pip install -e ./NeMo
```

### 5. Verify installation

```bash
python3 -c "
import torch
import nemo
import wandb
print('✓ All dependencies installed')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPUs: {torch.cuda.device_count()}')
"
```

---

## ▶️ Running the Experiment

### Inside your virtualenv:

```bash
# Activate environment
source ~/tts_env/bin/activate

# Navigate to repo
cd /root/TTS_pipelines

# Run everything
bash RUN_EXPERIMENT_A.sh
```

The script will:
- Print colored output showing each step
- Create logs in `data/saba_experiment_a/logs/`
- Display W&B dashboard link
- Auto-resume if interrupted

---

## 📊 What the Script Does

### Step 1: Environment Verification
```
✓ Python 3.10.12
✓ PyTorch 2.0.1
✓ NeMo installed
✓ 2 GPU(s) available
✓ CUDA 12.1
```

### Step 2: W&B Configuration
```
✓ W&B authenticated
  Project: georgian-tts
  Run: magpie-experiment-a
  Dashboard: https://wandb.ai/
```

### Step 3: Codec Extraction (2-3 hours)
```
Timeline: 2-3 hours (GPU-intensive)
Logs: data/saba_experiment_a/logs/codec_preparation_*.log

Step 1: Resampling 186234 audio files...
  Done: 186234 resampled

Step 2: Pre-computing codec tokens...
  Done: 186234 encoded

Step 3: Converting manifests to NeMo format...
  Wrote 182406 training entries
  Wrote 3770 validation entries
```

### Step 4: Verification
```
Codec files: 186234 (expected ~186k) ✓
NeMo train manifest: 182406 (expected 182406) ✓
NeMo val manifest: 3770 (expected 3770) ✓
```

### Step 5: Training Launch (2-3 weeks)
```
Timeline: 2-3 weeks (2 GPU DDP)
Logs: data/saba_experiment_a/logs/training_*.log

[Training starts with automatic checkpoint saving and early stopping]
```

---

## 📈 Monitoring During Execution

### In Another Terminal:

**Watch codec extraction:**
```bash
tail -f /root/TTS_pipelines/data/saba_experiment_a/logs/codec_preparation_*.log
```

**Watch training:**
```bash
tail -f /root/TTS_pipelines/data/saba_experiment_a/logs/training_*.log
```

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

**W&B Dashboard:**
Open in browser: https://wandb.ai/your-username/georgian-tts

---

## ⚠️ Important Notes

### Disk Space
- Required: ~65 GB
- You have: ~200 GB
- Status: ✅ **Sufficient**

### GPU Requirements
- Minimum: 2 GPUs with 45GB+ VRAM each
- Recommended: A100, H100, or RTX 6000
- Check: `nvidia-smi` (must show 2+ GPUs)

### If Interrupted

**The script automatically resumes from the last checkpoint:**

```bash
# Just re-run the script
bash RUN_EXPERIMENT_A.sh
```

It will:
- Skip completed codec extraction
- Resume training from last saved checkpoint
- No data loss

### If Something Goes Wrong

**Check logs:**
```bash
# Codec extraction errors
cat /root/TTS_pipelines/data/saba_experiment_a/logs/codec_preparation_*.log | grep ERROR

# Training errors
cat /root/TTS_pipelines/data/saba_experiment_a/logs/training_*.log | grep ERROR
```

**See troubleshooting:**
- `EXPERIMENT_A_SETUP.md` → Troubleshooting section
- `EXPERIMENT_A_LOGGING.md` → Troubleshooting section

---

## 📝 Summary

Everything is ready. Your W&B account is configured. Just:

1. **Activate virtualenv** (ensure PyTorch + NeMo installed)
2. **Run the script** `bash RUN_EXPERIMENT_A.sh`
3. **Monitor with logs** or W&B dashboard
4. **Wait for results** (codec: 2-3 hours, training: 2-3 weeks)

The script handles all the complexity. You just execute it and monitor progress.

---

## ✨ What You'll Get

After execution completes:

**Checkpoints:**
```
pipelines/magpie_tts/exp/experiment_a_punctuated_joint_training/version_0/checkpoints/
├── epoch=00-step=001000-val_loss=4.23.ckpt
├── epoch=05-step=005000-val_loss=3.56.ckpt
├── epoch=10-step=010000-val_loss=3.12.ckpt
├── epoch=20-step=020000-val_loss=2.89.ckpt
├── epoch=80-step=080000-val_loss=2.76.ckpt  (best)
└── last.ckpt
```

**Logs:**
```
data/saba_experiment_a/logs/
├── codec_preparation_20240402_150530.log
└── training_20240402_200000.log
```

**W&B Metrics:**
- Training/validation loss curves
- Learning rate decay
- GPU memory usage
- Epoch-by-epoch progress

---

## 🎯 Next: Evaluation

After training completes, evaluate on success criteria:
1. In-distribution CER < 3% (on FLEURS test set)
2. Zero-shot CER < 6% (on held-out speaker)

If both succeed → Proceed to Phase 2 (DPO/GRPO optimization)

---

**Ready?**
```bash
bash RUN_EXPERIMENT_A.sh
```

Good luck! 🚀
