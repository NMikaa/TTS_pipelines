# Experiment A: Quick Start Guide

**Status:** ✅ All preparation complete and ready to execute

---

## 🚀 Start Here: 5-Minute Setup

### Step 1: Activate Your Environment

```bash
source ~/tts_env/bin/activate  # Your virtualenv with torch + nemo_toolkit
cd /root/TTS_pipelines/pipelines/magpie_tts
```

**Verify dependencies:**
```bash
python -c "import torch; import nemo; print('✓ Dependencies OK')"
```

### Step 2: Setup W&B (Optional but Recommended)

```bash
# Get API key from: https://wandb.ai/authorize
bash setup_wandb.sh

# Follow prompt to enter API key
```

### Step 3: Run Codec Extraction (2-3 hours)

```bash
python prepare_experiment_a_codec.py
```

**Monitor in another terminal:**
```bash
tail -f ../../data/saba_experiment_a/logs/codec_preparation_*.log
```

### Step 4: Launch Training (2-3 weeks)

```bash
python run_experiment_a.py
```

**Monitor progress:**
```bash
# Option 1: File logs
tail -f ../../data/saba_experiment_a/logs/training_*.log

# Option 2: TensorBoard
tensorboard --logdir=exp/

# Option 3: W&B Dashboard
# https://wandb.ai/your-username/georgian-tts
```

---

## 📊 What You're Running

**Experiment A: Georgian TTS on 393 hours of punctuated speech**

| Setting | Value |
|---------|-------|
| **Data** | 182k training samples (~393h), 24 speakers |
| **Hold-out** | ზაალ სამადაშვილი (145 samples, 0.25h) for zero-shot testing |
| **Model** | nvidia/magpie_tts_multilingual_357m (357M params) |
| **Training** | 100 epochs (early stopping at ~80) |
| **Batch size** | 48 per GPU × 2 GPUs = 96 effective |
| **Learning rate** | 2e-5 (fine-tuning) |
| **Precision** | bf16-mixed (saves 50% VRAM) |
| **Distributed** | DDP on 2 GPUs |
| **Early stopping** | patience=5 epochs |

---

## 💾 Disk Space

**Required:** ~65 GB additional space

| Component | Size |
|-----------|------|
| Resampled audio (22.05kHz) | ~62 GB |
| Codec tokens (8 codebooks) | ~1-2 GB |
| **Total** | **~63-64 GB** |
| **Your available space** | **~200 GB** ✓ |

✅ **You have sufficient space with comfortable margin**

---

## 📈 Expected Timeline

| Phase | Duration | GPU |
|-------|----------|-----|
| Codec extraction | 2-3 hours | 1 GPU (24GB+) |
| Codec verification | 5 min | CPU |
| Training | 2-3 weeks | 2 GPUs (90GB total) |
| **Total** | **~3.5 weeks** | |

---

## ✅ Success Criteria

**Experiment A validates:**
1. ✓ **In-distribution quality**: CER < 3% on FLEURS test set
2. ✓ **Zero-shot capability**: CER < 6% on held-out speaker

**If both succeed:**
→ Proceed to Phase 2: DPO/GRPO preference optimization

---

## 📁 Files & Locations

### Preparation Scripts (Ready)
```
pipelines/magpie_tts/
├── prepare_experiment_a_codec.py    # Codec extraction
├── run_experiment_a.py              # Training launcher
├── setup_wandb.sh                   # W&B authentication
└── conf/
    └── magpietts_experiment_a.yaml  # Hydra config
```

### Data (Ready)
```
data/saba_experiment_a/
├── train_manifest.json              # 182k training samples
├── val_manifest.json                # 3.7k validation samples
├── holdout_manifest.json            # 145 hold-out samples
└── logs/                            # Will be created
    ├── codec_preparation_*.log
    └── training_*.log
```

### Training Outputs (Will be created)
```
pipelines/magpie_tts/
└── exp/
    └── experiment_a_punctuated_joint_training/
        └── version_0/
            ├── checkpoints/         # Top 5 + last checkpoint
            ├── logs/
            └── hparams.yaml
```

---

## 🔍 Monitoring

### Real-Time Logs

**Codec extraction progress:**
```bash
tail -f data/saba_experiment_a/logs/codec_preparation_*.log

# Watch: Resampling → Codec encoding → Manifest conversion
```

**Training progress:**
```bash
tail -f data/saba_experiment_a/logs/training_*.log

# Watch: Epoch N/100 | Loss decreasing | Early stopping patience
```

### W&B Dashboard

**Project:** georgian-tts  
**Run:** magpie-experiment-a  
**URL:** https://wandb.ai/your-username/georgian-tts

Tracks:
- Training/validation loss
- Learning rate decay
- GPU memory usage
- Training speed

### TensorBoard

```bash
tensorboard --logdir=pipelines/magpie_tts/exp/
# Open: http://localhost:6006
```

---

## 🛠️ Common Commands

### Check codec extraction progress
```bash
# Watch real-time log
tail -f data/saba_experiment_a/logs/codec_preparation_*.log

# Count extracted files
ls data/saba_experiment_a/codec_codes/ | wc -l  # Should reach ~186k
```

### Check training progress
```bash
# Watch training log
tail -f data/saba_experiment_a/logs/training_*.log

# View latest checkpoint
ls -lah pipelines/magpie_tts/exp/experiment_a_*/version_0/checkpoints/ | tail -5
```

### Monitor GPU
```bash
nvidia-smi -l 1  # Updates every second
watch -n 5 nvidia-smi  # Auto-refresh every 5s
```

### Resume interrupted training
```bash
# Just re-run the command — it resumes automatically
python run_experiment_a.py
```

---

## ⚠️ Important Notes

### Before You Start

1. **Disk space:** Verify you have 200GB available
   ```bash
   df -h data/saba_experiment_a/
   ```

2. **GPU availability:** Ensure 2 GPUs with 45GB+ VRAM each
   ```bash
   nvidia-smi
   ```

3. **Environment:** Activate virtualenv with torch + nemo_toolkit
   ```bash
   source ~/tts_env/bin/activate
   ```

### During Execution

1. **Don't delete files** during codec extraction or training
2. **Keep W&B logged in** for continuous monitoring
3. **Save logs** for reproducibility and debugging
4. **Monitor disk space** to ensure you don't run out

### If Something Goes Wrong

**Codec extraction fails?**
- Check: PyTorch version, CUDA availability, disk space
- See: EXPERIMENT_A_LOGGING.md → Troubleshooting

**Training crashes?**
- Automatic resume: Just re-run `python run_experiment_a.py`
- Check: logs in `data/saba_experiment_a/logs/training_*.log`
- See: EXPERIMENT_A_SETUP.md → Troubleshooting

**W&B login issues?**
- Training still works without W&B
- Logs are saved locally regardless
- Try: `wandb login --relogin YOUR_API_KEY`

---

## 📞 Reference Documents

- **EXPERIMENT_A_SETUP.md** — Detailed setup, execution, and decision tree
- **EXPERIMENT_A_LOGGING.md** — Monitoring, W&B, troubleshooting
- **docs/superpowers/specs/2026-04-02-magpie-experiment-a-design.md** — Technical design
- **docs/superpowers/plans/2026-04-02-magpie-experiment-a.md** — Implementation plan

---

## ✨ Summary

Everything is prepared and tested. You just need to:

1. ✅ Activate environment
2. ✅ Setup W&B (optional)
3. ✅ Run codec extraction (2-3 hours)
4. ✅ Run training (2-3 weeks)
5. ✅ Monitor with logs + W&B

**Ready? Start with:**
```bash
cd pipelines/magpie_tts
python prepare_experiment_a_codec.py
```

Good luck! 🚀
