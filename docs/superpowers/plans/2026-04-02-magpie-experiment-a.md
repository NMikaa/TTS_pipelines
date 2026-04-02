# MagPIE Experiment A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune MagPIE TTS on 450h punctuated Georgian with expanded baked embeddings + joint context encoder training, validating data quality hypothesis and zero-shot voice cloning.

**Architecture:** 
1. Filter saba_data to punctuated-only, hold out 1 speaker for zero-shot validation
2. Create NeMo manifests with speaker remapping (40 training speakers → embeddings 5-40)
3. Expand baked embeddings from 5→41, keep context encoder active for joint training
4. Train on 2 GPUs with DDP for 100 epochs (patience=5 early stopping)
5. Evaluate: in-distribution CER on FLEURS + zero-shot on held-out speaker

**Tech Stack:** PyTorch 2.11, NeMo, Hydra, DDP, OmniASR-LLM-7B for evaluation

---

## File Structure

**New files to create:**
- `pipelines/magpie_tts/prepare_experiment_a_data.py` — Filter punctuated + speaker remapping
- `pipelines/magpie_tts/conf/experiment_a_overrides.yaml` — Hydra config for training
- `pipelines/magpie_tts/evaluate_experiment_a.py` — CER/WER + zero-shot evaluation

**Files to modify:**
- `pipelines/magpie_tts/config.py` — Add `exp_a_config` with 450h defaults
- `pipelines/magpie_tts/train.py` — Add manifest creation for Experiment A (optional wrapper)

**Directories to create:**
- `exp/magpie_tts_georgian_saba_450h/` — Training output directory
- `results/experiment_a/` — Evaluation results

---

## Tasks

### Task 1: Analyze Speaker Distribution and Identify Hold-Out Speaker

**Files:**
- Read: `data/saba_clean/train_manifest.json`
- Output: Speaker list with hour counts

- [ ] **Step 1: Write script to count hours per speaker**

Create `/tmp/analyze_speakers.py`:
```python
import json
from collections import defaultdict

speaker_hours = defaultdict(float)
total_segs = 0

with open('/root/TTS_pipelines/data/saba_clean/train_manifest.json') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('has_punctuation'):  # Only punctuated
            speaker = entry['speaker_id']
            duration = entry.get('duration', 0)
            speaker_hours[speaker] += duration
            total_segs += 1

# Sort by hours
sorted_speakers = sorted(speaker_hours.items(), key=lambda x: x[1])
print(f"Total punctuated segments: {total_segs:,}")
print(f"Total hours: {sum(speaker_hours.values())/3600:.1f}h")
print(f"\nAll {len(speaker_hours)} speakers:")
for spk, hours in sorted_speakers:
    print(f"  {spk:30s}: {hours/3600:6.1f}h ({int(hours/3600*100/1384)}/{100}%)")

print(f"\nSmallest speaker (hold-out): {sorted_speakers[0][0]} ({sorted_speakers[0][1]/3600:.2f}h)")
```

- [ ] **Step 2: Run analysis**

```bash
cd /root/TTS_pipelines
python /tmp/analyze_speakers.py
```

Expected output:
```
Total punctuated segments: 182,600
Total hours: 450.1h

All 41 speakers:
  [speaker with lowest hours]: X.XXh (Y/100%)
  ...
  თეკო ჩუბინიძე: 300.9h (21/100%)

Smallest speaker (hold-out): [name] (X.XXh)
```

- [ ] **Step 3: Document hold-out speaker**

Save to `/tmp/exp_a_metadata.txt`:
```
HOLD_OUT_SPEAKER=[name from smallest speaker]
TRAINING_SPEAKERS=40
TOTAL_PUNCTUATED_HOURS=450.1
TOTAL_PUNCTUATED_SEGMENTS=182600
```

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
# No commit needed - analysis script in /tmp
```

---

### Task 2: Create Punctuated-Only Data Filter

**Files:**
- Create: `pipelines/magpie_tts/prepare_experiment_a_data.py`
- Input: `data/saba_clean/train_manifest.json`, `data/saba_clean/eval_manifest.json`
- Output: `data/experiment_a/train_manifest.json`, `data/experiment_a/eval_manifest.json`

- [ ] **Step 1: Write data filtering script**

```python
# pipelines/magpie_tts/prepare_experiment_a_data.py

import json
import argparse
from pathlib import Path
from collections import defaultdict

def filter_punctuated(input_manifest: str, output_manifest: str, hold_out_speaker: str):
    """Filter to punctuated-only and exclude hold-out speaker."""
    
    with open(input_manifest) as f_in, open(output_manifest, 'w') as f_out:
        count = 0
        held_out = 0
        
        for line in f_in:
            entry = json.loads(line)
            
            # Keep only punctuated
            if not entry.get('has_punctuation', False):
                continue
            
            # Exclude hold-out speaker
            if entry.get('speaker_id') == hold_out_speaker:
                held_out += 1
                continue
            
            # Write to output
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    
    return count, held_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-train', default='../../data/saba_clean/train_manifest.json')
    parser.add_argument('--input-eval', default='../../data/saba_clean/eval_manifest.json')
    parser.add_argument('--output-dir', default='../../data/experiment_a')
    parser.add_argument('--hold-out-speaker', required=True, help='Speaker name to exclude')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter train
    train_count, train_held = filter_punctuated(
        args.input_train,
        str(output_dir / 'train_manifest.json'),
        args.hold_out_speaker
    )
    
    # Filter eval
    eval_count, eval_held = filter_punctuated(
        args.input_eval,
        str(output_dir / 'eval_manifest.json'),
        args.hold_out_speaker
    )
    
    print(f"Train: {train_count:,} kept, {train_held} held out")
    print(f"Eval:  {eval_count:,} kept, {eval_held} held out")
    print(f"Total: {train_count + eval_count:,} segments for Experiment A")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run data filtering**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
HOLD_OUT_SPEAKER=$(cat /tmp/exp_a_metadata.txt | grep HOLD_OUT_SPEAKER | cut -d= -f2)
python prepare_experiment_a_data.py \
  --input-train ../../data/saba_clean/train_manifest.json \
  --input-eval ../../data/saba_clean/eval_manifest.json \
  --output-dir ../../data/experiment_a \
  --hold-out-speaker "$HOLD_OUT_SPEAKER"
```

Expected output:
```
Train: ~180,000 kept, ~2,600 held out
Eval:  ~3,600 kept, ~60 held out
Total: ~183,600 segments for Experiment A
```

- [ ] **Step 3: Verify output manifests**

```bash
cd /root/TTS_pipelines
wc -l data/experiment_a/*.json
head -1 data/experiment_a/train_manifest.json | python -m json.tool | head -20
```

Expected: Two valid JSONL files with has_punctuation=true

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/prepare_experiment_a_data.py
git commit -m "feat: add data filtering script for Experiment A (punctuated-only, hold-out speaker)"
```

---

### Task 3: Prepare Audio Resampling & Codec Token Extraction

**Files:**
- Use existing: `pipelines/magpie_tts/train.py` (Stage 1 & 2)
- Input: `data/experiment_a/train_manifest.json` (audio paths reference `saba_clean/audio/`)
- Output: `data/experiment_a/audio_22khz/`, `data/experiment_a/codec_codes/`

- [ ] **Step 1: Write resampling wrapper script**

```python
# pipelines/magpie_tts/prepare_experiment_a_codec.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import MagPIEConfig
from train import resample_audio, precompute_codec_tokens

def main():
    config = MagPIEConfig(
        sample_rate=22050,
        source_sample_rate=24000,  # saba_data is 22kHz, but be explicit
        codec_model='nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps'
    )
    
    data_dir = Path('../../data/experiment_a')
    
    # Resample
    print("Stage 1: Resampling audio to 22kHz...")
    resample_audio(config, data_dir)
    
    # Precompute codec
    print("Stage 2: Pre-computing codec tokens...")
    resampled_dir = data_dir / 'audio_22khz'
    precompute_codec_tokens(config, resampled_dir)
    
    print("✓ Codec preparation complete")
    print(f"  Resampled audio: {resampled_dir}")
    print(f"  Codec codes: {data_dir / 'codec_codes'}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run resampling (takes ~2-3 hours)**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
python prepare_experiment_a_codec.py 2>&1 | tee exp_a_codec_prep.log &
```

Run in background, monitor with:
```bash
tail -f /root/TTS_pipelines/pipelines/magpie_tts/exp_a_codec_prep.log
```

- [ ] **Step 3: Verify codec preparation**

Once complete:
```bash
cd /root/TTS_pipelines
ls data/experiment_a/ | sort
du -sh data/experiment_a/audio_22khz data/experiment_a/codec_codes
find data/experiment_a/codec_codes -name "*.pt" | wc -l
```

Expected:
- audio_22khz/: ~180GB (resampled audio)
- codec_codes/: ~50GB (pre-computed tokens)
- ~180k .pt files

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/prepare_experiment_a_codec.py
git commit -m "feat: add codec preparation script for Experiment A"
```

---

### Task 4: Create NeMo Manifests with Speaker Remapping

**Files:**
- Create: `pipelines/magpie_tts/prepare_experiment_a_nemo_manifests.py`
- Input: `data/experiment_a/train_manifest.json` + codec codes
- Output: `data/experiment_a/train_manifest_nemo.json`, `data/experiment_a/eval_manifest_nemo.json`

- [ ] **Step 1: Write NeMo manifest creator**

```python
# pipelines/magpie_tts/prepare_experiment_a_nemo_manifests.py

import json
import sys
from pathlib import Path
from collections import defaultdict

def create_nemo_manifests(data_dir: str, output_dir: str):
    """Convert saba_clean manifests to NeMo format with speaker remapping."""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build speaker ID → embedding index mapping (5-40)
    speaker_to_idx = {}
    speaker_idx = 5  # Start from 5 (0-4 are pre-trained)
    
    # First pass: collect unique speakers
    for manifest_file in ['train_manifest.json', 'eval_manifest.json']:
        with open(data_dir / manifest_file) as f:
            for line in f:
                entry = json.loads(line)
                spk = entry.get('speaker_id')
                if spk and spk not in speaker_to_idx:
                    speaker_to_idx[spk] = speaker_idx
                    speaker_idx += 1
    
    print(f"Mapped {len(speaker_to_idx)} speakers to embeddings 5-{speaker_idx-1}")
    
    # Process manifests
    for split in ['train', 'eval']:
        input_file = data_dir / f'{split}_manifest.json'
        output_file = output_dir / f'{split}_manifest_nemo.json'
        
        with open(input_file) as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                entry = json.loads(line)
                seg_id = entry['id']
                audio_path = entry['audio_path']
                text = entry['text']
                speaker = entry['speaker_id']
                duration = entry['duration']
                
                # Codec path
                codec_path = data_dir / 'codec_codes' / f'{seg_id}.pt'
                
                # NeMo manifest entry
                nemo_entry = {
                    'audio_filepath': str(audio_path.resolve()),
                    'text': text,
                    'duration': round(duration, 4),
                    'speaker': speaker_to_idx[speaker],  # Mapped index
                    'target_audio_codes_path': str(codec_path),
                }
                
                f_out.write(json.dumps(nemo_entry, ensure_ascii=False) + '\n')
        
        print(f"✓ {output_file}")

if __name__ == '__main__':
    create_nemo_manifests(
        '../../data/experiment_a',
        '../../data/experiment_a'
    )
```

- [ ] **Step 2: Run manifest creation**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
python prepare_experiment_a_nemo_manifests.py
```

Expected output:
```
Mapped 40 speakers to embeddings 5-44
✓ ../../data/experiment_a/train_manifest_nemo.json
✓ ../../data/experiment_a/eval_manifest_nemo.json
```

- [ ] **Step 3: Verify manifests**

```bash
cd /root/TTS_pipelines
wc -l data/experiment_a/*_nemo.json
head -1 data/experiment_a/train_manifest_nemo.json | python -m json.tool
```

Expected:
- Two valid JSONL files
- Each entry has: audio_filepath, text, duration, speaker (int 5-44), target_audio_codes_path

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/prepare_experiment_a_nemo_manifests.py
git commit -m "feat: create NeMo manifests with speaker remapping (5-44) for Experiment A"
```

---

### Task 5: Create Experiment A Training Configuration

**Files:**
- Create: `pipelines/magpie_tts/conf/experiment_a_overrides.yaml`
- Modify: `pipelines/magpie_tts/config.py` (add experiment_a config)

- [ ] **Step 1: Create Hydra overrides file**

```yaml
# pipelines/magpie_tts/conf/experiment_a_overrides.yaml

# Experiment A: 450h punctuated Georgian with expanded baked embeddings + joint context encoder

# Data
train_ds_meta:
  experiment_a_train:
    manifest_path: '${oc.env:DATA_DIR,../../data/experiment_a}/train_manifest_nemo.json'
    audio_dir: '${oc.env:DATA_DIR,../../data/experiment_a}/audio_22khz'
    feature_dir: '${oc.env:DATA_DIR,../../data/experiment_a}/codec_codes'
    sample_weight: 1.0
    tokenizer_names: [text_ce_tokenizer]

val_ds_meta:
  experiment_a_eval:
    manifest_path: '${oc.env:DATA_DIR,../../data/experiment_a}/eval_manifest_nemo.json'
    audio_dir: '${oc.env:DATA_DIR,../../data/experiment_a}/audio_22khz'
    feature_dir: '${oc.env:DATA_DIR,../../data/experiment_a}/codec_codes'
    sample_weight: 1.0
    tokenizer_names: [text_ce_tokenizer]

# Training hyperparameters
max_epochs: 100
batch_size: 48

# Model
model:
  num_speakers: 41  # 5 pre-trained + 36 new Georgian
  baked_speaker_embed_dim: 96  # Expand from 5 to 41
  context_encoder_active: true  # Always active (not frozen)
  codecmodel_path: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps
  optim:
    lr: 2e-5
    sched:
      name: ExponentialDecay
      decay_step: 1
      decay_rate: 0.998
    capturable: true

# Trainer
trainer:
  devices: 2
  strategy: ddp  # Distributed Data Parallel
  precision: bf16-mixed
  gradient_clip_val: 2.5
  log_every_n_steps: 50
  check_val_every_n_epoch: 1
  max_epochs: 100
  num_sanity_val_steps: 2

# Early stopping
callbacks:
  early_stopping:
    monitor: val_loss
    patience: 5
    min_delta: 0.0001

# Checkpointing
exp_manager:
  exp_dir: ./exp/magpie_tts_georgian_saba_450h
  name: experiment_a_punctuated_joint_training
  resume_if_exists: true
  version: null
  save_top_k: 5
```

- [ ] **Step 2: Update config.py with experiment_a defaults**

Add to `config.py`:
```python
# In MagPIEConfig dataclass, add comment/docstring:

"""
Experiment A Configuration:
- Data: 450h punctuated Georgian (40 training speakers + 1 held-out)
- Epochs: 100 with early stopping (patience=5)
- Batch size: 48 per GPU × 2 GPUs = 96 effective
- Learning rate: 2e-5 (10x lower than pretraining)
- Speakers: 41 (5 pre-trained baked + 36 new Georgian)
- Context encoder: Active from epoch 1 (joint training)
- Codec: NanoCodec (nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
"""

# Then add to __post_init__:
if os.getenv('EXPERIMENT_A'):
    self.max_epochs = 100
    self.batch_size = 48
    self.learning_rate = 2e-5
    self.num_speakers = 41
    # ... etc
```

- [ ] **Step 3: Verify YAML syntax**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
python -c "import yaml; yaml.safe_load(open('conf/experiment_a_overrides.yaml'))"
# Should print nothing (no errors)
```

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/conf/experiment_a_overrides.yaml
git add pipelines/magpie_tts/config.py
git commit -m "feat: add Experiment A training configuration (450h punctuated, expanded baked embeddings, joint context encoder)"
```

---

### Task 6: Launch Distributed Training

**Files:**
- Use existing: `pipelines/magpie_tts/train.py` (NeMo entry point)
- Create: `pipelines/magpie_tts/run_experiment_a.sh` (convenience script)

- [ ] **Step 1: Create training launch script**

```bash
#!/bin/bash
# pipelines/magpie_tts/run_experiment_a.sh

set -e

cd "$(dirname "$0")"

# Environment setup
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DATA_DIR="../../data/experiment_a"

echo "Experiment A: 450h Punctuated Georgian Fine-tuning"
echo "=================================================="
echo "Data dir: $DATA_DIR"
echo "Training on 2 GPUs with DDP"
echo ""

# Verify manifests exist
if [ ! -f "$DATA_DIR/train_manifest_nemo.json" ]; then
    echo "ERROR: $DATA_DIR/train_manifest_nemo.json not found"
    exit 1
fi

# Launch NeMo training with Experiment A overrides
python -m torch.distributed.launch --nproc_per_node 2 \
    /root/TTS_pipelines/NeMo/examples/tts/magpietts.py \
    --config-name magpietts_georgian \
    --config-path conf \
    $(cat <<'EOF'
max_epochs=100
batch_size=48
+model.optim.lr=2e-5
trainer.devices=2
trainer.strategy=ddp
trainer.precision=bf16-mixed
trainer.gradient_clip_val=2.5
trainer.log_every_n_steps=50
trainer.check_val_every_n_epoch=1
trainer.max_epochs=100
+train_ds_meta.experiment_a_train.manifest_path=$DATA_DIR/train_manifest_nemo.json
+train_ds_meta.experiment_a_train.audio_dir=$DATA_DIR/audio_22khz
+train_ds_meta.experiment_a_train.feature_dir=$DATA_DIR/codec_codes
+val_ds_meta.experiment_a_eval.manifest_path=$DATA_DIR/eval_manifest_nemo.json
+val_ds_meta.experiment_a_eval.audio_dir=$DATA_DIR/audio_22khz
+val_ds_meta.experiment_a_eval.feature_dir=$DATA_DIR/codec_codes
+model.num_speakers=41
+model.optim.lr=2e-5
exp_manager.exp_dir=./exp/magpie_tts_georgian_saba_450h
exp_manager.name=experiment_a_punctuated_joint
exp_manager.version=null
exp_manager.save_top_k=5
EOF
    )
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x /root/TTS_pipelines/pipelines/magpie_tts/run_experiment_a.sh
```

- [ ] **Step 3: Pre-training sanity checks**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts

# Check manifests
echo "Train manifest:"
wc -l $DATA_DIR/train_manifest_nemo.json
head -1 $DATA_DIR/train_manifest_nemo.json | python -m json.tool | head -10

# Check GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); print(f'VRAM per GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

Expected:
- Two GPU devices available
- ~45-48GB VRAM per GPU

- [ ] **Step 4: Launch training (in background)**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
nohup ./run_experiment_a.sh > exp_a_training.log 2>&1 &
echo $! > exp_a_training.pid
cat exp_a_training.pid
```

Monitor with:
```bash
tail -f /root/TTS_pipelines/pipelines/magpie_tts/exp_a_training.log
```

Expected to see:
```
Experiment A: 450h Punctuated Georgian Fine-tuning
...
[NeMo] Model type: MagpieTTSModel
[NeMo] Initializing from pretrained model...
[Epoch 1/100] Training...
```

- [ ] **Step 5: Commit launch script**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/run_experiment_a.sh
git commit -m "feat: add Experiment A DDP training launch script"
```

---

### Task 7: Monitor Training & Track Checkpoints

**Files:**
- Monitor: `exp/magpie_tts_georgian_saba_450h/experiment_a_punctuated_joint/checkpoints/`
- Create: `pipelines/magpie_tts/monitor_experiment_a.py` (optional monitoring script)

- [ ] **Step 1: Wait for training to reach epoch 10**

```bash
# Check training progress every 30 minutes
watch -n 1800 'tail -20 /root/TTS_pipelines/pipelines/magpie_tts/exp_a_training.log'
```

Or manually:
```bash
tail -50 /root/TTS_pipelines/pipelines/magpie_tts/exp_a_training.log | grep -E "Epoch|loss"
```

- [ ] **Step 2: Verify checkpoint saving**

Once epoch 1 completes:
```bash
ls -lh /root/TTS_pipelines/exp/magpie_tts_georgian_saba_450h/experiment_a_punctuated_joint/checkpoints/
```

Expected:
```
-rw-r--r-- last.ckpt (~1.4GB)
-rw-r--r-- epoch_1.ckpt
-rw-r--r-- epoch_2.ckpt (after epoch 2 completes)
```

- [ ] **Step 3: Create monitoring script (optional)**

```python
# pipelines/magpie_tts/monitor_experiment_a.py

import json
import glob
from pathlib import Path

def monitor_training():
    """Quick status check of Experiment A training."""
    
    ckpt_dir = Path('./exp/magpie_tts_georgian_saba_450h/experiment_a_punctuated_joint/checkpoints')
    
    # Count checkpoints
    checkpoints = sorted(glob.glob(str(ckpt_dir / '*.ckpt')))
    print(f"✓ {len(checkpoints)} checkpoints saved")
    
    # Show recent epochs
    recent = sorted(glob.glob(str(ckpt_dir / 'epoch_*.ckpt')))[-3:]
    for ckpt in recent:
        size_gb = Path(ckpt).stat().st_size / 1e9
        print(f"  {Path(ckpt).name}: {size_gb:.1f}GB")
    
    # Check for best.ckpt
    if (ckpt_dir / 'best.ckpt').exists():
        print(f"✓ Best checkpoint selected: best.ckpt")
    else:
        print("⊘ Best checkpoint not yet selected (training in progress)")

if __name__ == '__main__':
    monitor_training()
```

- [ ] **Step 4: Run monitoring script**

```bash
cd /root/TTS_pipelines/pipelines/magpie_tts
python monitor_experiment_a.py
```

- [ ] **Step 5: Commit (when training starts successfully)**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/monitor_experiment_a.py
git commit -m "feat: add training monitoring script for Experiment A"
```

---

### Task 8: Create Evaluation Script for CER Computation

**Files:**
- Create: `pipelines/magpie_tts/evaluate_experiment_a.py`
- Input: Best 5 checkpoints from training
- Output: CER/WER results

- [ ] **Step 1: Write evaluation script**

```python
# pipelines/magpie_tts/evaluate_experiment_a.py

import torch
import json
import tempfile
from pathlib import Path
from collections import defaultdict

def evaluate_checkpoint(checkpoint_path: str, output_dir: str = './results/experiment_a'):
    """
    Evaluate a checkpoint on FLEURS Georgian test set.
    Requires: checkpoint, FLEURS dataset, omniASR-LLM-7B
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from nemo.collections.tts.models import MagpieTTSModel
    from shared.evaluation.intelligibility import transcribe_omnilingual
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Load model
    model = MagpieTTSModel.restore_from(checkpoint_path)
    model.eval()
    model.cuda()
    
    # Load FLEURS Georgian test set
    from datasets import load_dataset
    fleurs = load_dataset('google/fleurs', 'ka_GE')
    test_samples = fleurs['test'].select(range(min(979, len(fleurs['test']))))
    
    generated_audios = []
    
    # Generate speech for FLEURS samples
    print(f"Generating {len(test_samples)} audio samples...")
    for i, sample in enumerate(test_samples):
        text = sample['text']
        
        try:
            # Generate using speaker 1 (baked embedding, fast path)
            with torch.no_grad():
                audio = model.generate_speech(
                    text=text,
                    speaker=1,
                    use_cfg=True
                )
            
            # Save temp audio
            temp_file = output_dir / f'temp_{i}.wav'
            model.save_audio(audio, temp_file, 22050)
            generated_audios.append(temp_file)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(test_samples)}")
        
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            continue
    
    # Transcribe with omniASR
    print(f"\nTranscribing {len(generated_audios)} generated samples...")
    transcriptions = transcribe_omnilingual(
        [str(a) for a in generated_audios],
        lang='kat_Geor',
        batch_size=8
    )
    
    # Compute CER/WER
    from jiwer import cer, wer
    
    reference_texts = [sample['text'] for sample in test_samples[:len(generated_audios)]]
    
    cer_score = cer(reference_texts, transcriptions)
    wer_score = wer(reference_texts, transcriptions)
    
    result = {
        'checkpoint': checkpoint_path,
        'num_samples': len(generated_audios),
        'cer': float(cer_score),
        'wer': float(wer_score),
    }
    
    # Save results
    result_file = output_dir / 'evaluation_results.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Evaluation complete")
    print(f"  CER: {cer_score:.4f}")
    print(f"  WER: {wer_score:.4f}")
    print(f"  Results saved to {result_file}")
    
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--output-dir', default='./results/experiment_a')
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.output_dir)
```

- [ ] **Step 2: Install evaluation dependencies**

```bash
pip install -q jiwer google-datasets
```

- [ ] **Step 3: Test evaluation script (placeholder)**

```bash
# Will run after training completes
cd /root/TTS_pipelines/pipelines/magpie_tts
echo "Evaluation script ready. Will run after training completes."
echo "Command: python evaluate_experiment_a.py --checkpoint ./exp/magpie_tts_georgian_saba_450h/experiment_a_punctuated_joint/checkpoints/best.ckpt"
```

- [ ] **Step 4: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/evaluate_experiment_a.py
git commit -m "feat: add Experiment A evaluation script (CER/WER via round-trip ASR)"
```

---

### Task 9: Create Zero-Shot Voice Cloning Test Script

**Files:**
- Create: `pipelines/magpie_tts/test_zeroshot_experiment_a.py`
- Input: Best checkpoint, held-out speaker reference audio, FLEURS test set

- [ ] **Step 1: Write zero-shot test script**

```python
# pipelines/magpie_tts/test_zeroshot_experiment_a.py

import torch
import json
from pathlib import Path

def test_zeroshot_voice_cloning(checkpoint_path: str, hold_out_speaker: str, output_dir: str = './results/experiment_a'):
    """
    Test zero-shot voice cloning on held-out speaker.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from nemo.collections.tts.models import MagpieTTSModel
    from shared.evaluation.intelligibility import transcribe_omnilingual
    from jiwer import cer
    
    print(f"Zero-shot voice cloning test for: {hold_out_speaker}")
    
    # Load model
    model = MagpieTTSModel.restore_from(checkpoint_path)
    model.eval()
    model.cuda()
    
    # Load FLEURS Georgian test set
    from datasets import load_dataset
    fleurs = load_dataset('google/fleurs', 'ka_GE')
    test_samples = fleurs['test'].select(range(min(50, len(fleurs['test']))))  # Use 50 samples for speed
    
    # Get reference audio from held-out speaker
    # (Simplified: assumes we saved reference audio during data prep)
    ref_audio_path = Path('../../data/experiment_a') / f'{hold_out_speaker}_reference.wav'
    
    if not ref_audio_path.exists():
        print(f"⚠ Reference audio not found: {ref_audio_path}")
        print("  Skipping zero-shot test. Ensure reference audio is saved during data prep.")
        return None
    
    # Load reference audio
    import torchaudio
    ref_audio, sr = torchaudio.load(ref_audio_path)
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        ref_audio = resampler(ref_audio)
    
    generated_audios = []
    
    # Generate speech using context encoder (voice cloning)
    print(f"Generating {len(test_samples)} samples with context encoder...")
    for i, sample in enumerate(test_samples):
        text = sample['text']
        
        try:
            with torch.no_grad():
                # Use context encoder pathway
                audio = model.generate_speech(
                    text=text,
                    context_audio=ref_audio,
                    use_cfg=True
                )
            
            temp_file = output_dir / f'zeroshot_{i}.wav'
            model.save_audio(audio, temp_file, 22050)
            generated_audios.append(temp_file)
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(test_samples)}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Evaluate
    print(f"\nTranscribing {len(generated_audios)} zero-shot samples...")
    transcriptions = transcribe_omnilingual(
        [str(a) for a in generated_audios],
        lang='kat_Geor',
        batch_size=4
    )
    
    reference_texts = [sample['text'] for sample in test_samples[:len(generated_audios)]]
    cer_score = cer(reference_texts, transcriptions)
    
    result = {
        'held_out_speaker': hold_out_speaker,
        'num_samples': len(generated_audios),
        'zero_shot_cer': float(cer_score),
    }
    
    result_file = output_dir / 'zeroshot_results.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Zero-shot test complete")
    print(f"  Held-out speaker: {hold_out_speaker}")
    print(f"  Zero-shot CER: {cer_score:.4f}")
    print(f"  Results saved to {result_file}")
    
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--hold-out-speaker', required=True)
    parser.add_argument('--output-dir', default='./results/experiment_a')
    args = parser.parse_args()
    
    test_zeroshot_voice_cloning(args.checkpoint, args.hold_out_speaker, args.output_dir)
```

- [ ] **Step 2: Commit**

```bash
cd /root/TTS_pipelines
git add pipelines/magpie_tts/test_zeroshot_experiment_a.py
git commit -m "feat: add zero-shot voice cloning test for Experiment A"
```

---

### Task 10: Document Results & Decision Tree

**Files:**
- Create: `results/experiment_a/RESULTS.md`

- [ ] **Step 1: Document evaluation procedure**

```markdown
# Experiment A Results

## Training Summary
- **Data**: 450h punctuated Georgian, 40 training speakers + 1 held-out
- **Epochs**: 100 (stopped at epoch ~XX with early stopping)
- **Best checkpoint**: best.ckpt (epoch X, val_loss=X.XXX)

## In-Distribution Evaluation (40 Training Speakers)

### CER/WER on FLEURS Georgian test set
- **CER**: X.XX% (Target: < 3%)
- **WER**: X.XX%
- **Interpretation**: [Meets/Exceeds/Below target]

### Speaker Embedding Quality
- Baked embeddings clustered properly: [Yes/No]
- Gradients flowing through all 41 embeddings: [Yes/No]

### Inference Speed
- Baked embedding path: < 1s ✓
- Context encoder path: ~2-3s ✓

## Zero-Shot Voice Cloning (Held-Out Speaker)

### Zero-Shot CER
- **Held-out speaker**: [name]
- **Zero-shot CER**: X.XX% (Target: < 6%, preferably within ~2% of training speaker CER)
- **Generalization**: [Success/Failure]

### Qualitative Samples
- Audio samples: [links to selected generated clips]
- Listener notes: [subjective quality assessment]

## Decision & Next Steps

### If CER < 3% AND zero-shot < 6%:
✅ **HYPOTHESIS VALIDATED**
- High-quality data + joint training works
- Proceed to Phase 2: DPO/GRPO preference optimization
- Expected CER improvement: ~40% reduction (to ~1.5-1.8%)

### If CER 3-5% AND zero-shot works:
⚠️ **ACCEPTABLE, BUT ROOM FOR IMPROVEMENT**
- Data quality hypothesis validated (no major issues)
- Compare with Experiment B (all 1,384h data) OR proceed to Phase 2
- Decision: More data vs preference optimization trade-off

### If CER > 5% OR zero-shot fails:
❌ **DEBUG REQUIRED**
- Check: Tokenizer offsets, DDP sync, learning rate scale, codec paths
- Investigate before running Experiments B/C
- Potential fixes: Adjust LR, increase patience, verify manifests

## Files & Checkpoints

- **Best checkpoint**: `exp/magpie_tts_georgian_saba_450h/experiment_a_punctuated_joint/checkpoints/best.ckpt`
- **Top 5 checkpoints**: `checkpoints/epoch_*.ckpt`
- **Training log**: `pipelines/magpie_tts/exp_a_training.log`
```

- [ ] **Step 2: After training completes, fill in results**

```bash
cd /root/TTS_pipelines

# Copy template
cp results/experiment_a/RESULTS.md results/experiment_a/RESULTS.md.backup

# Fill in results from evaluation scripts
# (This happens after eval_experiment_a.py and test_zeroshot_experiment_a.py complete)
```

- [ ] **Step 3: Final commit with results**

```bash
cd /root/TTS_pipelines
git add results/experiment_a/RESULTS.md
git commit -m "docs: document Experiment A results and decision tree"
```

---

## Summary

**Total Implementation Steps: 39**

**Key Milestones:**
1. Data filtering & manifest creation (Tasks 1-4) — ~1 day
2. Codec preparation (Task 5) — ~3 hours
3. Training configuration & launch (Tasks 6-7) — ~1 day setup + 2-3 weeks training
4. Evaluation (Tasks 8-10) — ~1 day

**Expected Timeline:**
- Data prep: 2-3 days
- Training: 2-3 weeks (with potential early stopping at epoch ~70-80)
- Evaluation: 1-2 days
- **Total: ~3-4 weeks**

**Success Criteria:**
- ✅ CER < 3% on FLEURS (target 1.5-2.5%, beat 2.16% baseline)
- ✅ Zero-shot CER < 6% (success if < 2% worse than training speakers)
- ✅ No NaN/Inf losses, proper DDP synchronization
- ✅ Top 5 checkpoints saved, best.ckpt identified

**Next Steps After Experiment A:**
- If successful: → Phase 2 (DPO/GRPO preference optimization)
- If mediocre: → Run Experiment B (all 1,384h data)
- If failed: → Debug tokenizer offsets, DDP sync, learning rate