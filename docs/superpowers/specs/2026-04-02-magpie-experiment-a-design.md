# MagPIE TTS Experiment A: Georgian Fine-tuning Design

**Date:** 2026-04-02  
**Status:** Design Approved  
**Project:** Georgian TTS Benchmark on saba_data (1,524h corpus)  
**Objective:** Validate data quality hypothesis + zero-shot voice cloning capability

---

## Executive Summary

Experiment A trains MagPIE TTS on 450 hours of **high-quality punctuated Georgian speech** with:
- **40 training speakers** + **1 held-out speaker for zero-shot validation**
- **Expanded baked embeddings** (5 pre-trained + 36 new) for fast speaker inference
- **Joint context encoder training** for zero-shot voice cloning capability
- **100 epochs** with early stopping to reach convergence on large dataset

**Success criteria:**
- In-distribution CER < 3% on FLEURS test set
- Zero-shot voice cloning works on held-out speaker (CER < 6%)
- Both baked embeddings and context encoder train successfully

---

## Section 1: Data Strategy

### Dataset Composition

**Source:** `saba_clean/` (prepared from NMikka/saba_data, 1,524h raw corpus)

**Training data selection:**
- **Punctuated segments only:** 182,600 segments (~450 hours)
- **Reason:** Punctuation provides prosody cues (pauses, intonation) essential for quality TTS. Mixed punctuated/unpunctuated training degrades prosody consistency.
- **Data filtering already applied:** Intro/outro clipping (first 5 + last 5 segments per book), duration bounds (1.5-25s), CPS range (5-25), empty text removal

**Speaker split:**
- **Training:** 40 speakers (all 450h distributed across them)
- **Zero-shot held-out:** 1 speaker (smallest data volume, never trained on)
- **Rationale:** Testing on data-poorest speaker is hardest case; if zero-shot works here, it proves generalization

**Train/validation split:**
- **98% training / 2% validation** via deterministic hash on segment ID
- **No speaker overlap** between train and validation (same speaker appears in both, but different segments)
- **Validation set size:** ~3,600 segments (~9 hours)

**Metadata preserved in manifest:**
- `has_punctuation`: Boolean flag (all True for this experiment)
- `multi_speaker`: Boolean flag (all False after filtering)
- `book_title`: Source audiobook identifier
- `speaker_id`: Speaker name (40 unique values)
- `duration`: Segment duration in seconds
- `audio_path`: Path to WAV file

### Rationale for Data Choice

**Why 450h punctuated first?**
- Validates core hypothesis: Does clean, high-quality data produce strong results?
- Establishes quality ceiling before testing mixed/unpunctuated data
- 450h is substantial (~12.8x your previous 35h training), enough for convergence but not overwhelming
- Punctuated data reduces confounding variables (prosody uncertainty)

---

## Section 2: Model Configuration

### Base Architecture

**Pretrained model:** `nvidia/magpie_tts_multilingual_357m`
- 357M total parameters
- Text encoder: 6-layer transformer (ByT5-small tokenization)
- Decoder: 12-layer transformer with CTC alignment
- Codec: NanoCodec (8 codebooks, 22.05kHz, 21.5 fps)

### Speaker Conditioning: Hybrid Approach

**Baked embeddings (fast inference):**
- Expand from 5 → 41 speakers
- Keep 5 pre-trained embeddings (speakers 0-4, from multilingual pretrain)
- Add 36 new random-initialized embeddings for Georgian speakers (speakers 5-40)
- Embedding dimension: `T*D` (time × embedding_dim from pretrained model)
- **Training:** All 41 embeddings trainable (no freezing)

**Context encoder (zero-shot voice cloning):**
- 1-layer transformer (18.1M parameters, part of pretrained model)
- Input: Reference audio (3-8s random window at 22.05kHz)
- Output: Speaker conditioning vector
- **Training:** Active from epoch 1 (not frozen)
- **Purpose:** Learn to extract speaker identity from arbitrary reference audio

### Joint Training of Both Pathways

**During training:**
- Every sample includes reference audio (different clip from same speaker)
- Both baked embedding pathway AND context encoder pathway are active
- Model learns to use whichever is most useful per speaker
- Gradients flow through both pathways simultaneously

**Why both pathways together:**
- Baked embeddings: Fast inference for known speakers (41 Georgian speakers in production)
- Context encoder: Zero-shot capability for new speakers (arbitrary voice cloning)
- Joint training: Model naturally learns best allocation; no artificial weighting needed
- Proven approach: Multi-task learning lets both specialize while sharing decoder representations

### Critical Implementation Notes

**Tokenizer offset preservation:**
- MagPIE uses combined vocabulary from 10+ tokenizers
- Georgian `text_ce_tokenizer` has fixed offset in embedding table
- **Must load tokenizer from checkpoint at inference** to avoid offset mismatch
- See CLAUDE.md for tokenizer bug details

**Voice cloning inference (after training):**
```python
# Strip baked embeddings if testing context encoder in isolation
model.baked_context_embedding = None
model._baked_embedding_T = None
model._baked_embedding_D = None
model.baked_context_embedding_len = None

# Load checkpoint with trained context encoder
model.load_state_dict(ckpt['state_dict'], strict=False)

# Pass reference audio at inference
batch = {
    'text': tokens,
    'text_lens': token_lens,
    'context_audio': ref_waveform,  # (1, time) at 22050 Hz
    'context_audio_lens': ref_len,
    'context_sample_rate': 22050,
    'context_text_tokens': torch.zeros(1, 1),
    'context_text_tokens_lens': torch.zeros(1),
    'has_text_context': torch.tensor([False]),
}
```

---

## Section 3: Training Setup

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Max epochs** | 100 | 450h (12.8x more data than 35h baseline) requires more iterations to converge |
| **Early stopping patience** | 5 | Stop if validation loss doesn't improve for 5 checkpoints |
| **Batch size** | 48 per GPU × 2 = 96 effective | Fits in 90GB VRAM with bf16; standard for MagPIE |
| **Learning rate** | 2e-5 | Standard fine-tuning rate (10x lower than pretraining 2e-4) |
| **LR scheduler** | Exponential decay, γ=0.998/step | Gradual LR reduction over epochs |
| **Gradient clip** | 2.5 | Prevents training instability in transformers |
| **Precision** | bf16-mixed | Saves ~50% VRAM vs FP32, maintains numerical stability |
| **Warmup steps** | 500 | Linear warmup to full learning rate |

### Distributed Training

**Strategy:** DDP (Distributed Data Parallel) on 2 GPUs

```bash
# Environment setup
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

**Data loading:**
- `num_workers=2` for training (prevent thread exhaustion)
- `num_workers=1` for validation
- Resampling: Audio resampled to 22.05kHz (NanoCodec requirement)
- Codec tokens: Pre-computed before training (stored in manifest paths)

**GPU synchronization:**
- Sync gradients every step (standard DDP)
- AllReduce for loss averaging across GPUs
- No manual gradient accumulation (batch_size already large)

### Checkpointing Strategy

**Save configuration:**
- Top 5 checkpoints (by lowest validation loss)
- Last checkpoint (for resume capability)
- Save interval: Every epoch
- Resume: Automatic if training interrupted

**Checkpoint contents:**
- Model state dict
- Optimizer state
- Scheduler state
- Hyperparameter config (for inference time tokenizer loading)

---

## Section 4: Validation Strategy

### During Training

**Per epoch validation:**
- Compute validation loss on 2% validation split (~3,600 segments)
- Track loss curve for early stopping detection
- Save top 5 checkpoints by validation loss
- Stop training if no improvement for 5 consecutive checkpoints

### End-of-Training Evaluation (All 5 Checkpoints)

**1. In-distribution performance (40 training speakers):**

*Round-trip intelligibility test:*
- Generate audio for FLEURS Georgian test set (979 samples) using best checkpoint
- Transcribe generated audio with Meta omniASR-LLM-7B (trained at ~2% CER on Georgian)
- Compute Character Error Rate (CER) and Word Error Rate (WER)
- **Target:** CER < 3% (your 35h baseline achieved 2.16%)
- **Expected range with 450h:** 1.5-2.5% CER

*Speaker representation quality:*
- Extract baked embedding vectors for each training speaker
- Compute cosine similarity within/between speakers
- Verify: Same speaker embeddings cluster, different speakers separate
- Check: Gradients flowing through all 41 embeddings during training

*Inference speed validation:*
- Time inference with baked embeddings (should be < 1s for typical text)
- Time inference with context encoder (should be ~2-3s, slower but enables zero-shot)

**2. Zero-shot voice cloning (1 held-out speaker):**

*Generalization test:*
- Use context encoder pathway (strip baked embeddings temporarily)
- Provide 5-10 seconds of reference audio from held-out speaker
- Generate speech on FLEURS test samples using held-out speaker context
- Transcribe with omniASR-LLM-7B
- Compute CER/WER for zero-shot samples

*Success criteria:*
- **Zero-shot CER < 6%** (acceptable if within ~2% of training speaker average)
- If achieved: Context encoder learned general speaker representation (not memorization)
- If failed: Context encoder may need longer training or more speaker diversity

**3. Qualitative evaluation:**

*Listening tests:*
- Sample 10-20 generated audio clips from training speakers (baked path)
- Sample 10-20 generated audio clips from held-out speaker (context encoder path)
- Listen for: Naturalness, prosody (pauses, intonation), speaker identity clarity

*Optional MOS (Mean Opinion Score):*
- If time/resources permit, collect 3-5 listeners
- Rate naturalness on 1-5 scale
- Rate speaker identity match on 1-5 scale

---

## Section 5: Success Metrics

### Training Convergence

✅ **Convergence validation:**
- Training completes without NaN/Inf losses
- Early stopping engages (stops before epoch 100)
- Validation loss plateaus (shows learning, not overfitting)
- Both GPU processes synchronize properly (DDP health check)

### In-Distribution Quality (40 Training Speakers)

✅ **CER performance:**
- **Minimum acceptable:** CER < 5%
- **Target:** CER < 3%
- **Stretch goal:** CER < 2.5% (match or beat 35h baseline of 2.16%)

✅ **Model stability:**
- All 41 baked embeddings receive non-zero gradients
- Context encoder loss decreases over training
- No gradient explosions or collapse

### Zero-Shot Voice Cloning (1 Held-Out Speaker)

✅ **Generalization capability:**
- Context encoder produces usable embeddings for held-out speaker
- **Zero-shot CER < 6%** (within ~2% of training speaker average CER)
- Generated audio is intelligible (verified by ASR transcription)
- **Interpretation:** If this succeeds, context encoder learned speaker identity abstraction across 40 diverse speakers

### Post-Experiment A Decision Tree

**If CER < 3% AND zero-shot works (CER < 6%):**
- ✅ Hypothesis validated: High-quality data + joint training works
- → Proceed to Phase 2: DPO/GRPO preference optimization (expected to reduce CER by ~40%)
- → Data quality is not the limiting factor

**If CER 3-5% AND zero-shot works:**
- ⚠️ Acceptable results, but headroom for improvement
- → Evaluate Experiment B (all 1,384h data) vs jumping to Phase 2 (DPO/GRPO)
- → Trade-off: More data vs preference optimization

**If CER > 5% OR zero-shot fails:**
- ❌ Something wrong with training setup
- → Debug before running Experiments B/C
- → Check: Tokenizer offsets, distributed training sync, learning rate scale, codec token paths

---

## Technical Implementation Notes

### Data Preparation Pipeline

1. **Manifest creation:** Already done (saba_clean/train_manifest.json, eval_manifest.json)
2. **Audio resampling:** 22.05kHz (handled by train.py Stage 1)
3. **Codec token extraction:** Pre-compute NanoCodec tokens (handled by train.py Stage 2)
4. **NeMo manifest conversion:** Convert to NeMo format with absolute paths (handled by train.py Stage 3)

### NeMo Configuration (Hydra)

Base: `conf/magpietts_georgian.yaml`

Key overrides for Experiment A:
```bash
--config-name=magpietts_georgian
max_epochs=100
batch_size=48
+model.optim.lr=2e-5
trainer.devices=2
trainer.strategy=ddp
trainer.patience=5
trainer.check_val_every_n_epoch=1
trainer.log_every_n_steps=50
+init_from_pretrained_model=nvidia/magpie_tts_multilingual_357m
exp_manager.exp_dir=./exp/magpie_tts_georgian_saba_450h
exp_manager.name=experiment_a_punctuated_joint_training
```

### NeMo Local Patches Required

See CLAUDE.md for details:
1. `nemo/utils/callbacks/nemo_model_checkpoint.py:240` — `weights_only=False`
2. `nemo/core/classes/modelPT.py:1414,1428` — `weights_only=False`
3. `examples/tts/magpietts.py:75-92` — `freeze_for_cloning` support

---

## Timeline & Resources

| Phase | Duration | Resource | Output |
|-------|----------|----------|--------|
| Data prep (resampling + codec) | 2-3 days | 2 GPUs | Codec tokens, manifests |
| Training (100 epochs, ~80 actual with early stop) | 2-3 weeks | 2 GPUs (DDP) | 5 best checkpoints |
| Evaluation (CER, zero-shot, MOS) | 2-3 days | 1 GPU | CER%, zero-shot success/fail |
| **Total** | **~3 weeks** | **2 GPUs** | **Full validation of hypothesis** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Tokenizer offset mismatch | Load tokenizer from checkpoint at inference (see Section 2 notes) |
| DDP synchronization issues | Run sanity check: verify loss reduces on both GPUs, all-reduce working |
| Memory OOM with batch_size=48 | Start with batch_size=32, increase if VRAM available |
| Zero-shot fails on held-out speaker | Might indicate context encoder needs more speaker diversity (then run Exp B) |
| Early stopping triggers too early | Increase patience from 5 to 7-10 if validation loss is still trending down |
| Training diverges | Check learning rate (2e-5 is standard); if loss increases, halve LR |

---

## Appendix: Comparison to Previous Training (35h baseline)

| Metric | 35h Baseline | Experiment A (450h) | Expected Improvement |
|--------|--------------|-------------------|----------------------|
| CER | 2.16% | < 3% (target < 2.5%) | -15% relative error |
| Training time | ~1 week | ~2-3 weeks | Due to 12.8x data size |
| Convergence epoch | ~40 | ~60-80 (with early stop) | More data = more iterations |
| Speakers | 12 | 40 (+ 1 held-out) | 3.3x more speaker diversity |
| Zero-shot capability | Not tested | Tested on held-out speaker | New capability validation |

---

## Sign-off

- **Design reviewed:** ✅
- **Data strategy:** ✅ Approved
- **Model configuration:** ✅ Approved
- **Training setup:** ✅ Approved
- **Validation strategy:** ✅ Approved
- **Success metrics:** ✅ Approved

Ready for implementation planning phase.