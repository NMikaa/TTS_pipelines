# TTS Pipelines — Project Context

## What This Is
Georgian text-to-speech training pipelines. Two model architectures fine-tuned on Common Voice Georgian (40k samples, 71k audio files total).

## Pipelines

### Pocket TTS (Kyutai CALM-based, ~100M params)
- Entry point: `python train.py`
- Fine-tunes FlowLM (flow matching head) using LSD (Lagrangian Self-Distillation) loss
- Trains on pre-computed Mimi latents (audio → Mimi encoder → input_proj → 32-dim latents)
- Package: `pocket_tts_training/` (config, dataset, lsd_loss, trainer with EMA + AMP + cosine LR)
- Upstream library: `pocket-tts/` (git submodule from kyutai-labs/pocket-tts)
- Model config: `pocket-tts/pocket_tts/config/b6369a24.yaml` — ldim=32, d_model=1024, flow_dim=512, 6 layers, 16 heads, n_bins=4000, sample_rate=24000, frame_rate=12.5

### CSM-1B (Sesame)
- Entry point: `python train_csm.py`
- Fine-tunes with LoRA (r=128, alpha=32) via Unsloth + HuggingFace Trainer
- 2-word look-ahead streaming training scheme with word-level alignment
- Package: `csm_training/` (config, alignment, dataset, trainer, inference)
- Inference: `python -m csm_training.inference --text "..." --context-audio ref.wav --output out.wav`
  - KV-cache streaming: backbone StaticCache (4096) + depth decoder StaticCache (32)
  - Word-by-word text injection on EOS detection

## Data
- Audio: Common Voice Georgian, resampled to 24kHz (`clips_24k/`)
- Manifest: `alignment/voice_actor_manifest.json` — JSONL with audio_filepath, text, source
- Alignments: `alignments.json` — word-level timing from NeMo Forced Aligner (CTM files)
- Latents: `latents_cache/` — pre-computed via `scripts/precompute_latents.py`
  - 40,769 samples, 0 skipped
  - Stats: min=28 frames, max=133, mean=73.5
  - Includes `normalization_stats.pt` (emb_mean, emb_std from pretrained FlowLM)

## S3 (bucket: ttsopensource, region: eu-central-1)
All training data is on S3 under prefix `tts-georgian/`:
- `audio.rar` (12.1 GB) — internal folder: `clips_24k/`, needs rename to `audio/` on extraction
- `latents_cache.rar` (374 MB)
- `alignments.json` (43 MB)
- `alignment/voice_actor_manifest.json` (11 MB, uses relative paths: `audio/<file>.wav`)
- `quantizer_input_proj_weight.safetensors` (0.1 MB)

## Key Architectural Details

### CSM-1B Dataset (dataset.py)
- `TTSTrainingDataset`: wraps HF Dataset with shifted audio context
- `_build_messages()`: creates chat format — context turn (masked from loss) + word-by-word turns
- First message: 3 words of text + first word audio
- Subsequent: " " + next look-ahead word + that word's audio
- Context audio loss masked via `_mask_first_audio_run()`
- Audio constants: FRAME=1920, TARGET_SR=24000, MAX_AUDIO=24000*120, MAX_TEXT=4096

### CSM-1B Alignment (alignment.py)
- `AlignmentProcessor`: processes CTM files from NeMo Forced Aligner
- `adjust_word_boundaries_weighted()`: distributes gaps inversely proportional to word duration
- `merge_short_words()`: merges words < 80ms with adjacent words
- The `alignment/` folder (nfa_temp/, outputs/) is intermediate data — NOT needed for training

### Pocket TTS Training (trainer.py)
- EMA (decay=0.999), AMP (autocast bfloat16/float16), cosine LR
- LSD loss: flow matching with linear interpolation, SimpleMLPAdaLN flow head
- Mimi codec frozen (weights from pre-computed latents, not loaded at training time)

## Utility Scripts
- `scripts/precompute_latents.py` — Encode audio through Mimi + input_proj, save as .pt
- `scripts/generate_alignments.py` — Process NFA CTM outputs into alignments.json
- `scripts/process_alignments.py` — Alignment pre-processing
- `scripts/monitor_progress.py` — Training progress monitoring
- `scripts/renumber_speakers.py` — Speaker ID remapping

## Remote Training
Managed by separate repo: NMikaa/Training_Agent (chat-powered Vast.ai agent)
