# TTS Pipelines — Complete Project Context

## What This Is
The **first Georgian TTS benchmark** — a comparative study of 6 open-source TTS architectures for Georgian (low-resource language). Each pipeline fine-tunes a different model on the same Common Voice Georgian dataset and evaluates on the same FLEURS Georgian benchmark with the same metrics. The result is an open-source framework + paper/report.

## Final Model Lineup (pipelines/)

| # | Pipeline | Architecture | Params | Voice Cloning | Fine-tuning Method | License (Weights) |
|---|----------|-------------|--------|---------------|-------------------|-------------------|
| 1 | **F5-TTS** | Non-AR flow matching (DiT) | 335M | Yes (reference audio) | Full fine-tune via repo CLI | CC-BY-NC-4.0 |
| 2 | **CosyVoice 3** | LLM + conditional flow matching | 0.5B | Yes (3s prompt) | Full SFT (llm + flow + hifigan) | Apache 2.0 |
| 3 | **Orpheus** | Pure LLM (Llama 3.2 + SNAC) | 3B | Yes (prompt) | LoRA via Unsloth | Apache 2.0 |
| 4 | **OpenAudio S1-mini** | Dual AR (LLAMA + DAC codec) | 0.5B | Yes (reference audio) | LoRA on LLAMA component | CC-BY-NC-SA-4.0 |
| 5 | **Qwen3-TTS** | Multi-codebook LM | 0.6B / 1.7B | Yes (reference audio) | Full SFT via official scripts | Apache 2.0 |
| 6 | **CSM-1B** | Llama + Mimi codec | 1B | No (multi-speaker) | LoRA via Unsloth | Apache 2.0 |

### Models DROPPED and why
- **XTTS v2** — Coqui shut down December 2025. Dead project, no maintenance, restrictive license.
- **VITS/MMS** — Legacy quality (~30M params). Far below 2025-2026 SOTA.

### Models ADDED and why
- **Qwen3-TTS** (January 2026) — Alibaba/Qwen. Discrete multi-codebook LM, trained on 5M+ hours, 10 languages. Apache 2.0.
- **OpenAudio S1-mini** (Fish Speech) — Highest open-source TTS Arena ELO as of early 2026.

### Model UPGRADED
- **CosyVoice 2 -> CosyVoice 3** — `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, released December 2025, trained on 1M+ hours.

## File Structure Per Pipeline
Each pipeline has: `README.md`, `config.py`, `train.py`, `infer.py`, `evaluate.py`, `generate_report.py`, `requirements.txt`

## Shared Code (shared/)
- **shared/data/** — Download (S3), prepare (unified manifest format), fixed deterministic splits
- **shared/evaluation/** — CER (Meta Omnilingual ASR), UTMOS, speaker similarity (ECAPA-TDNN, voice-cloning only), FAD (VGGish)

---

## Data

### Training Data
- **Source**: Mozilla Common Voice Georgian (~71k WAV files, 24kHz)
- **Storage**: S3 bucket `ttsopensource`, region `eu-central-1`
- **Files on S3**: `audio.rar` (12.1GB), `voice_actor_manifest.json`, `alignments.json`
- **After extraction**: `audio/` directory with WAV files, renamed from `clips_24k/`

### Data Quality Pipeline (TODO — implement in shared/data/)
Common Voice data is noisy. Before training, apply 6-stage Emilia-inspired filtering:

1. **Standardize** — Resample to 24kHz mono, normalize loudness to -23 LUFS
2. **DNSMOS filter** — Microsoft DNSMOS P.835 model, threshold >= 3.0 (drops noisy recordings)
   - Package: `pip install onnxruntime`, model from Microsoft DNS Challenge
3. **VAD trim** — Silero VAD to trim leading/trailing silence, drop clips with >50% silence
   - Package: `torch.hub.load('snakers4/silero-vad', 'silero_vad')`
4. **SNR filter** — Waveform-based SNR estimation, threshold >= 15 dB
5. **Transcript verification** — Round-trip ASR (Meta Omnilingual) -> CER against original text, drop if CER > 0.20
   - This catches wrong transcripts, code-switching, and garbage audio
6. **Speaker selection** — ECAPA-TDNN speaker clustering, keep top speakers with most data

Expected: ~71k -> ~50-55k samples after filtering (keep ~75%)

### Evaluation Benchmark
- **Source**: FLEURS Georgian (~400 test samples, different speakers AND different text from training data)
- **Load via**: `datasets.load_dataset("google/fleurs", "ka_ge", split="test")`
- **Why FLEURS**: Only standardized Georgian speech dataset with professional recordings. Different speakers/text from Common Voice ensures no data leakage. ~400 samples is enough for statistical significance.
- **No existing Georgian TTS benchmark exists** — this project creates the first one.

### Data Splits (for training data)
- Deterministic hash-based: 5% test, 5% val, 90% train
- Same split regardless of file ordering (uses MD5 hash of clip ID)
- All pipelines MUST use `shared.data.get_splits()` for fair comparison

---

## Evaluation

### Core Principle
All metrics are **round-trip or reference-free** — no matched same-speaker reference audio needed. This is critical because training speakers (Common Voice) != evaluation speakers (FLEURS).

### Metrics

#### 1. CER (Primary metric — Intelligibility)
- **Method**: Round-trip — TTS generates audio -> Meta Omnilingual ASR 7B transcribes it -> compare transcription to original text via Character Error Rate
- **ASR model**: `omnilingual-asr` package, model `omniASR_LLM_7B_v2`, language code `kat_Geor`
- **ASR quality**: 1.9% CER on Georgian (SOTA)
- **Install**: `pip install omnilingual-asr`

**WARNING**: Whisper is CATASTROPHICALLY BAD for Georgian — 78-88% WER. NEVER use Whisper for Georgian ASR. NVIDIA FastConformer is decent (5.7% WER) but Meta Omnilingual is far better.

#### 2. UTMOS (Naturalness)
- **Method**: Automatic MOS prediction model
- **Load via**: `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`
- **Input**: 16kHz audio
- **CAVEAT**: UTMOS is trained on English data only. Use for **relative ranking between models only**, NOT absolute quality. This caveat MUST appear in all reports.

#### 3. FAD (Distribution quality)
- **Method**: Frechet Audio Distance using VGGish embeddings
- **Compares**: Distribution of generated audio vs distribution of real Georgian speech (FLEURS)
- **Language-agnostic** — VGGish captures acoustic features, not language content
- **Needs**: >= 10 samples per set for statistical validity

#### 4. Speaker Similarity (Voice-cloning models ONLY)
- **Method**: ECAPA-TDNN cosine similarity via SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`)
- **What it measures**: Whether the generated voice matches the voice prompt that was provided
- **ONLY valid for**: F5-TTS, CosyVoice 3, Orpheus, OpenAudio S1-mini, Qwen3-TTS
- **NOT valid for**: CSM-1B (multi-speaker, no voice cloning condition)
- **NOT valid for**: Cross-speaker comparison (comparing generated vs unrelated reference)

### Metrics EXCLUDED and why
- **MCD (Mel Cepstral Distortion)** — Requires same-speaker reference audio. We don't have matched pairs (training speakers != test speakers). Would measure speaker difference, not synthesis quality.
- **PESQ** — Same problem as MCD. Designed for same-speaker degraded vs clean comparison.
- **Whisper-based CER** — Catastrophically bad on Georgian (78-88% WER). Would make every model look terrible regardless of actual quality.

---

## Fine-tuning Details Per Model (VERIFIED)

---

### 1. F5-TTS (SWivid/F5-TTS)

- **GitHub**: https://github.com/SWivid/F5-TTS
- **HuggingFace**: `SWivid/F5-TTS`
- **Checkpoint for fine-tuning**: `F5TTS_v1_Base_no_zero_init/model_1250000.safetensors` (recommended starting point)
- **Install**: `pip install f5-tts` (v1.1.17+) OR clone repo + `pip install -e .`
- **License**: Code: MIT, **Weights: CC-BY-NC-4.0**

#### Fine-tuning method
**Full fine-tune only. No LoRA/adapter support.**
Continuation of pretraining with flow-matching loss. NOT SFT in the LLM sense.

#### Data format
Pipe-delimited CSV:
```
audio_file|text
/path/to/audio_0001.wav|The transcription.
```
Audio: 24kHz WAV, optimal 3-12 seconds per clip. Minimum ~40 hours recommended.

#### Commands
```bash
# Step 1: Prepare data (generates raw.arrow, duration.json, vocab.txt)
python src/f5_tts/train/datasets/prepare_csv_wavs.py /path/to/metadata.csv /path/to/output_dir

# Step 2: Fine-tune
python src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name my_dataset \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 3200 \
    --max_samples 64 \
    --epochs 100 \
    --num_warmup_updates 500 \
    --save_per_updates 50000 \
    --finetune \
    --pretrain /path/to/F5TTS_v1_Base_no_zero_init/model_1250000.safetensors \
    --tokenizer char \
    --logger wandb

# OR use Gradio UI:
f5-tts_finetune-gradio
```

#### Key details
- **Config**: `src/f5_tts/configs/F5TTS_v1_Base.yaml` (dim=1024, depth=22, heads=16)
- **GPU**: RTX 4090 (24GB) works with batch_size ~3200-4000 frames
- **Use `--bnb_optimizer`** for 8-bit Adam to reduce VRAM
- **EMA warning**: `use_ema=True` can hurt early checkpoints — disable for eval
- **Recommended**: LR 1e-5, warmup 500 steps (not 20,000 as in pretraining)
- **Tokenizer**: Use `char` for Georgian (not `pinyin` which is Chinese-specific)

---

### 2. CosyVoice 3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

- **GitHub**: https://github.com/FunAudioLLM/CosyVoice
- **HuggingFace**: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (also RL variant: `Fun-CosyVoice3-0.5B-2512_RL`)
- **Install**: Clone repo + `pip install -r requirements.txt` (NO pip package)
- **System deps**: `sudo apt-get install sox libsox-dev`
- **License**: Apache 2.0 (both code and weights)

#### Fine-tuning method
**Full SFT — trains 3 sub-models sequentially: llm, flow, hifigan.** No LoRA.

#### Data format
Kaldi-style files per split:
- `wav.scp` — `<utt_id> <path_to_wav>`
- `text` — `<utt_id> <transcript>`
- `utt2spk` — `<utt_id> <speaker_id>`
- `spk2utt` — `<speaker_id> <utt_id1> <utt_id2> ...`

#### Pipeline (6 stages)
```bash
# Download model
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# Stage 0: Prepare Kaldi-style data files
python local/prepare_data.py --src_dir $data_dir --des_dir data/train

# Stage 1: Extract speaker embeddings (CampPlus ONNX)
python tools/extract_embedding.py --dir data/train --onnx_path $model_dir/campplus.onnx

# Stage 2: Extract discrete speech tokens
python tools/extract_speech_token.py --dir data/train --onnx_path $model_dir/speech_tokenizer_v3.onnx

# Stage 3: Convert to Parquet
python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 10 \
    --src_dir data/train --des_dir data/train/parquet

# Stage 5: Train all 3 sub-models
for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
        cosyvoice/bin/train.py \
        --train_engine torch_ddp \
        --config conf/cosyvoice2.yaml \
        --train_data data/train.data.list \
        --cv_data data/dev.data.list \
        --model $model \
        --checkpoint $model_dir/$model.pt \
        --model_dir $(pwd)/exp/cosyvoice3/$model \
        --use_amp
done

# Stage 6: Average best 5 checkpoints
python cosyvoice/bin/average_model.py --dst_model $decode_ckpt \
    --src_path $(pwd)/exp/cosyvoice3/$model --num 5 --val_best
```

#### Key details
- **Sample rate**: CosyVoice 3 uses **22,050 Hz** (v2 was 24kHz)
- **Config**: `conf/cosyvoice2.yaml` — LR 1e-5, warmup 2500 steps, max 200 epochs, grad clip 5
- **GPU**: 4x GPUs recommended. DeepSpeed ZeRO Stage 2 supported. ~24GB VRAM per GPU.
- **Stages 1-2 can be skipped** — online feature extraction is supported but slower
- **Also supports**: DPO training (`--dpo` flag) and GRPO (reinforcement learning)

---

### 3. Orpheus (unsloth/orpheus-3b-0.1-ft) — LoRA via Unsloth

- **GitHub**: https://github.com/canopyai/Orpheus-TTS
- **HuggingFace**: `unsloth/orpheus-3b-0.1-ft` (Unsloth copy) or `canopylabs/orpheus-3b-0.1-ft` (official)
- **Install**: `pip install --upgrade unsloth unsloth_zoo`
- **License**: Apache 2.0
- **Colab**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb

#### Fine-tuning method
**LoRA via Unsloth** on the Llama 3.2 backbone. SNAC codec (24kHz) used for audio tokenization only.

#### Data format
Dataset with `text` and `audio` columns. Audio resampled to 24kHz. SNAC tokenization + sequence formatting handled by Unsloth notebook code.

Input sequence: `[start_human] + text_tokens + [end_human] + [start_ai] + [start_speech] + snac_codes + [end_speech] + [end_ai]`

Text can include emotion tags: `<laugh>`, `<sigh>`, `<cough>`, `<yawn>`, `<gasp>`, `<giggles>`

#### Code
```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    load_in_4bit=False,  # 16-bit LoRA recommended for TTS
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train with HuggingFace Trainer
from transformers import TrainingArguments, Trainer
trainer = Trainer(
    model=model, train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        warmup_steps=5, max_steps=60, learning_rate=2e-4,
        optim="adamw_8bit", output_dir="outputs",
    ),
)
trainer.train()
model.save_pretrained("lora_model")
```

#### Key details
- **Only 3B variant is publicly available** on HuggingFace. 1B/400M/150M mentioned but not released.
- **LoRA**: rank=64, alpha=64, targets all attention + MLP projections
- **Sample rate**: 24kHz (SNAC codec)
- **Example dataset**: `MrDragonFox/Elise` on HuggingFace

---

### 4. OpenAudio S1-mini (fishaudio/openaudio-s1-mini) — formerly Fish Speech

- **GitHub**: https://github.com/fishaudio/fish-speech (rebranded to OpenAudio)
- **HuggingFace**: `fishaudio/openaudio-s1-mini` (0.5B, gated model)
- **Fine-tuning docs**: https://speech.fish.audio/finetune/
- **Install**: Clone repo + `pip install -e .[cu129]` (NOT `pip install fish-speech`)
- **System deps**: `apt install portaudio19-dev libsox-dev ffmpeg`
- **License**: Code: Apache 2.0, **Weights: CC-BY-NC-SA-4.0** (non-commercial)

#### Fine-tuning method
**LoRA on LLAMA component only** (default: rank=8, alpha=16). Codec stays frozen.

#### Data format
Speaker subdirectories with `.wav` + `.lab` (plain text transcription) files:
```
data/SPK1/audio1.wav
data/SPK1/audio1.lab
```

#### Pipeline (3-step + merge)
```bash
# Download model
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Step 1: Extract VQ tokens
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"

# Step 2: Build protobuf dataset
python tools/llama/build_dataset.py \
    --input "data" --output "data/protos" --text-extension .lab --num-workers 16

# Step 3: LoRA fine-tune
python fish_speech/train.py --config-name text2semantic_finetune \
    project=georgian_fish_v1 \
    +lora@model.model.lora_config=r_8_alpha_16

# Step 4: Merge LoRA weights
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/openaudio-s1-mini \
    --lora-weight results/georgian_fish_v1/checkpoints/<best_step>.ckpt \
    --output checkpoints/openaudio-s1-mini-georgian/
```

#### Key details
- **Config**: `fish_speech/configs/text2semantic_finetune.yaml` (Hydra + PyTorch Lightning)
- **Defaults**: batch_size=4, max_steps=10000, save/val every 100 steps
- **Min GPU**: 12GB VRAM for inference, more for training
- **Known issue**: Some users report gibberish output (GitHub #1136). Earlier checkpoints often work better.
- **Windows**: Add `trainer.strategy.process_group_backend=gloo`

---

### 5. Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-Base or 1.7B-Base)

- **GitHub**: https://github.com/QwenLM/Qwen3-TTS
- **HuggingFace models**:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — lighter, for fine-tuning
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — larger, for fine-tuning
  - `Qwen/Qwen3-TTS-Tokenizer-12Hz` — audio tokenizer (required)
  - Also: `*-CustomVoice` (9 built-in voices), `*-VoiceDesign` (text-described voices)
- **Install**: `pip install -U qwen-tts` OR clone repo + `pip install -e .`
- **Optional**: `pip install flash-attn` (reduces VRAM)
- **License**: Apache 2.0 (both code and weights)
- **Architecture**: Discrete multi-codebook LM (2048 codebook, 16 quantizers, 12Hz frame rate). NOT DiT.

#### Fine-tuning method
**Full SFT** via official `sft_12hz.py`. Community LoRA repos exist but are not official.

#### Data format
JSONL file:
```json
{"audio": "path/to/audio.wav", "text": "transcript", "ref_audio": "path/to/reference_speaker.wav"}
```
Audio: 24kHz WAV. Same `ref_audio` across all samples for speaker consistency. **Single-speaker fine-tuning only** (multi-speaker planned).

#### Commands
```bash
# Step 1: Tokenize audio into codes
python finetuning/prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl

# Step 2: SFT training
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output_model_path output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 32 \
    --lr 2e-6 \
    --num_epochs 10 \
    --speaker_name georgian_speaker
```

#### Key details
- **Trained on**: 5M+ hours, 10 languages (zh, en, ja, ko, de, fr, ru, pt, es, it)
- **Token rate**: 12Hz (12.5Hz technically)
- **Output**: 24kHz mono
- **Precision**: bfloat16 recommended
- **Fine-tuning is experimental** — official docs recommend zero-shot voice cloning (3s reference) for best results
- **Community LoRA**: https://github.com/cheeweijie/qwen3-tts-lora-finetuning (lora_scale=0.3)

---

### 6. CSM-1B (sesame/csm-1b) — LoRA via Unsloth

- **GitHub**: https://github.com/SesameAILabs/csm
- **HuggingFace**: `sesame/csm-1b` (gated) or `unsloth/csm-1b` (Unsloth copy)
- **Install**: `pip install --upgrade unsloth unsloth_zoo` + requires `transformers>=4.52.1`
- **License**: Apache 2.0
- **Colab**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)-TTS.ipynb

#### Fine-tuning method
**LoRA via Unsloth** on the Llama backbone. Mimi codec MUST stay frozen (`model.codec_model.eval()`). Decoder trains on 1/16 random subset of frames (compute amortization).

For **new languages or major domain shifts**, full-weight fine-tuning is recommended over LoRA (per Speechmatics blog guide).

#### Data format
Dataset with `text`, `audio` (24kHz), and `source` (speaker ID) columns.
Uses `processor.apply_chat_template()` for formatting.

#### Code
```python
from unsloth import FastModel
from transformers import CsmForConditionalGeneration, AutoProcessor
import torch

# Load model
model, processor = FastModel.from_pretrained(
    model_name="unsloth/csm-1b",
    max_seq_length=2048,
    auto_model=CsmForConditionalGeneration,
    load_in_4bit=False,
)

# Apply LoRA
model = FastModel.get_peft_model(
    model,
    r=32, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)

# Preprocess: use processor.apply_chat_template() with conversation format
# Train with HuggingFace Trainer (same pattern as Orpheus)
```

#### Key details
- **Architecture**: 1B Llama backbone (codebook 0 prediction) + 100M decoder (codebooks 1-31) + Mimi codec (frozen)
- **Sample rate**: 24kHz
- **CRITICAL**: Always `model.codec_model.eval()` during training
- **LoRA**: rank=32, alpha=32 (Unsloth default)
- **For Georgian (new language)**: Consider full fine-tune instead of LoRA — LoRA better for voice cloning within known languages
- **Speechmatics guide**: https://blog.speechmatics.com/sesame-finetune

---

## Fairness Note

Different models use different fine-tuning methods (LoRA vs full, Unsloth vs custom pipelines). This is **intentional and itself a finding**. The report should document for each model:
- Fine-tuning method (LoRA rank, full fine-tune, etc.)
- Compute used (GPU hours, GPU type)
- Training hyperparameters
- Number of trainable parameters

The "unfairness" of some models being easier to fine-tune is part of the story — a model that achieves great results with LoRA on a single GPU is more practical than one that needs full fine-tuning on 8 GPUs.

---

## Key Design Principles
- Each pipeline is **self-contained and independently runnable**
- All pipelines share the same evaluation metrics and benchmark data
- Someone can pick one pipeline, follow its README, and train/evaluate without touching anything else
- New pipelines can be added by following the existing pattern
- The framework is language-agnostic in design — swap the data and ASR model to benchmark any language

## Remote Training
Managed by separate repo: **Training_Agent** (chat-powered Vast.ai GPU orchestrator)

## What's Implemented vs TODO

### Done
- Repository structure with all 6 pipeline directories (f5_tts, cosyvoice, orpheus, fish_speech, qwen3_tts, csm_1b)
- Shared data download/prepare/splits code
- Shared evaluation code (CER, UTMOS, FAD, speaker similarity)
- Pipeline scaffolds (config, train, infer, evaluate, generate_report) with correct imports and CLI args
- Fish Speech train.py has working data format converter (prepare_lab_files)
- Top-level README, CLAUDE.md, requirements.txt, .gitignore

### TODO
- **Implement data quality pipeline**: `shared/data/quality.py` with the 6-stage filtering
- **Implement train.py for each pipeline**: Wire up actual model loading and training calls
- **Implement infer.py for each pipeline**: Wire up actual inference
- **Update CosyVoice sample rate**: v3 uses 22,050 Hz (not 24kHz)
- **Update Qwen3-TTS pipeline files**: Correct architecture description (multi-codebook LM, not DiT)
- **Write comparative report template**: `report/` directory with cross-model comparison
