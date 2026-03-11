# MagPIE TTS — Georgian Fine-tuning Pipeline

Fine-tune [NVIDIA MagPIE TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) (357M) for Georgian text-to-speech synthesis using NVIDIA NeMo.

**Trained model**: [NMikka/Magpie-TTS-Geo-357m](https://huggingface.co/NMikka/Magpie-TTS-Geo-357m)

## Results

Evaluated on the full [FLEURS Georgian](https://huggingface.co/datasets/google/fleurs) test set (979 samples):

| Metric | Score |
|--------|-------|
| **CER** | **2.16%** |
| **WER** | **7.08%** |

> Round-trip evaluation: TTS generates audio -> [Meta Omnilingual ASR 7B](https://huggingface.co/facebook/omniASR-LLM-7B) transcribes -> compare to original text. The ASR itself has ~1.9% CER on Georgian, so the TTS is near-perfectly intelligible.

## Architecture

MagPIE TTS is an **encoder-decoder transformer** with CTC monotonic alignment:

```
Text -> [ByT5-small] -> [6-layer Encoder] -> [CTC Alignment] -> [12-layer Decoder] -> [NanoCodec] -> Waveform
```

- **ByT5-small** (byte-level text encoder) — handles Georgian natively, no phoneme conversion
- **CTC monotonic alignment** — prevents hallucinations (no skipped/repeated words)
- **NanoCodec** — 22kHz, 8 codebooks, 21.5 fps, 1.89 kbps
- **Classifier-Free Guidance (CFG)** — two forward passes for better text adherence
- **5 baked speaker embeddings** from pretraining (speaker 1 sounds best for Georgian)

## File Structure

```
pipelines/magpie_tts/
├── README.md              # This file
├── config.py              # Training hyperparameters
├── train.py               # Full training pipeline (resample + codec + train)
├── generate.py            # Inference script (generate_speech API)
├── evaluate.py            # FLEURS evaluation (CER/WER)
├── infer.py               # Legacy inference scaffold
├── generate_report.py     # Report generation scaffold
├── requirements.txt       # Python dependencies
├── MODEL_CARD.md          # HuggingFace model card
├── conf/
│   ├── magpietts_georgian.yaml    # Hydra config for NeMo training
│   └── pretrained_dicts/          # Phoneme dictionaries from pretrained model
│       ├── ipa_cmudict-0.7b_nv23.01.txt
│       ├── heteronyms-052722
│       ├── es_ES_nv230301.dict
│       ├── de_nv230119.dict
│       ├── de_nv230119.heteronym
│       └── ipa_dict_nv23.05.txt
└── results_final/
    └── evaluation.json    # Full 979-sample evaluation results
```

## Quick Start: Inference

```bash
# MagPIE TTS requires NeMo 2.8+ (install from source)
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && pip install -e ".[tts]"
pip install huggingface_hub
```

```python
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

# Download and load model
nemo_path = hf_hub_download(repo_id="NMikka/Magpie-TTS-Geo-357m", filename="magpie_tts_georgian.nemo")
model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
model = model.eval().cuda()

# Synthesize
text = "გამარჯობა, მე მქვია მაგპაი და ქართულად ვლაპარაკობ."

chunked_tokens, chunked_tokens_len, _ = chunk_text_for_inference(
    text=text,
    language="ka",
    tokenizer_name="text_ce_tokenizer",
    text_tokenizer=model.tokenizer,
    eos_token_id=model.eos_id,
)

chunk_state = model.create_chunk_state(batch_size=1)
all_codes = []

for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
    batch = {
        "text": toks.unsqueeze(0).cuda(),
        "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
        "speaker_indices": 1,  # speaker 1 is best for Georgian
    }
    with torch.no_grad():
        output = model.generate_speech(
            batch,
            chunk_state=chunk_state,
            end_of_text=[i == len(chunked_tokens) - 1],
            beginning_of_text=(i == 0),
            use_cfg=True,
            use_local_transformer_for_inference=True,
        )
    if output.predicted_codes_lens[0] > 0:
        all_codes.append(output.predicted_codes[0, :, :output.predicted_codes_lens[0]])

# Decode to waveform
codes = torch.cat(all_codes, dim=1).unsqueeze(0)
codes_lens = torch.tensor([codes.shape[2]], device="cuda", dtype=torch.long)
audio, audio_lens, _ = model.codes_to_audio(codes, codes_lens)
waveform = audio[0, :audio_lens[0]].cpu().float().unsqueeze(0)

torchaudio.save("output.wav", waveform, 22050)
```

Or use our `generate.py` CLI:

```bash
python generate.py --text "გამარჯობა, როგორ ხარ?" --speakers 1
python generate.py --text "გამარჯობა" --no-cfg          # 2x faster
python generate.py --text "გამარჯობა" --fast             # fastest (no CFG + torch.compile)
```

## Training Pipeline

### Prerequisites

- GPU with **48GB VRAM** (A6000, A100). Reduce `batch_size` to 16-32 for 24GB GPUs.
- NeMo 2.8+ installed from source
- Training data: [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned)

### Step 1: Prepare Data + Train

The `train.py` script handles everything in one command:

```bash
python train.py --data-dir ../../data/clean
```

This runs 4 stages automatically:
1. **Resample** all audio from 24kHz to 22,050 Hz (NanoCodec requirement)
2. **Pre-compute NanoCodec tokens** (`.pt` files with shape `[8, num_frames]`)
3. **Convert manifests** to NeMo JSONL format with absolute paths
4. **Fine-tune** via NeMo's `magpietts.py` with Hydra overrides

Or prepare data only:
```bash
python train.py --data-dir ../../data/clean --prepare-only
```

### Step 2: Configure

Edit `config.py` or pass CLI overrides:

```bash
python train.py --data-dir ../../data/clean --lr 2e-5 --batch-size 48 --epochs 100
```

Key hyperparameters (from `config.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 2e-5 | 10x lower than pretraining default (2e-4) |
| `batch_size` | 48 | ~40GB VRAM on A6000 with bf16 |
| `max_epochs` | 100 | We got best checkpoint at epoch 37 |
| `precision` | bf16-mixed | Half VRAM, negligible quality loss |
| `grad_clip_val` | 2.5 | Prevents training instability |
| `warmup_steps` | 500 | Linear warmup |

### Step 3: Hydra Config (Critical)

The Hydra config at `conf/magpietts_georgian.yaml` defines the tokenizer vocabulary. **You MUST include all tokenizers from the pretrained model** to keep `text_embedding.weight` shape at 2362 tokens.

Before training, update the absolute paths in the YAML:
```bash
sed -i "s|/ABSOLUTE/PATH/TO/pretrained_dicts|$(pwd)/conf/pretrained_dicts|g" conf/magpietts_georgian.yaml
```

Then copy this config to NeMo's config directory:
```bash
cp conf/magpietts_georgian.yaml /path/to/NeMo/examples/tts/conf/magpietts/
```

### Step 4: Select Best Checkpoint

```bash
# Generate 10 FLEURS samples per checkpoint and compare CER
python evaluate.py --checkpoint /path/to/checkpoint.ckpt --speaker 1 --num-samples 10
```

### Step 5: Full Evaluation

```bash
python evaluate.py --checkpoint /path/to/best.ckpt --speaker 1 --output-dir results_final
```

### Data Format

**Input manifest** (JSONL, one entry per line):
```json
{"id": "clip_001", "audio_path": "data/audio/clip_001.wav", "text": "ტექსტი აქ", "speaker_id": "0", "duration": 5.2}
```

**NeMo manifest** (generated by `train.py` Step 3):
```json
{"audio_filepath": "/abs/path/clip_001.wav", "text": "ტექსტი აქ", "duration": 5.2, "speaker": 0, "target_audio_codes_path": "/abs/path/clip_001.pt"}
```

## The Tokenizer Offset Bug (Lessons Learned)

This section documents a critical bug we encountered and solved. If you're fine-tuning MagPIE TTS for a new language, read this carefully.

### The Problem

After training, our model produced **perfect predictions during validation** (teacher-forced) but **complete gibberish during inference** (autoregressive). Georgian text like "გამარჯობა" would generate noise or wrong sounds.

### Root Cause

MagPIE TTS uses a **combined vocabulary** from multiple tokenizers (English IPA, Spanish IPA, German IPA, Mandarin, Japanese, French, Hindi, Italian, Vietnamese, and ByT5 `text_ce_tokenizer`). Each tokenizer occupies a fixed offset range in the 2362-token embedding table:

```
Token offsets in pretrained model:
  english_phoneme:     0 - 95
  spanish_phoneme:    96 - 191
  german_phoneme:   192 - 318
  mandarin_phoneme: 319 - 654
  japanese_phoneme: 655 - 859
  ...
  text_ce_tokenizer: 1976 - 2361  <-- Georgian goes through THIS tokenizer
```

During training, our Hydra config listed the tokenizers in a **different order** than the pretrained model. This shifted `text_ce_tokenizer` to a different offset (e.g., offset 96 instead of 1976). Training proceeded normally because the model learned the new offsets. But at inference time, if you load the pretrained model and then overlay the fine-tuned weights, the **pretrained tokenizer** is used — with the original offsets. Georgian token IDs that should map to offset 1976+ now map to offset 96+, hitting Spanish/German phoneme embeddings instead.

### The Fix

When loading for inference, **always replace the tokenizer with the one from the training checkpoint**:

```python
# Load pretrained model
model = MagpieTTSModel.from_pretrained('nvidia/magpie_tts_multilingual_357m')

# Load fine-tuned weights
ckpt = torch.load('checkpoint.ckpt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)

# CRITICAL: Use training config's tokenizer (correct offsets)
training_cfg = ckpt['hyper_parameters']['cfg']
model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)
```

Or use the `.nemo` export (which bakes in the correct config):
```python
model = MagpieTTSModel.restore_from("magpie_tts_georgian.nemo")  # offsets are correct
```

### How to Avoid This

1. **Match the pretrained model's tokenizer order exactly** in your Hydra config. Extract the config from the `.nemo` archive and replicate it.
2. **Always export to `.nemo` format** for distribution — it bundles the correct config with the weights.
3. **Test inference after training**, not just validation loss. Teacher-forced validation won't catch offset bugs.

## Known Issues

### `do_tts()` / `infer_batch()` Repetition Bug

NeMo's built-in `infer_batch` has a [known issue](https://github.com/NVIDIA-NeMo/NeMo/issues/15300) where it doesn't handle EOS correctly during autoregressive decoding. This causes word repetitions like "იცოდედედედეთ" instead of "იცოდეთ".

**Workaround**: Use `generate_speech()` (chunked inference) instead. This is what `generate.py` uses. The `do_tts()` convenience method internally calls `infer_batch` and is therefore also affected.

### NeMo Version

MagPIE TTS was added in NeMo 2.8 (release candidate as of March 2026). It's not yet available via `pip install nemo_toolkit`. Install from source:

```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && pip install -e ".[tts]"
```

## Evaluation

Evaluation uses round-trip CER/WER:

1. Generate audio for all 979 FLEURS Georgian test samples
2. Transcribe with [Meta Omnilingual ASR 7B](https://huggingface.co/facebook/omniASR-LLM-7B) (1.9% CER on Georgian, SOTA)
3. Compute Character Error Rate and Word Error Rate vs original text


**Why not Whisper?** Whisper scores 78-88% WER on Georgian. Catastrophically bad. Never use Whisper for Georgian ASR.

```bash
# Install ASR
pip install omnilingual-asr

# Run evaluation
python evaluate.py \
    --checkpoint /path/to/best.ckpt \
    --speaker 1 \
    --output-dir results_final
```

## Training Details

| | |
|---|---|
| **Base model** | nvidia/magpie_tts_multilingual_357m (357M params) |
| **Method** | Full SFT (all parameters trainable) |
| **Data** | [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) (~71k clips) |
| **Best checkpoint** | step 15,614, epoch 37, val_loss 9.5569 |
| **Learning rate** | 2e-5 with 500-step warmup |
| **Precision** | bf16-mixed |
| **GPU** | 1x NVIDIA A6000 (48GB) |
| **Sample rate** | 22,050 Hz (NanoCodec) |
| **Codec** | NanoCodec (8 codebooks, 21.5 fps) |
