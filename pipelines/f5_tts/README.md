# F5-TTS Georgian Pipeline

Fine-tuning [F5-TTS](https://github.com/SWivid/F5-TTS) (335M params) for Georgian text-to-speech with zero-shot voice cloning.

**Model on HuggingFace:** [NMikka/F5-TTS-Georgian](https://huggingface.co/NMikka/F5-TTS-Georgian)

## Results

### FLEURS Georgian Benchmark (979 unseen samples, unseen speakers)

| Metric | Value |
|--------|-------|
| **CER mean** | **5.09%** |
| CER median | 3.09% |
| CER p90 | 11.83% |
| WER mean | 18.66% |

65.9% of samples achieve < 5% CER. Zero catastrophic failures (> 50%).

Evaluated with round-trip ASR: F5-TTS generates audio → Meta Omnilingual ASR 7B transcribes → CER against original text.

**Voice cloning note:** The model clones training speakers (Common Voice Georgian) well, but zero-shot voice cloning for arbitrary Georgian speakers is not reliable yet. This is an active area of improvement.

### In-Domain (5 Georgian test sentences, speaker 3)

| Metric | Value |
|--------|-------|
| CER | 2.55% |
| MCD-DTW | 6.53 dB |
| Speaker similarity (ECAPA-TDNN) | 0.716 |

## Architecture

- **Type:** Non-autoregressive flow matching with Diffusion Transformer (DiT) + ConvNeXt V2
- **Key feature:** No duration model, text encoder, or phoneme alignment needed — text is padded with filler tokens to match speech length
- **Inference:** Sway Sampling for speedup
- **Pre-trained on:** 100K hours multilingual data (Emilia dataset)
- **Cross-lingual:** Proven transfer to unseen languages (German, French, Hindi, Korean)

## Prerequisites

- **GPU:** NVIDIA GPU with 48GB+ VRAM (tested on RTX A6000). 24GB GPUs work with reduced batch size.
- **Python:** 3.12+
- **OS:** Linux (tested on Ubuntu)

```bash
pip install -r requirements.txt
```

## Reproducing the Full Training Pipeline

### Step 0: Data

Training data comes from the [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) dataset — 20,300 cleaned samples from Mozilla Common Voice Georgian, 12 speakers, 24kHz WAV.

The dataset was prepared by a 6-stage quality pipeline (NISQA MOS filtering, VAD trimming, SNR filtering, ASR transcript verification, speaker selection). See `shared/data/DATASET.md` for details.

From the repository root:

```bash
# Download and prepare data
python -m shared.data.download --output-dir ./data
```

This downloads from HuggingFace (`NMikka/Common-Voice-Geo-Cleaned`) and creates `data/clean/` with:
- `audio/` — 24kHz WAV files
- `train_manifest.json` — JSONL with `{id, audio_filepath, text, speaker_id, duration}`
- `eval_manifest.json` — 1,001 held-out samples for evaluation
- `speaker_refs_manifest.json` — Best reference clips per speaker (for voice cloning eval)

### Step 1: Vocabulary Extension

F5-TTS uses a character-level tokenizer. The pretrained model has a pinyin-based vocabulary of 2,545 characters covering CJK, Latin, Arabic, Cyrillic, etc. — but no Georgian.

We extend the vocabulary by appending 34 Georgian Unicode characters (ა-ჰ + „):

```
pretrained_vocab.txt  →  2,545 characters (original F5-TTS)
extended_vocab.txt    →  2,579 characters (+ 34 Georgian)
```

The 34 added characters:
```
ა ბ გ დ ე ვ ზ თ ი კ ლ მ ნ ო პ ჟ რ ს ტ უ ფ ქ ღ ყ შ ჩ ც ძ წ ჭ ხ ჯ ჰ „
```

The `extended_vocab.txt` file is already provided in `data/`. The training script handles vocab extension automatically.

### Step 2: Dataset Preparation

The training script converts the JSONL manifest to F5-TTS's pipe-delimited CSV format and creates an Arrow dataset:

```bash
cd pipelines/f5_tts
python train.py --data-dir ../../data/clean --prepare-only
```

This produces:
- `data/train.csv` — Pipe-delimited file: `audio_file|text` (20,300 entries)
- `data/georgian_tts_char/raw.arrow` — Arrow binary dataset
- `data/georgian_tts_char/duration.json` — Audio durations
- `data/georgian_tts_char/vocab.txt` — Copy of `extended_vocab.txt`

### Step 3: Checkpoint Preparation

The training script automatically:
1. Downloads the pretrained checkpoint `F5TTS_v1_Base/model_1250000.safetensors` from HuggingFace
2. Resizes the text embedding layer from 2,546 → 2,580 dimensions (to accommodate new Georgian tokens)
3. Initializes new Georgian character embeddings with the **mean of existing pretrained embeddings**
4. Saves the extended checkpoint as `ckpts/georgian_tts/pretrained_model_1250000_extended.safetensors`

This is the starting point for fine-tuning — the model already knows how to generate speech in many languages, and we're teaching it Georgian characters.

### Step 4: Training

```bash
python train.py --data-dir ../../data/clean
```

**Hyperparameters** (see `config.py`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 | Fine-tuning rate (pretraining uses 7.5e-5) |
| Warmup steps | 500 | Fine-tuning warmup (pretraining uses 20,000) |
| Batch size | 9,600 frames/GPU | Dynamic batching by audio frames, not samples |
| Max sequences/batch | 64 | Upper bound on sequences per batch |
| Epochs | 100 | ~1,200 update steps per epoch |
| Optimizer | 8-bit Adam (bitsandbytes) | Saves ~8GB VRAM |
| Gradient clipping | 1.0 | Max gradient norm |
| Gradient accumulation | 1 | No accumulation needed |
| Tokenizer | `char` | **Must be `char`, not `pinyin`** |
| Save checkpoints | Every 10,000 updates | ~3.2GB per checkpoint |

**Key training decisions:**
- **`use_ema=False` for inference** — EMA can hurt quality on fine-tuned checkpoints since the EMA weights lag behind. We use non-EMA weights for generation.
- **`char` tokenizer** — The default `pinyin` tokenizer is Chinese-specific. `char` treats each character as a token, which works for Georgian.
- **Frame-based batching** — F5-TTS batches by total audio frames (not sample count), so each batch has roughly equal compute regardless of utterance length.

**Resume from interruption** — just re-run the same command. The trainer automatically detects and resumes from the latest checkpoint.

**Custom settings:**
```bash
# Lower VRAM (24GB GPU)
python train.py --data-dir ../../data/clean --batch-size 4000

# Different learning rate
python train.py --data-dir ../../data/clean --lr 5e-6

# Without W&B logging
python train.py --data-dir ../../data/clean --no-wandb
```

**Training time:** ~18 minutes per epoch, ~30 hours total for 100 epochs on a single RTX A6000 (48GB).

### Step 5: Speaker Selection

After training, evaluate which speaker reference produces the best voice cloning:

```bash
python eval_speakers.py
```

This generates 5 Georgian test sentences for each of the 12 speakers across checkpoints, and computes ECAPA-TDNN speaker similarity. Results are saved to `eval_speaker_results/results.json`.

**Best speaker per checkpoint (model_110000):**

| Speaker | Similarity | CER |
|---------|-----------|-----|
| **14** | **0.788** | 2.54% |
| **3** | **0.716** | 2.55% |
| 4 | 0.708 | 6.92% |
| 12 | 0.672 | 4.11% |

Speaker 3 was selected for FLEURS evaluation after manually listening to the top 2 speakers by similarity (14 and 3) — speaker 3 sounded best overall.

### Step 6: FLEURS Evaluation (CER)

```bash
python eval_fleurs.py
```

This runs the full evaluation pipeline:
1. Loads FLEURS Georgian test set (979 samples)
2. Generates audio for all samples using model_110000 + speaker 3 reference
3. Transcribes generated audio via Meta Omnilingual ASR 7B (runs in `/root/asr_env` subprocess with fairseq2)
4. Computes CER and WER against original FLEURS transcriptions

Results are saved to `eval_fleurs/results.json`.

**Note:** The ASR model (omniASR_LLM_7B, ~30GB) requires a separate Python environment with `torch>=2.8` and `fairseq2>=0.6`. It runs as a subprocess via `/root/asr_env/bin/python3`.

## Inference

### Single utterance

```bash
python infer.py \
    --text "გამარჯობა, როგორ ხარ?" \
    --checkpoint ckpts/georgian_tts/model_110000.pt \
    --ref-audio path/to/reference.wav \
    --ref-text "რეფერენს ტექსტი"
```

### Batch evaluation mode

```bash
python infer.py --eval \
    --checkpoint ckpts/georgian_tts/model_110000.pt \
    --ref-audio path/to/reference.wav \
    --ref-text "რეფერენს ტექსტი" \
    --manifest ../../data/clean/eval_manifest.json \
    --output-dir results/generated/
```

### Python API

```python
from f5_tts.api import F5TTS
import soundfile as sf

model = F5TTS(
    ckpt_file="ckpts/georgian_tts/model_110000.pt",
    vocab_file="data/extended_vocab.txt",
    device="cuda",
    use_ema=False,
)

wav, sr, _ = model.infer(
    ref_file="reference.wav",
    ref_text="რეფერენს ტექსტი",
    gen_text="საქართველო მდებარეობს კავკასიის რეგიონში.",
)
sf.write("output.wav", wav, sr)
```

## File Structure

```
pipelines/f5_tts/
├── config.py              # Training hyperparameters
├── train.py               # Data preparation + training
├── infer.py               # Single and batch inference
├── evaluate.py            # Full evaluation pipeline
├── eval_speakers.py       # Speaker selection evaluation
├── eval_fleurs.py         # FLEURS CER benchmark
├── eval_cer_mcd.py        # CER + MCD computation (uses ASR subprocess)
├── upload_model.py        # Push model to HuggingFace
├── generate_report.py     # Report generation from results
├── MODEL_CARD.md          # HuggingFace model card
├── requirements.txt       # Dependencies
├── data/
│   ├── train.csv                    # Pipe-delimited manifest (20,300 samples)
│   ├── pretrained_vocab.txt         # Original F5-TTS vocab (2,545 chars)
│   ├── extended_vocab.txt           # Extended vocab (2,579 chars)
│   └── georgian_tts_char/           # Prepared Arrow dataset
│       ├── raw.arrow
│       ├── duration.json
│       └── vocab.txt
├── ckpts/georgian_tts/
│   ├── model_100000.pt              # Checkpoint at 100k updates
│   ├── model_110000.pt              # Final checkpoint at 110k updates
│   └── pretrained_model_1250000_extended.safetensors
├── eval_fleurs/                     # FLEURS evaluation outputs
│   ├── results.json                 # CER/WER results
│   ├── generated/                   # Generated audio (979 WAVs)
│   └── fleurs_audio/               # FLEURS reference audio
└── eval_speaker_results/            # Speaker evaluation outputs
    ├── results.json                 # Speaker similarity results
    ├── full_results.json            # CER + MCD + similarity
    ├── model_100000/                # Generated audio per speaker
    └── model_110000/
```

## References

- [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579)
- [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) — Training dataset
- [NMikka/F5-TTS-Georgian](https://huggingface.co/NMikka/F5-TTS-Georgian) — Fine-tuned model
