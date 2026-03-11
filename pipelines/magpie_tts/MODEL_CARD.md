---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
language:
- ka
tags:
- tts
- text-to-speech
- georgian
- nemo
- magpie-tts
base_model: nvidia/magpie_tts_multilingual_357m
datasets:
- NMikka/Common-Voice-Geo-Cleaned
pipeline_tag: text-to-speech
---

# MagPIE TTS — Georgian

A fine-tuned [MagPIE TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) model for Georgian (ქართული) text-to-speech synthesis.

This is the **first open-source TTS model fine-tuned specifically for Georgian**, produced as part of the [Georgian TTS Benchmark](https://github.com/NikaGaworworw/TTS_pipelines) — a comparative study of 6 open-source TTS architectures on a low-resource language.

## Evaluation Results

Evaluated on the full [FLEURS Georgian](https://huggingface.co/datasets/google/fleurs) test set (979 samples) using round-trip intelligibility:

| Metric | Score |
|--------|-------|
| **CER** | **2.16%** |
| **WER** | **7.08%** |

> CER/WER measured via round-trip: TTS generates audio → [Meta Omnilingual ASR 7B](https://huggingface.co/facebook/omniASR-LLM-7B) transcribes it → compare to original text. The ASR model itself has ~1.9% CER on Georgian, meaning almost all measured "error" is ASR noise — the TTS output is near-perfectly intelligible.

## Quick Start

### Installation

```bash
# MagPIE TTS requires NeMo 2.8+ (not yet on PyPI — install from source)
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && pip install -e ".[tts]"
pip install huggingface_hub
```

> Requires Python 3.10+, PyTorch 2.0+, CUDA 11.8+

### Inference

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
        "speaker_indices": 1,  # speaker index (0-4)
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

### Convenience Wrapper

For easier use, here's a helper function:

```python
def synthesize(model, text, speaker=1, use_cfg=True):
    """Generate Georgian speech from text.

    Args:
        model: Loaded MagpieTTSModel
        text: Georgian text string
        speaker: Baked speaker index (0-4). Speaker 1 recommended.
        use_cfg: Use classifier-free guidance (better quality, 2x slower)

    Returns:
        waveform (torch.Tensor): Audio tensor, shape (1, num_samples), 22050 Hz
    """
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
            "speaker_indices": speaker,
        }
        with torch.no_grad():
            output = model.generate_speech(
                batch,
                chunk_state=chunk_state,
                end_of_text=[i == len(chunked_tokens) - 1],
                beginning_of_text=(i == 0),
                use_cfg=use_cfg,
                use_local_transformer_for_inference=True,
            )
        if output.predicted_codes_lens[0] > 0:
            all_codes.append(output.predicted_codes[0, :, :output.predicted_codes_lens[0]])

    if not all_codes:
        return None

    codes = torch.cat(all_codes, dim=1).unsqueeze(0)
    codes_lens = torch.tensor([codes.shape[2]], device="cuda", dtype=torch.long)
    audio, audio_lens, _ = model.codes_to_audio(codes, codes_lens)
    return audio[0, :audio_lens[0]].cpu().float().unsqueeze(0)


# Usage:
waveform = synthesize(model, "გამარჯობა მსოფლიო")
torchaudio.save("hello_world.wav", waveform, 22050)
```

## How It Works

MagPIE TTS is an **encoder-decoder transformer** (not a diffusion or flow model):

1. **ByT5-small** encodes text at the byte level — no language-specific tokenizer needed
2. **6-layer causal encoder** processes text embeddings
3. **CTC monotonic alignment** maps text to audio frames (prevents hallucinations — no skipped or repeated words)
4. **12-layer causal decoder** autoregressively generates NanoCodec tokens
5. **NanoCodec** (22kHz, 8 codebooks) decodes tokens to waveform

**Classifier-Free Guidance (CFG)** runs two forward passes (with/without text conditioning) and interpolates. Set `use_cfg=False` for ~2x faster inference with slightly lower quality.

## Speakers

The model has 5 baked speaker embeddings from pretraining. Set via `speaker_indices` in the batch dict.

| Index | Quality |
|-------|---------|
| **1** | **Best** (recommended) |
| 0 | Good |
| 2 | Acceptable |
| 3 | Poor |
| 4 | Poor |

## Parameters

You can tune inference parameters via `model.inference_parameters`:

```python
model.inference_parameters.temperature = 0.6    # sampling temperature (lower = more deterministic)
model.inference_parameters.topk = 80            # top-k sampling (lower = more focused)
model.inference_parameters.cfg_scale = 2.5      # CFG strength (higher = follows text more strictly)
model.inference_parameters.max_decoder_steps = 500  # max generation length in frames
```

## Training Details

| | |
|---|---|
| **Base model** | [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) |
| **Method** | Full SFT via NeMo |
| **Training data** | [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) (~71k clips, 24kHz, resampled to 22,050 Hz) |
| **Parameters** | 357M (all trainable) |
| **Epochs** | 37 |
| **Steps** | 15,614 |
| **Learning rate** | 2e-5 |
| **Precision** | bf16-mixed |
| **GPU** | 1x A6000 (48GB) |
| **Best val_loss** | 9.5569 |
| **Sample rate** | 22,050 Hz |
| **Codec** | NanoCodec (8 codebooks, 21.5 fps, 1.89 kbps) |

## How to Reproduce Training

This section documents every step needed to fine-tune MagPIE TTS on Georgian (or any language) from scratch.

### Prerequisites

```bash
# 1. Clone NeMo (MagPIE TTS requires NeMo 2.8+, not yet on PyPI)
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && pip install -e ".[tts]"

# 2. Install additional dependencies
pip install huggingface_hub torchaudio tqdm wandb

# 3. Hardware: GPU with 48GB VRAM (A6000, A100, etc.)
#    Training uses ~40GB VRAM at batch_size=48 with bf16-mixed precision
```

### Step 1: Prepare Your Data

Your training data should be a JSONL manifest where each line is:

```json
{"id": "clip_001", "audio_path": "data/audio/clip_001.wav", "text": "ტექსტი აქ", "speaker_id": "0", "duration": 5.2}
```

- **Audio format**: WAV, 24kHz mono (will be resampled to 22,050 Hz)
- **Text**: Raw text in target language (Georgian uses ByT5 byte-level tokenizer — no phoneme conversion needed)
- **speaker_id**: Speaker label (used for context audio pairing)
- **duration**: Clip duration in seconds

We used [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) — Mozilla Common Voice Georgian filtered for quality (~71k clips). The dataset includes NISQA MOS scores and ASR-verified transcripts.

You need two manifests: `train_manifest.json` and `eval_manifest.json`.

### Step 2: Resample Audio to 22,050 Hz

MagPIE TTS uses NanoCodec which requires 22,050 Hz audio (not 24kHz).

```python
import torchaudio
from pathlib import Path
from tqdm import tqdm

resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=22050)
input_dir = Path("data/audio")
output_dir = Path("data/audio_22khz")
output_dir.mkdir(exist_ok=True)

for wav_path in tqdm(sorted(input_dir.glob("*.wav"))):
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:  # stereo to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = resampler(waveform)
    torchaudio.save(str(output_dir / wav_path.name), waveform, 22050)
```

### Step 3: Pre-compute NanoCodec Tokens

Pre-computing codec tokens avoids redundant on-the-fly encoding each epoch. This produces `.pt` files with shape `[8, num_frames]` (8 codebooks).

```python
import torch
import torchaudio
from nemo.collections.tts.models import AudioCodecModel
from pathlib import Path
from tqdm import tqdm

codec = AudioCodecModel.from_pretrained("nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps").eval().cuda()

audio_dir = Path("data/audio_22khz")
codes_dir = Path("data/codec_codes")
codes_dir.mkdir(exist_ok=True)

for wav_path in tqdm(sorted(audio_dir.glob("*.wav"))):
    codes_path = codes_dir / f"{wav_path.stem}.pt"
    if codes_path.exists():
        continue

    waveform, sr = torchaudio.load(str(wav_path))
    waveform = waveform.squeeze(0).unsqueeze(0).cuda()  # (1, time)
    wav_len = torch.tensor([waveform.shape[-1]], device="cuda")

    with torch.no_grad():
        codes, codes_len = codec.encode(audio=waveform, audio_len=wav_len)
        codes = codes.squeeze(0).cpu()  # [8, num_frames]

    torch.save(codes, str(codes_path))
```

### Step 4: Convert Manifests to NeMo Format

NeMo's MagPIE TTS expects a specific JSONL format with absolute paths and codec token paths:

```python
import json
from pathlib import Path

def convert_manifest(input_path, output_path, audio_dir, codes_dir):
    audio_dir = Path(audio_dir).resolve()
    codes_dir = Path(codes_dir).resolve()

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            name = Path(entry["audio_path"]).name
            stem = Path(name).stem

            nemo_entry = {
                "audio_filepath": str(audio_dir / name),
                "text": entry["text"],
                "duration": entry.get("duration", 0.0),
                "speaker": int(entry.get("speaker_id", 0)),
                "target_audio_codes_path": str(codes_dir / f"{stem}.pt"),
            }
            f_out.write(json.dumps(nemo_entry, ensure_ascii=False) + "\n")

convert_manifest("train_manifest.json", "train_manifest_nemo.json", "data/audio_22khz", "data/codec_codes")
convert_manifest("eval_manifest.json", "eval_manifest_nemo.json", "data/audio_22khz", "data/codec_codes")
```

### Step 5: Extract Phoneme Dictionaries from Pretrained Model

The pretrained model bundles phoneme dictionaries for English, Spanish, German, Mandarin, Japanese, and French inside its `.nemo` archive. You must extract them to keep the tokenizer vocabulary aligned (2362 tokens). Without this, the `text_embedding.weight` shape won't match and inference will produce gibberish.

```bash
# Find the cached .nemo file
find ~/.cache/huggingface -name "magpie_tts_multilingual_357m.nemo" 2>/dev/null

# Extract phoneme dicts from the archive
mkdir -p pretrained_dicts
cd pretrained_dicts
python3 -c "
import tarfile, sys
nemo_path = sys.argv[1]  # path to magpie_tts_multilingual_357m.nemo
with tarfile.open(nemo_path, 'r') as tar:
    for member in tar.getmembers():
        if any(ext in member.name for ext in ['.dict', '.txt', 'heteronym']):
            # Strip hash prefix for cleaner filenames
            import re
            clean_name = re.sub(r'^[a-f0-9]{32}_', '', member.name)
            member.name = clean_name
            tar.extract(member)
            print(f'Extracted: {clean_name}')
" /path/to/magpie_tts_multilingual_357m.nemo
```

This should produce:
- `ipa_cmudict-0.7b_nv23.01.txt` (English)
- `heteronyms-052722` (English)
- `es_ES_nv230301.dict` (Spanish)
- `de_nv230119.dict` (German)
- `de_nv230119.heteronym` (German)
- `ipa_dict_nv23.05.txt` (Mandarin)

### Step 6: Create Hydra Config for Your Language

MagPIE TTS uses NeMo's Hydra config system. You need a custom config that includes **all tokenizers from the pretrained model** (to keep the vocabulary at 2362 tokens) and routes your language through `text_ce_tokenizer` (ByT5 byte-level tokenizer).

Save this as `NeMo/examples/tts/conf/magpietts/magpietts_georgian.yaml`:

```yaml
# Extends the base magpietts.yaml config
defaults:
  - magpietts

# Override text_tokenizers — MUST match pretrained model's full set
# so that text_embedding.weight shape matches (2362 tokens)
model:
  text_tokenizers:
    english_phoneme:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer
      punct: true
      apostrophe: true
      pad_with_space: false
      g2p:
        _target_: nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p
        phoneme_dict: "/absolute/path/to/pretrained_dicts/ipa_cmudict-0.7b_nv23.01.txt"
        heteronyms: "/absolute/path/to/pretrained_dicts/heteronyms-052722"
        phoneme_probability: 0.8
        ignore_ambiguous_words: false
        use_chars: true
        use_stresses: true
    spanish_phoneme:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer
      locale: "es-ES"
      punct: true
      apostrophe: true
      pad_with_space: true
      g2p:
        _target_: nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p
        locale: "es-ES"
        phoneme_dict: "/absolute/path/to/pretrained_dicts/es_ES_nv230301.dict"
        phoneme_probability: 0.8
        ignore_ambiguous_words: false
        use_chars: true
        use_stresses: true
    german_phoneme:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer
      locale: "de-DE"
      punct: true
      apostrophe: true
      pad_with_space: true
      g2p:
        _target_: nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p
        locale: "de-DE"
        phoneme_dict: "/absolute/path/to/pretrained_dicts/de_nv230119.dict"
        heteronyms: "/absolute/path/to/pretrained_dicts/de_nv230119.heteronym"
        phoneme_probability: 0.8
        ignore_ambiguous_words: false
        use_chars: true
        use_stresses: true
        grapheme_case: "mixed"
        grapheme_prefix: "#"
    mandarin_phoneme:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.ChinesePhonemesTokenizer
      punct: true
      apostrophe: true
      pad_with_space: true
      g2p:
        _target_: nemo.collections.tts.g2p.models.zh_cn_pinyin.ChineseG2p
        phoneme_dict: "/absolute/path/to/pretrained_dicts/ipa_dict_nv23.05.txt"
        word_segmenter: "jieba"
        phoneme_prefix: ""
        phoneme_case: "lower"
        tone_prefix: "#"
        ascii_letter_prefix: ""
        ascii_letter_case: "upper"
    japanese_phoneme:
      _target_: nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.JapanesePhonemeTokenizer
      punct: true
      apostrophe: false
      pad_with_space: true
      g2p:
        _target_: nemo.collections.tts.g2p.models.ja_jp_ipa.JapaneseKatakanaAccentG2p
        ascii_letter_prefix: ""
        ascii_letter_case: "upper"
    french_chartokenizer:
      _target_: AutoTokenizer
      pretrained_model_name: "google/byt5-small"
    text_ce_tokenizer:
      _target_: AutoTokenizer
      pretrained_model_name: "google/byt5-small"
```

> **Why all these tokenizers?** The pretrained model's `text_embedding.weight` has 2362 rows — one per token across all tokenizers. Each tokenizer occupies a fixed offset range. If you only include `text_ce_tokenizer`, the offsets shift and Georgian text maps to wrong embeddings, producing gibberish. You must include every tokenizer from the pretrained model in the same order.

### Step 7: Run Training

```bash
cd NeMo

python examples/tts/magpietts.py \
    --config-name=magpietts_georgian \
    max_epochs=100 \
    batch_size=48 \
    +train_ds_meta.georgian_train.manifest_path=/abs/path/train_manifest_nemo.json \
    +train_ds_meta.georgian_train.audio_dir=/abs/path/data/audio_22khz \
    +train_ds_meta.georgian_train.feature_dir=/abs/path/data/codec_codes \
    +train_ds_meta.georgian_train.sample_weight=1.0 \
    "+train_ds_meta.georgian_train.tokenizer_names=[text_ce_tokenizer]" \
    +val_ds_meta.georgian_eval.manifest_path=/abs/path/eval_manifest_nemo.json \
    +val_ds_meta.georgian_eval.audio_dir=/abs/path/data/audio_22khz \
    +val_ds_meta.georgian_eval.feature_dir=/abs/path/data/codec_codes \
    +val_ds_meta.georgian_eval.sample_weight=1.0 \
    "+val_ds_meta.georgian_eval.tokenizer_names=[text_ce_tokenizer]" \
    model.codecmodel_path=nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
    model.optim.lr=2e-5 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=2.5 \
    trainer.log_every_n_steps=10 \
    trainer.check_val_every_n_epoch=1 \
    trainer.strategy=auto \
    exp_manager.exp_dir=exp \
    exp_manager.name=magpie_tts_georgian \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    +init_from_pretrained_model=nvidia/magpie_tts_multilingual_357m \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project=georgian-tts \
    exp_manager.wandb_logger_kwargs.name=magpie-tts
```

**Key training arguments explained:**

| Argument | Value | Why |
|----------|-------|-----|
| `tokenizer_names=[text_ce_tokenizer]` | ByT5 tokenizer | Routes Georgian text through the byte-level tokenizer (not IPA phoneme tokenizers which don't support Georgian) |
| `model.optim.lr=2e-5` | 10x lower than pretraining | Prevents catastrophic forgetting of pretrained weights |
| `batch_size=48` | Frames per GPU | Uses ~40GB VRAM on A6000 with bf16. Reduce to 16-32 for 24GB GPUs |
| `trainer.precision=bf16-mixed` | BFloat16 | Halves VRAM usage with minimal quality loss |
| `trainer.gradient_clip_val=2.5` | Gradient clipping | Prevents training instability |
| `init_from_pretrained_model` | NVIDIA's pretrained weights | Starting point for fine-tuning |
| `exp_manager.resume_if_exists=true` | Auto-resume | Automatically resumes from last checkpoint if training is interrupted |

### Step 8: Select Best Checkpoint

Training saves checkpoints ranked by validation loss. We found that **lower val_loss doesn't always mean better audio quality**. We recommend:

1. Pick the top 5-6 checkpoints by val_loss
2. Generate 10 samples from each using the FLEURS test set
3. Run round-trip CER evaluation on each
4. Pick the one with best CER

Our checkpoint comparison (10 FLEURS samples each, speaker 1):

| Checkpoint (step) | Val Loss | CER | WER |
|-------------------|----------|-----|-----|
| 13926 | 9.5551 | — | — |
| **15614** | **9.5569** | **best** | — |
| 14348 | 9.5537 | — | — |
| 15192 | 9.5565 | — | — |
| 16880 | 9.5571 | — | — |
| 42200 (last) | 9.7229 | — | — |

Step 15614 was selected as the best overall based on CER evaluation.

### Step 9: Export to .nemo

To package the checkpoint as a portable `.nemo` file:

```python
import torch
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

# Load pretrained model (sets up correct architecture)
model = MagpieTTSModel.restore_from("path/to/magpie_tts_multilingual_357m.nemo", map_location="cpu")

# Load fine-tuned weights
ckpt = torch.load("path/to/best_checkpoint.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=False)

# Fix tokenizer offsets (use training config's tokenizer order)
training_cfg = ckpt["hyper_parameters"]["cfg"]
model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)
model.cfg.text_tokenizers = training_cfg.text_tokenizers

# Save
model.save_to("magpie_tts_georgian.nemo")
```

### Known Issues

- **`do_tts()` / `infer_batch()` repetition bug**: NeMo's built-in `infer_batch` has a [known issue](https://github.com/NVIDIA-NeMo/NeMo/issues/15300) where it doesn't handle EOS correctly, causing word repetitions (e.g., "იცოდედედედეთ"). Use `generate_speech()` (chunked inference) instead — it handles EOS properly.
- **Tokenizer offset mismatch**: The pretrained model and training config may place `text_ce_tokenizer` at different offsets in the combined vocabulary. The training checkpoint bakes in the correct offsets. When loading for inference, always use the tokenizer from the checkpoint's `hyper_parameters.cfg`, not from the pretrained model. See Step 9 above.

## Limitations

- **Single language**: Fine-tuned on Georgian only. The base model supports 105 languages but this checkpoint is specialized.
- **No voice cloning**: Uses 5 baked speaker embeddings from pretraining. Reference audio cloning was not trained.
- **Autoregressive**: Not real-time. RTF ~0.3-0.5 on A6000 with CFG, ~0.15-0.25 without.
- **NeMo dependency**: Requires NVIDIA NeMo toolkit. Not a standalone model.
- **NanoCodec dependency**: The codec model (`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) is downloaded automatically on first use.

## Citation

```bibtex
@misc{magpie-tts-georgian-2026,
  title={MagPIE TTS Georgian: Fine-tuned Text-to-Speech for Georgian},
  author={TODO},
  year={2026},
  url={https://huggingface.co/NMikka/Magpie-TTS-Geo-357m}
}
```

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the MagPIE TTS architecture and training framework
- [NMikka/Common-Voice-Geo-Cleaned](https://huggingface.co/datasets/NMikka/Common-Voice-Geo-Cleaned) for the cleaned Georgian speech dataset
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) for the evaluation benchmark
