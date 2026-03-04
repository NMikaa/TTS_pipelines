# OpenAudio S1-mini (Fish Speech) Pipeline

Fine-tuning [OpenAudio S1-mini](https://huggingface.co/fishaudio/openaudio-s1-mini) (0.5B params) for Georgian TTS.

## Architecture

- **Type:** Dual autoregressive — LLAMA text-to-semantic + DAC codec
- **Parameters:** 0.5B (distilled from S1)
- **Key feature:** Highest open-source TTS Arena ELO (early 2026), LoRA fine-tuning on LLAMA component
- **Voice cloning:** Yes, from reference audio prompt at inference time
- **License:** Code: Apache 2.0, **Weights: CC-BY-NC-SA-4.0** (non-commercial)

## Why this model

OpenAudio S1-mini (formerly Fish Speech) has the highest ELO on the open-source TTS Arena. Its dual AR architecture is architecturally distinct from the other models in the benchmark. LoRA fine-tuning on just the LLAMA component is efficient. Replaces VITS/MMS (legacy quality).

## Setup

```bash
# Clone Fish Speech repo (rebranded as OpenAudio)
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
pip install -e .[cu129]  # or cu126 for CUDA 12.6

# System deps (Linux)
apt install portaudio19-dev libsox-dev ffmpeg

# Download model checkpoint
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## Data Format

Fish Speech expects speaker subdirectories with audio + `.lab` text files:
```
data/
├── SPK1/
│   ├── audio1.wav
│   ├── audio1.lab    # plain text transcription
│   ├── audio2.wav
│   └── audio2.lab
└── SPK2/
    └── ...
```

Optional: normalize loudness first with `fap loudness-norm data-raw data --clean`

## Training (3-step pipeline)

### Step 1: Extract semantic tokens (VQ)
```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

### Step 2: Pack dataset into protobuf format
```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

### Step 3: LoRA fine-tune
```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=georgian_fish_v1 \
    +lora@model.model.lora_config=r_8_alpha_16
```

### Step 4: Merge LoRA weights for inference
```bash
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/openaudio-s1-mini \
    --lora-weight results/georgian_fish_v1/checkpoints/step_000000010.ckpt \
    --output checkpoints/openaudio-s1-mini-georgian/
```

**Important notes:**
- On Windows, add `trainer.strategy.process_group_backend=gloo`
- Earlier checkpoints often perform better — try multiple checkpoints
- Default LoRA: rank=8, alpha=16. Config at `fish_speech/configs/text2semantic_finetune.yaml`
- Min GPU: 12GB VRAM (inference), more for training

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/openaudio-s1-mini-georgian --data-dir ./data --output-dir results/
```

## Generate report

```bash
python generate_report.py --results-dir results/ --output report.md
```

## Known Issues

- Some users report gibberish output after fine-tuning (GitHub #1136, unresolved)
- The model learns speech patterns, not timbre by default — still need reference audio for voice cloning at inference
- Increasing training steps to learn timbre risks overfitting

## References

- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [Official Fine-tuning Docs](https://speech.fish.audio/finetune/)
- [OpenAudio S1-mini HuggingFace](https://huggingface.co/fishaudio/openaudio-s1-mini)
