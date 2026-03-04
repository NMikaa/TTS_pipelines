# Qwen3-TTS Pipeline

Fine-tuning [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) (~500M params) for Georgian TTS.

## Architecture

- **Type:** LLM (Qwen3 backbone) + DiT vocoder
- **Parameters:** ~500M
- **Key feature:** Strong multilingual capabilities from Qwen3 foundation, reference-audio voice cloning
- **Released:** January 2026
- **License:** Apache 2.0

## Why this model

Qwen3-TTS represents Alibaba's latest TTS architecture built on their strong Qwen3 LLM foundation. LLM-based TTS with excellent multilingual transfer and modern tooling. Replaces XTTS v2 (Coqui shut down Dec 2025).

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir ./data --run-name georgian_qwen3tts_v1
```

## Inference

```bash
python infer.py --checkpoint checkpoints/best.pt --text "გამარჯობა მსოფლიო" --output output.wav
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data-dir ./data --output-dir results/
```

## Generate report

```bash
python generate_report.py --results-dir results/ --output report.md
```

## References

- [Qwen3-TTS HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
