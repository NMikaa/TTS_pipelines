# Orpheus TTS Pipeline

Fine-tuning [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) for Georgian TTS.

## Architecture

- **Type:** Pure autoregressive LLM (Llama backbone)
- **Sizes:** 150M, 400M, 1B, 3B
- **Key feature:** Extends Llama with audio token generation — pure language modeling approach
- **Fine-tuning:** LoRA or full fine-tune, Unsloth integration
- **Pre-trained on:** English, Spanish, French, German, Italian, Portuguese, Chinese, Hindi, Korean
- **License:** Apache 2.0

## Why this model

Orpheus represents the "pure LLM" approach to TTS — treating speech synthesis as a language modeling problem. It has explicit guidance for extending to new languages, multiple model sizes for compute/quality tradeoffs, and a permissive license.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir ./data --run-name georgian_orpheus_v1
python train.py --data-dir ./data --run-name georgian_orpheus_v1 --model-size 1b  # larger model
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

- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [Orpheus Multilingual](https://canopylabs.ai/releases/orpheus_can_speak_any_language)
