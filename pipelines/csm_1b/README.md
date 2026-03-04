# CSM-1B Pipeline

Fine-tuning [CSM-1B](https://github.com/SesameAILabs/csm) (1B params) for Georgian TTS.

## Architecture

- **Type:** Llama backbone + Mimi audio codec decoder
- **Parameters:** ~1B
- **Key feature:** Conversational speech generation — designed for dialogue, not just read speech
- **Fine-tuning:** LoRA via Unsloth + HuggingFace Trainer, or full fine-tune
- **Pre-trained on:** Primarily English
- **License:** Apache 2.0

## Why this model

CSM-1B is the only model in this benchmark designed specifically for conversational speech. It represents the Llama + neural codec approach. While English-centric, its Apache 2.0 license and HuggingFace Trainer compatibility make it practical for research.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# LoRA fine-tuning (recommended, less VRAM)
python train.py --data-dir ./data --run-name georgian_csm_v1

# Full fine-tuning (more VRAM)
python train.py --data-dir ./data --full-finetune --run-name georgian_csm_full
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

- [CSM-1B GitHub](https://github.com/SesameAILabs/csm)
- [CSM-1B HuggingFace](https://huggingface.co/sesame/csm-1b)
- [Sesame Fine-tuning Guide](https://blog.speechmatics.com/sesame-finetune)
