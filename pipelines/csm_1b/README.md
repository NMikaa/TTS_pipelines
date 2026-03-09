# CSM-1B Pipeline

Fine-tuning [CSM-1B](https://huggingface.co/sesame/csm-1b) (1B params) for Georgian TTS using Unsloth + HuggingFace Trainer.

## Architecture

- **Type:** Llama backbone + Mimi audio codec decoder
- **Parameters:** ~1B (58M trainable with LoRA)
- **Fine-tuning:** LoRA (r=64, alpha=64) via Unsloth on all attention + MLP projections
- **Sample rate:** 24 kHz
- **License:** Apache 2.0

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (batch size, LoRA, LR, etc.) |
| `dataset.py` | Data loading and preprocessing — converts JSONL manifests to CSM training format |
| `callbacks.py` | W&B callback — logs reference vs generated audio table on each eval |
| `train.py` | Training script — model loading, LoRA, Trainer orchestration |
| `infer.py` | Inference — single text or batch from manifest |
| `evaluate.py` | Full evaluation pipeline (CER, MCD, ECAPA-TDNN speaker sim) |
| `eval_checkpoints.py` | Checkpoint comparison across all metrics |
| `eval_fleurs.py` | FLEURS Georgian benchmark evaluation (CER/WER) |
| `eval_speaker_sim.py` | Speaker similarity evaluation |
| `REPORT.md` | Evaluation results and methodology |
| `generate_report.py` | Markdown report from evaluation results |

## Setup

```bash
pip install -r requirements.txt
```

## Data format

JSONL manifest with one entry per line:
```json
{"id": "clip_001", "audio_path": "data/audio/clip_001.wav", "text": "გამარჯობა", "speaker_id": "spk_01", "duration_sec": 3.2}
```

To convert your own data, use `dataset.py` directly:
```python
from dataset import load_manifest, preprocess_example
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("sesame/csm-1b")
entries = load_manifest("my_manifest.json")
result = preprocess_example(entries[0], processor)
# result has: input_ids, attention_mask, labels, input_values, input_values_cutoffs, text, speaker_id
```

## Training

```bash
# Default config (batch=64, lr=5e-5, 15 epochs)
python train.py --data-dir ../../data/clean

# Custom hyperparameters
python train.py --data-dir ../../data/clean --lr 2e-4 --num-epochs 20 --batch-size 48

# Quick test (5 steps)
python train.py --data-dir ../../data/clean --max-steps 5

# Resume from checkpoint
python train.py --data-dir ../../data/clean --resume checkpoints/checkpoint-500
```

## Inference

```bash
# Single utterance
python infer.py --checkpoint checkpoints/final --text "გამარჯობა მსოფლიო"

# Batch from manifest
python infer.py --checkpoint checkpoints/final --eval-manifest ../../data/clean/eval_manifest.json --output-dir outputs/
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/final --data-dir ../../data/clean --output-dir results/
python generate_report.py --results-dir results/
```

## Environment variables

```bash
export HF_TOKEN=...           # HuggingFace token (csm-1b is gated)
export WANDB_API_KEY=...      # Weights & Biases
export WANDB_PROJECT=georgian-tts
```

## References

- [CSM-1B HuggingFace](https://huggingface.co/sesame/csm-1b)
- [Unsloth CSM Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)_TTS.ipynb)
- [Speechmatics Fine-tuning Guide](https://blog.speechmatics.com/sesame-finetune)
