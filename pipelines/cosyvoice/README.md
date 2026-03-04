# CosyVoice 3 Pipeline

Fine-tuning [CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice) (0.5B params) for Georgian TTS.

## Architecture

- **Type:** LLM (text -> semantic tokens) + conditional flow matching with DiT (tokens -> speech)
- **Key feature:** Supervised semantic tokens from multilingual ASR encoder with FSQ
- **Pre-trained on:** 1M+ hours multilingual data (v3, December 2025)
- **Voice cloning:** Yes, from 3-second prompt
- **License:** Apache 2.0

## Why this model

CosyVoice 3 represents the LLM + flow matching hybrid approach backed by Alibaba. Trained on 1M+ hours (up from 200K in v2), with ASR-based semantic tokens that capture phonetic structure well. Apache 2.0 license makes it fully open-source friendly.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir ./data --run-name georgian_cosyvoice_v1
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

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice 2 Paper](https://arxiv.org/abs/2412.10117)
