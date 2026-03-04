# F5-TTS Pipeline

Fine-tuning [F5-TTS](https://github.com/SWivid/F5-TTS) (335M params) for Georgian TTS.

## Architecture

- **Type:** Non-autoregressive, flow matching with Diffusion Transformer (DiT) + ConvNeXt V2
- **Key feature:** No duration model, text encoder, or phoneme alignment needed — text is padded with filler tokens to match speech length
- **Inference speedup:** Sway Sampling
- **Pre-trained on:** 100K hours multilingual data (Emilia dataset)
- **Cross-lingual:** Proven transfer to unseen languages (German, French, Hindi, Korean)

## Why this model

F5-TTS represents the state-of-the-art in non-autoregressive TTS. Its flow matching approach and proven cross-lingual transfer make it a strong candidate for Georgian despite never being trained on it.

## Setup

```bash
pip install -r requirements.txt
```

## Data preparation

```bash
# Download shared data (run from repo root)
python -m shared.data.download --output-dir ./data

# Or if data is already downloaded, just ensure ./data/ exists with:
#   - audio/          (WAV files)
#   - alignment/voice_actor_manifest.json
```

## Training

```bash
python train.py --data-dir ./data --run-name georgian_f5_v1
```

See `config.py` for all hyperparameters.

## Inference

```bash
python infer.py \
    --checkpoint checkpoints/georgian_f5_v1/best.pt \
    --text "გამარჯობა მსოფლიო" \
    --output output.wav
```

## Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/georgian_f5_v1/best.pt \
    --data-dir ./data \
    --output-dir results/
```

## Generate report

```bash
python generate_report.py --results-dir results/ --output report.md
```

## References

- [F5-TTS Paper](https://arxiv.org/abs/2410.06885)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579)
