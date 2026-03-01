"""
Pre-compute latents for Pocket TTS training.

Encodes all audio files through Mimi encoder + input_proj once,
saving the latent representations to disk. This removes the Mimi
encoder from the training loop, making training much faster.

Usage:
    python scripts/precompute_latents.py \
        --manifest alignment/voice_actor_manifest.json \
        --output latents_cache
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pocket-tts"))


def load_mimi_encoder(model_variant: str = "b6369a24"):
    """Load the Mimi encoder from pretrained Pocket TTS model."""
    from pocket_tts.models.tts_model import TTSModel

    print("Loading pretrained Pocket TTS model...")
    tts_model = TTSModel.load_model(config=model_variant)
    mimi = tts_model.mimi
    mimi.eval()
    return mimi, tts_model.flow_lm


def load_input_proj(weights_path: str):
    """Load input_proj weights for encoding 512-dim -> 32-dim."""
    from safetensors.torch import load_file

    state = load_file(weights_path)
    key = next(k for k in state.keys())
    weight = state[key]
    print(f"Loaded input_proj from {weights_path}, shape: {weight.shape}")

    proj = torch.nn.Conv1d(
        in_channels=weight.shape[1],
        out_channels=weight.shape[0],
        kernel_size=1,
        bias=False,
    )
    proj.weight.data = weight
    proj.eval()
    return proj


@torch.no_grad()
def encode_audio(
    audio_path: str,
    mimi: torch.nn.Module,
    input_proj: torch.nn.Module,
    sample_rate: int = 24000,
    max_audio_seconds: float = 15.0,
) -> torch.Tensor | None:
    """Encode a single audio file to latent representation.

    Returns: [S, 32] latent tensor, or None if file is too short/invalid.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        return None

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    # Skip too short (< 0.5s)
    if waveform.shape[1] < sample_rate // 2:
        return None

    # Trim to max length
    max_samples = int(max_audio_seconds * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # [1, 1, T] for Mimi
    audio = waveform.unsqueeze(0)

    # Mimi encode: [1, 1, T] -> [1, 512, S]
    encoded = mimi.encode_to_latent(audio)

    # input_proj: [1, 512, S] -> [1, 32, S]
    latent = input_proj(encoded)

    # Transpose to [S, 32]
    latent = latent.squeeze(0).transpose(0, 1)

    return latent


def main():
    parser = argparse.ArgumentParser(description="Pre-compute latents for training")
    parser.add_argument("--manifest", required=True, help="Path to voice_actor_manifest.json")
    parser.add_argument("--output", default="latents_cache", help="Output directory")
    parser.add_argument("--weights", default="quantizer_input_proj_weight.safetensors")
    parser.add_argument("--model-variant", default="b6369a24")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models (CPU to avoid CUDA issues with Mimi streaming)
    mimi, flow_lm = load_mimi_encoder(args.model_variant)
    input_proj = load_input_proj(args.weights)

    # Save emb_mean and emb_std from pretrained model (needed for training)
    torch.save(
        {"emb_mean": flow_lm.emb_mean.cpu(), "emb_std": flow_lm.emb_std.cpu()},
        output_dir / "normalization_stats.pt",
    )
    print(f"Saved normalization stats (emb_mean, emb_std) to {output_dir}/normalization_stats.pt")

    # Load manifest
    samples = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from manifest")

    # Encode all audio
    metadata = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc="Encoding audio")):
        audio_path = sample["audio_filepath"]
        latent = encode_audio(audio_path, mimi, input_proj, max_audio_seconds=args.max_seconds)

        if latent is None:
            skipped += 1
            continue

        # Save latent tensor
        latent_filename = f"{idx:06d}.pt"
        torch.save(latent, output_dir / latent_filename)

        metadata.append(
            {
                "latent_path": latent_filename,
                "text": sample["text"],
                "speaker_id": sample.get("source", "unknown"),
                "num_frames": latent.shape[0],
                "audio_filepath": audio_path,
            }
        )

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Encoded {len(metadata)} samples, skipped {skipped}")
    print(f"Latents saved to: {output_dir}/")
    print(f"Metadata saved to: {metadata_path}")

    # Print statistics
    lengths = [m["num_frames"] for m in metadata]
    print(f"\nLatent statistics:")
    print(f"  Total samples: {len(lengths)}")
    print(f"  Min frames: {min(lengths)}")
    print(f"  Max frames: {max(lengths)}")
    print(f"  Mean frames: {sum(lengths) / len(lengths):.1f}")
    print(f"  Total frames: {sum(lengths)}")


if __name__ == "__main__":
    main()
