"""
Download Common Voice Georgian Cleaned data from HuggingFace.

Usage:
    python -m shared.data.download --output-dir ./data
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


HF_DATASET = "NMikka/Common-Voice-Geo-Cleaned"


def download_from_hf(output_dir: str) -> Path:
    """Download dataset from HuggingFace and export to local audio + manifest files.

    Creates:
        output_dir/
            audio/           — WAV files (24kHz mono)
            train_manifest.json   — JSONL manifest for training split
            eval_manifest.json    — JSONL manifest for eval split
            test_manifest.json    — JSONL manifest for test split (speaker refs)
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)

    ds = load_dataset(HF_DATASET)

    for split_name in ds:
        split = ds[split_name]
        manifest_path = output / f"{split_name}_manifest.json"

        if manifest_path.exists():
            print(f"  Manifest already exists: {manifest_path}")
            # Still check if audio files exist
            existing_audio = sum(1 for _ in audio_dir.glob("*.wav"))
            if existing_audio >= len(split):
                print(f"  Audio already extracted ({existing_audio} files)")
                continue

        print(f"  Processing {split_name} split ({len(split)} samples)...")
        entries = []

        for sample in tqdm(split, desc=split_name):
            clip_id = sample["id"]
            wav_path = audio_dir / f"{clip_id}.wav"

            # Write audio if not already present
            if not wav_path.exists():
                audio = sample["audio"]
                sf.write(str(wav_path), audio["array"], audio["sampling_rate"])

            entries.append({
                "id": clip_id,
                "audio_filepath": str(wav_path.resolve()),
                "text": sample["text"],
                "speaker_id": str(sample["speaker_id"]),
                "duration": sample["duration"],
            })

        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(entries)} entries to {manifest_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Download Georgian TTS data from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="./data")
    args = parser.parse_args()

    print(f"Downloading {HF_DATASET} from HuggingFace...")
    download_from_hf(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
