"""
Download Common Voice Georgian data from S3.

Usage:
    python -m shared.data.download --output-dir ./data
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3
from tqdm import tqdm


S3_BUCKET = "ttsopensource"
S3_REGION = "eu-central-1"
S3_PREFIX = "tts-georgian/"
S3_FILES = {
    "tts-georgian/manifests/train_manifest.json": "train_manifest.json",
    "tts-georgian/audio_clean.tar.gz": "audio_clean.tar.gz",
    "tts-georgian/manifests/eval_manifest.json": "eval_manifest.json",
    "tts-georgian/manifests/speaker_refs_manifest.json": "speaker_refs_manifest.json",
}


def download_from_s3(output_dir: str) -> Path:
    """Download all data files from S3 and extract audio."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=S3_REGION)

    for s3_key, local_name in S3_FILES.items():
        local_path = output / local_name
        if local_path.exists():
            print(f"  Already exists: {local_path}")
            continue

        print(f"  Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
        # Get file size for progress bar
        meta = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        total_size = meta["ContentLength"]

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=local_name) as pbar:
            s3.download_file(
                S3_BUCKET, s3_key, str(local_path),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )

    # Extract audio.rar
    audio_dir = output / "audio"
    rar_path = output / "audio.rar"
    if not audio_dir.exists() and rar_path.exists():
        print(f"  Extracting {rar_path} ...")
        subprocess.run(["unrar", "x", "-o+", str(rar_path), str(output) + "/"], check=True)
        # Common Voice clips are in clips_24k/, rename to audio/
        clips_dir = output / "clips_24k"
        if clips_dir.exists() and not audio_dir.exists():
            clips_dir.rename(audio_dir)
            print(f"  Renamed clips_24k/ -> audio/")
    elif audio_dir.exists():
        print(f"  Audio already extracted: {audio_dir}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Download Georgian TTS data from S3")
    parser.add_argument("--output-dir", type=str, default="./data")
    args = parser.parse_args()

    print("Downloading data from S3...")
    download_from_s3(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
