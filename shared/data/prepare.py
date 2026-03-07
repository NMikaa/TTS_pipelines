"""
Prepare unified manifest from Common Voice Georgian data.

Reads voice_actor_manifest.json (JSONL) and produces a list of entry dicts
with keys: id, audio_path, text, speaker_id.
"""

import json
from pathlib import Path
from typing import List, Dict


def prepare_dataset(data_dir: str) -> List[Dict]:
    """Load and validate the dataset manifest.

    Args:
        data_dir: Path to the data directory containing voice_actor_manifest.json
                  and audio/ subdirectory.

    Returns:
        List of dicts with keys: id, audio_path, text, speaker_id
    """
    data_path = Path(data_dir)
    manifest_path = data_path / "voice_actor_manifest.json"
    audio_dir = data_path / "audio"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    entries = []
    skipped = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Extract fields from manifest
            audio_filepath = record.get("audio_filepath", "")
            text = record.get("text", "")
            speaker_id = record.get("source", record.get("speaker_id", "unknown"))

            # Resolve audio path — manifest may have relative paths
            if audio_filepath.startswith("/"):
                audio_path = Path(audio_filepath)
            else:
                # Try audio/ subdir
                filename = Path(audio_filepath).name
                audio_path = audio_dir / filename

            if not audio_path.exists():
                skipped += 1
                continue

            clip_id = audio_path.stem

            entries.append({
                "id": clip_id,
                "audio_path": str(audio_path),
                "text": text,
                "speaker_id": str(speaker_id),
            })

    print(f"Loaded {len(entries)} entries ({skipped} skipped — audio not found)")
    return entries
