"""
Prepare Experiment A data: concatenate train+eval, filter to punctuated, apply new split.

Steps:
1. Concatenate train_manifest.json and eval_manifest.json from saba_clean
2. Filter to punctuated segments only
3. Exclude hold-out speaker (reserve for zero-shot testing)
4. Deterministic 98/2 train/val split using hash on segment ID (new split, not original)
5. Create speaker ID → embedding index mapping
6. Write experiment-specific manifests with speaker remapping
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Configuration
HOLDOUT_SPEAKER = "ზაალ სამადაშვილი"
INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "saba_clean"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "saba_experiment_a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_split(segment_id: str, train_ratio: float = 0.98) -> str:
    """Deterministic hash-based split assignment."""
    h = hashlib.md5(segment_id.encode()).hexdigest()
    if int(h[:8], 16) / 0xFFFFFFFF < train_ratio:
        return "train"
    return "val"

def collect_all_segments(input_dir):
    """Concatenate all segments from train and eval manifests.

    Returns:
        list: List of segment dicts from both manifests
    """
    segments = []

    for manifest_file in ["train_manifest.json", "eval_manifest.json"]:
        path = input_dir / manifest_file
        print(f"  Reading {manifest_file}...")
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                segments.append(entry)
        print(f"    Added {sum(1 for _ in open(path)):,} entries")

    return segments

def create_speaker_mapping(segments, exclude_speaker):
    """Create mapping from speaker_id to embedding index.

    Args:
        segments: List of segment dicts
        exclude_speaker: Speaker ID to exclude (hold-out)

    Returns:
        dict: {speaker_id -> embedding_index}
    """
    speaker_counts = defaultdict(int)

    for entry in segments:
        speaker = entry["speaker_id"]
        # Only count speakers that:
        # 1. Are not the hold-out speaker
        # 2. Are not multi-speaker entries (contain "," or other multi indicators)
        if speaker != exclude_speaker and "," not in speaker:
            speaker_counts[speaker] += 1

    # Sort by count (descending) for deterministic order
    sorted_speakers = sorted(speaker_counts.items(), key=lambda x: (-x[1], x[0]))

    # Create mapping: 0-4 reserved for pretrained, 5-44 for Georgian speakers
    speaker_to_idx = {}
    for idx, (speaker, _) in enumerate(sorted_speakers):
        speaker_to_idx[speaker] = idx + 5

    return speaker_to_idx

# Collect all segments from both manifests
print("Concatenating train_manifest.json and eval_manifest.json...")
all_segments = collect_all_segments(INPUT_DIR)
print(f"Total segments concatenated: {len(all_segments):,}\n")

# Create speaker mapping
print("Creating speaker mapping...")
speaker_to_idx = create_speaker_mapping(all_segments, HOLDOUT_SPEAKER)
print(f"Found {len(speaker_to_idx)} training speakers\n")

# Separate by split and remap speakers
train_entries = []
val_entries = []
holdout_entries = []
filtered_counts = defaultdict(int)

print("Filtering and splitting...")
for entry in all_segments:
    speaker = entry["speaker_id"]

    # Filter 1: Must be punctuated
    if not entry.get("has_punctuation", False):
        filtered_counts["not_punctuated"] += 1
        continue

    # Filter 2: Skip multi-speaker entries
    if entry.get("multi_speaker", False) or "," in speaker:
        filtered_counts["multi_speaker"] += 1
        continue

    # Separate hold-out speaker
    if speaker == HOLDOUT_SPEAKER:
        holdout_entries.append(entry)
        continue

    # Skip unmapped speakers
    if speaker not in speaker_to_idx:
        filtered_counts["unmapped_speaker"] += 1
        continue

    # Remap speaker to embedding index
    entry["speaker_id_original"] = speaker
    entry["speaker_id"] = speaker_to_idx[speaker]

    # Deterministic 98/2 train/val split
    if get_split(entry["id"]) == "train":
        train_entries.append(entry)
    else:
        val_entries.append(entry)

# Write manifests
train_path = OUTPUT_DIR / "train_manifest.json"
with open(train_path, "w") as f:
    for entry in train_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

val_path = OUTPUT_DIR / "val_manifest.json"
with open(val_path, "w") as f:
    for entry in val_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

holdout_path = OUTPUT_DIR / "holdout_manifest.json"
with open(holdout_path, "w") as f:
    for entry in holdout_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Compute statistics
train_hours = sum(e["duration"] / 3600 for e in train_entries)
val_hours = sum(e["duration"] / 3600 for e in val_entries)
holdout_hours = sum(e["duration"] / 3600 for e in holdout_entries)

print("\n" + "=" * 60)
print("EXPERIMENT A DATA PREPARATION SUMMARY")
print("=" * 60)
print(f"Training speakers: {len(speaker_to_idx)}")
print(f"  Train set: {len(train_entries):,} samples ({train_hours:.1f}h)")
print(f"  Val set:   {len(val_entries):,} samples ({val_hours:.1f}h)")
print(f"Hold-out speaker: {HOLDOUT_SPEAKER}")
print(f"  Zero-shot set: {len(holdout_entries):,} samples ({holdout_hours:.2f}h)")
print()
print(f"Output directory: {OUTPUT_DIR}")
print(f"  - {train_path}")
print(f"  - {val_path}")
print(f"  - {holdout_path}")

if filtered_counts:
    print()
    print("Filtered out:")
    for reason, count in sorted(filtered_counts.items()):
        print(f"  {reason}: {count:,}")
