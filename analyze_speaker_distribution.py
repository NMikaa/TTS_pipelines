"""Analyze speaker distribution in saba_clean dataset."""
import json
from collections import defaultdict
from pathlib import Path

manifest_path = Path("data/saba_clean/train_manifest.json")

speaker_stats = defaultdict(lambda: {"count": 0, "hours": 0.0})

with open(manifest_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        speaker = entry["speaker_id"]
        duration = entry["duration"]
        speaker_stats[speaker]["count"] += 1
        speaker_stats[speaker]["hours"] += duration / 3600

# Sort by hours (descending)
sorted_speakers = sorted(speaker_stats.items(), key=lambda x: -x[1]["hours"])

print(f"Total speakers: {len(speaker_stats)}")
print(f"{'Speaker':<40} {'Samples':>10} {'Hours':>10}")
print("-" * 62)

total_hours = 0
for speaker, stats in sorted_speakers:
    print(f"{speaker:<40} {stats['count']:>10} {stats['hours']:>10.2f}")
    total_hours += stats["hours"]

print("-" * 62)
print(f"{'TOTAL':<40} {sum(s['count'] for s in speaker_stats.values()):>10} {total_hours:>10.2f}")

# Identify hold-out speaker (smallest)
holdout_speaker, holdout_stats = sorted_speakers[-1]
print(f"\nHold-out speaker (smallest): {holdout_speaker}")
print(f"  Samples: {holdout_stats['count']}")
print(f"  Hours: {holdout_stats['hours']:.2f}")