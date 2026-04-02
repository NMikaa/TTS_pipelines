"""Analyze speaker distribution in punctuated-only segments."""
import json
from collections import defaultdict
from pathlib import Path

manifest_path = Path("data/saba_clean/train_manifest.json")

speaker_stats = defaultdict(lambda: {"count": 0, "hours": 0.0})

with open(manifest_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        if not entry.get("has_punctuation", False):
            continue
        speaker = entry["speaker_id"]
        duration = entry["duration"]
        speaker_stats[speaker]["count"] += 1
        speaker_stats[speaker]["hours"] += duration / 3600

# Sort by hours (descending)
sorted_speakers = sorted(speaker_stats.items(), key=lambda x: -x[1]["hours"])

print(f"Speakers with punctuated segments: {len(speaker_stats)}")
print(f"{'Rank':<5} {'Speaker':<40} {'Samples':>10} {'Hours':>10}")
print("-" * 68)

total_hours = 0
for rank, (speaker, stats) in enumerate(sorted_speakers, 1):
    print(f"{rank:<5} {speaker:<40} {stats['count']:>10} {stats['hours']:>10.2f}")
    total_hours += stats["hours"]

print("-" * 68)
print(f"{'TOTAL':<47} {sum(s['count'] for s in speaker_stats.values()):>10} {total_hours:>10.2f}")

# Show bottom 5 candidates for hold-out
print(f"\nBottom 5 candidates for hold-out (for zero-shot testing):")
for rank, (speaker, stats) in enumerate(sorted_speakers[-5:], 1):
    print(f"  {speaker}: {stats['hours']:.2f}h ({stats['count']} samples)")