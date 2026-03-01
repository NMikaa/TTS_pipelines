"""
Renumber speaker IDs to incremental format (1, 2, 3, etc.)
Largest speaker (most samples) gets ID 1, next gets 2, and so on.
Updates both voice_actor_manifest.json and alignments.json
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

print("="*70)
print("RENUMBERING SPEAKER IDs")
print("="*70)

# Paths
voice_actor_manifest = Path("C:/Users/nikam/Projects/TTS/alignment/voice_actor_manifest.json")
alignments_json = Path("C:/Users/nikam/Projects/TTS/alignments.json")

# Backup files
print("\nCreating backups...")
voice_actor_backup = voice_actor_manifest.with_suffix('.json.backup')
alignments_backup = alignments_json.with_suffix('.json.backup')

import shutil
shutil.copy2(voice_actor_manifest, voice_actor_backup)
shutil.copy2(alignments_json, alignments_backup)
print(f"  Voice actor manifest backup: {voice_actor_backup}")
print(f"  Alignments backup: {alignments_backup}")

# Step 1: Count samples per speaker from voice_actor_manifest
print("\nCounting samples per speaker...")
speaker_counts = Counter()

with open(voice_actor_manifest, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        speaker_counts[entry['source']] += 1

print(f"Found {len(speaker_counts)} unique speakers")

# Step 2: Sort speakers by count (descending) and create mapping
print("\nCreating speaker ID mapping...")
sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)

speaker_mapping = {}
for idx, (speaker_id, count) in enumerate(sorted_speakers, start=1):
    speaker_mapping[speaker_id] = str(idx)
    speaker_short = speaker_id[:12] + "..." if len(speaker_id) > 12 else speaker_id
    print(f"  Speaker {idx}: {speaker_short} ({count:,} samples)")

# Step 3: Update voice_actor_manifest.json
print(f"\nUpdating {voice_actor_manifest.name}...")
temp_manifest = voice_actor_manifest.with_suffix('.json.temp')

with open(voice_actor_manifest, 'r', encoding='utf-8') as f_in:
    with open(temp_manifest, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Processing manifest", total=sum(speaker_counts.values())):
            entry = json.loads(line)
            entry['source'] = speaker_mapping[entry['source']]
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Replace original with temp
temp_manifest.replace(voice_actor_manifest)
print(f"Updated {voice_actor_manifest}")

# Step 4: Update alignments.json
print(f"\nUpdating {alignments_json.name}...")
with open(alignments_json, 'r', encoding='utf-8') as f:
    alignments_data = json.load(f)

print(f"Loaded {len(alignments_data)} alignment entries")

# Update speaker IDs in alignment keys
# The keys are audio paths, but we need to update based on the manifest mapping
# First, build a path -> speaker mapping from the updated manifest
print("Building path to speaker mapping...")
path_to_speaker = {}
with open(voice_actor_manifest, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Mapping paths", total=sum(speaker_counts.values())):
        entry = json.loads(line)
        path_to_speaker[entry['audio_filepath']] = entry['source']

# Alignments.json structure is: {audio_path: [...word timestamps...]}
# We don't need to change it since it doesn't contain speaker info
# The speaker info is only in the voice_actor_manifest.json
print("Alignments.json doesn't contain speaker IDs - no changes needed")

print("\n" + "="*70)
print("RENUMBERING COMPLETE!")
print("="*70)
print(f"\nUpdated files:")
print(f"  {voice_actor_manifest}")
print(f"\nBackup files:")
print(f"  {voice_actor_backup}")
print(f"  {alignments_backup}")
print(f"\nSpeaker mapping summary:")
for idx, (speaker_id, count) in enumerate(sorted_speakers[:5], start=1):
    speaker_short = speaker_id[:12] + "..." if len(speaker_id) > 12 else speaker_id
    print(f"  {speaker_short} -> Speaker {idx} ({count:,} samples)")

if len(sorted_speakers) > 5:
    print(f"  ... and {len(sorted_speakers) - 5} more speakers")

print(f"\nReady to train with renumbered speaker IDs!")
