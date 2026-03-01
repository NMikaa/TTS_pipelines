"""
Simple alignment generation script
Creates manifest from data.parquet and runs NeMo Forced Aligner
"""

import os
import sys
import json
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm

print("="*70)
print("ALIGNMENT GENERATION")
print("="*70)

# Configuration
DATA_PARQUET = r"C:\Users\nikam\Projects\TTS\data.parquet"
CLIPS_DIR = r"C:\Users\nikam\Projects\TTS\cv-corpus-22.0-2025-06-20\ka\clips_24k"
MODEL_PATH = r"C:\Users\nikam\Projects\TTS\models\fast_conformer_georgian.nemo"
ALIGNMENT_BASE = Path(r"C:\Users\nikam\Projects\TTS\alignment")
NFA_SCRIPT = r"C:\Users\nikam\Projects\TTS\NeMo\tools\nemo_forced_aligner\align.py"

# Paths
manifest_path = ALIGNMENT_BASE / "nfa_input_manifest.json"
output_dir = ALIGNMENT_BASE / "nfa_temp"
nfa_dir = Path(NFA_SCRIPT).parent

# Create directories
ALIGNMENT_BASE.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Create manifest
print("\nStep 1: Creating NFA manifest...")
df = pd.read_parquet(DATA_PARQUET)
print(f"Loaded {len(df)} samples from parquet")

manifest_entries = []
missing = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating manifest"):
    audio_file = row['path'].replace('.mp3', '.wav')
    audio_path = Path(CLIPS_DIR) / audio_file

    if audio_path.exists():
        manifest_entries.append({
            'audio_filepath': str(audio_path.absolute()),
            'text': row['sentence'],
            'duration': row['duration_seconds']
        })
    else:
        missing += 1

# Write manifest
with open(manifest_path, 'w', encoding='utf-8') as f:
    for entry in manifest_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"\nManifest created:")
print(f"  Path: {manifest_path}")
print(f"  Valid entries: {len(manifest_entries):,}")
print(f"  Missing files: {missing:,}")

# Step 2: Run NFA
print(f"\n{'='*70}")
print("Step 2: Running NeMo Forced Aligner")
print("="*70)

cmd = [
    sys.executable,
    str(NFA_SCRIPT),
    f"model_path={MODEL_PATH}",
    f"manifest_filepath={manifest_path}",
    f"output_dir={output_dir}",
    "batch_size=32",
    "transcribe_device=cuda",
    "viterbi_device=cuda",
    "use_local_attention=true",
    "save_output_file_formats=[ctm]",
    "ctm_file_config.remove_blank_tokens=true",
]

print(f"\nStarting NFA with GPU acceleration...")
print(f"Expected time: 4-5 hours for {len(manifest_entries):,} files")
print(f"\nTo monitor progress, open another terminal and run:")
print(f"  python monitor_progress.py")
print(f"\n{'='*70}\n")

try:
    result = subprocess.run(cmd, cwd=nfa_dir)

    if result.returncode == 0:
        print(f"\n{'='*70}")
        print("COMPLETED!")
        print("="*70)

        ctm_dir = output_dir / "ctm" / "words"
        if ctm_dir.exists():
            ctm_count = len(list(ctm_dir.glob("*.ctm")))
            print(f"\nGenerated {ctm_count:,} CTM files")
            print(f"Success rate: {ctm_count/len(manifest_entries)*100:.1f}%")

        print(f"\nNext step:")
        print(f"  python process_alignments.py")
        sys.exit(0)
    else:
        print(f"\nFailed with return code: {result.returncode}")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    ctm_dir = output_dir / "ctm" / "words"
    if ctm_dir.exists():
        print(f"Processed: {len(list(ctm_dir.glob('*.ctm'))):,} files so far")
    sys.exit(1)
