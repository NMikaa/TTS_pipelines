"""
Monitor alignment generation progress
Run this in a separate terminal while NFA is running
"""

import time
from pathlib import Path
from datetime import datetime, timedelta

print("="*70)
print("ALIGNMENT PROGRESS MONITOR")
print("="*70)
print("\nMonitoring alignment generation...")
print("Press Ctrl+C to stop monitoring\n")

alignment_base = Path("C:/Users/nikam/Projects/TTS/alignment")
nfa_temp = alignment_base / "nfa_temp"
ctm_dir = nfa_temp / "ctm" / "words"
manifest_path = alignment_base / "nfa_input_manifest.json"

# Get total files
if manifest_path.exists():
    with open(manifest_path, 'r', encoding='utf-8') as f:
        total_files = sum(1 for _ in f)
else:
    print("Manifest not found!")
    exit(1)

print(f"Total files to process: {total_files:,}\n")

last_count = 0
start_time = time.time()
last_update_time = start_time

try:
    while True:
        if ctm_dir.exists():
            current_count = len(list(ctm_dir.glob("*.ctm")))

            if current_count != last_count:
                elapsed = time.time() - start_time
                rate = current_count / elapsed if elapsed > 0 else 0
                remaining = total_files - current_count
                eta_seconds = remaining / rate if rate > 0 else 0

                elapsed_str = str(timedelta(seconds=int(elapsed)))
                eta_str = str(timedelta(seconds=int(eta_seconds)))

                percent = (current_count / total_files * 100)

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Progress: {current_count:,}/{total_files:,} ({percent:.1f}%) | "
                      f"Rate: {rate:.2f} files/s | "
                      f"Elapsed: {elapsed_str} | "
                      f"ETA: {eta_str}      ", end='', flush=True)

                last_count = current_count
                last_update_time = time.time()
            else:
                # Check if stalled
                time_since_update = time.time() - last_update_time
                if time_since_update > 60 and current_count > 0:
                    print(f"\n⚠️  No progress for {int(time_since_update)}s - may be stuck")
                    last_update_time = time.time()  # Reset warning

        time.sleep(5)  # Check every 5 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
    if ctm_dir.exists():
        final_count = len(list(ctm_dir.glob("*.ctm")))
        print(f"Final count: {final_count:,}/{total_files:,} files")
