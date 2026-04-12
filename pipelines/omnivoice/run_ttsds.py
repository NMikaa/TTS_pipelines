"""TTSDS prosody/speaker/intelligibility evaluation for all OmniVoice checkpoints.

Runs in venv_ttsds.

Usage:
    CUDA_VISIBLE_DEVICES=1 /root/TTS_pipelines/venv_ttsds/bin/python pipelines/omnivoice/run_ttsds.py
"""

import os
import json
import sys
from pathlib import Path
from ttsds import BenchmarkSuite
from ttsds.util.dataset import DirectoryDataset
from ttsds.benchmarks.benchmark import BenchmarkCategory

SCRIPT_DIR = Path(__file__).parent
EVAL_RESULTS_DIR = SCRIPT_DIR / "eval_results"
REAL_REF_DIR = SCRIPT_DIR / "eval_data" / "cv_geo" / "audio"
OUTPUT_DIR = EVAL_RESULTS_DIR / "ttsds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all checkpoint dirs that have generated CV-GEO audio
def get_checkpoints():
    ckpts = []
    for ckpt_dir in sorted(EVAL_RESULTS_DIR.iterdir()):
        if not ckpt_dir.is_dir() or ckpt_dir.name == "ttsds" or ckpt_dir.name == "magpie_tts":
            continue
        cv_geo_dir = ckpt_dir / "cv_geo"
        if cv_geo_dir.exists():
            wavs = list(cv_geo_dir.glob("*.wav"))
            if len(wavs) >= 50:
                ckpts.append((ckpt_dir.name, cv_geo_dir))
    return ckpts


def main():
    checkpoints = get_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints with CV-GEO audio")

    # Real reference is the same for all
    real = [DirectoryDataset(str(REAL_REF_DIR), name="cv_geo_real")]
    print(f"Reference: {len(real[0])} real files\n")

    # Build synthetic datasets - all checkpoints
    synth = [DirectoryDataset(str(d), name=name) for name, d in checkpoints]

    # Run benchmark suite
    print("Running TTSDS benchmark suite (multilingual mode)...")
    print("Categories: Prosody, Speaker, Intelligibility, Generic (Environment excluded)")

    suite = BenchmarkSuite(
        datasets=synth,
        reference_datasets=real,
        multilingual=True,
        write_to_file=str(OUTPUT_DIR / "ttsds_results.csv"),
        skip_errors=True,
        include_environment=False,
        category_weights={
            BenchmarkCategory.SPEAKER: 0.25,
            BenchmarkCategory.INTELLIGIBILITY: 0.25,
            BenchmarkCategory.PROSODY: 0.25,
            BenchmarkCategory.GENERIC: 0.25,
        },
    )

    results = suite.run()
    print("\n=== RESULTS ===")
    agg = suite.get_aggregated_results()
    print(agg)

    # Save aggregated results as JSON for the report
    agg_path = OUTPUT_DIR / "aggregated.json"
    if hasattr(agg, "to_dict"):
        with open(agg_path, "w") as f:
            json.dump(agg.to_dict(), f, indent=2)
    print(f"\nSaved to {agg_path}")
    print(f"Full results in {OUTPUT_DIR / 'ttsds_results.csv'}")


if __name__ == "__main__":
    main()
