"""TTSDS prosody on FLEURS-KA — Pitch + mHuBERT-147 Token SR.

Compares synthetic FLEURS-KA generations against real FLEURS-KA reference audio.
979 samples per checkpoint. Includes MagPIE TTS for cross-architecture comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 /root/TTS_pipelines/venv_ttsds/bin/python pipelines/omnivoice/run_ttsds_fleurs.py
"""

import os
import json
from pathlib import Path
from ttsds import BenchmarkSuite
from ttsds.util.dataset import DirectoryDataset
from ttsds.benchmarks.benchmark import BenchmarkCategory
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
from ttsds.benchmarks.prosody.mhubert_token import MHubert147TokenSRBenchmark

SCRIPT_DIR = Path(__file__).parent
EVAL_RESULTS_DIR = SCRIPT_DIR / "eval_results"
REAL_REF_DIR = SCRIPT_DIR / "eval_data" / "fleurs_ka" / "audio"
OUTPUT_DIR = EVAL_RESULTS_DIR / "ttsds_fleurs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_checkpoints():
    ckpts = []
    for ckpt_dir in sorted(EVAL_RESULTS_DIR.iterdir()):
        if not ckpt_dir.is_dir() or ckpt_dir.name.startswith("ttsds"):
            continue
        fleurs_dir = ckpt_dir / "fleurs_ka"
        if fleurs_dir.exists():
            wavs = list(fleurs_dir.glob("*.wav"))
            if len(wavs) >= 100:
                ckpts.append((ckpt_dir.name, fleurs_dir))
    return ckpts


def main():
    checkpoints = get_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints with FLEURS-KA audio:")
    for name, _ in checkpoints:
        print(f"  - {name}")

    real = [DirectoryDataset(str(REAL_REF_DIR), name="fleurs_ka_real")]
    print(f"\nReference: {len(real[0])} real FLEURS-KA files\n")

    synth = [DirectoryDataset(str(d), name=name) for name, d in checkpoints]

    print("Running TTSDS prosody on FLEURS-KA: Pitch + mHuBERT-147 Token SR\n")

    suite = BenchmarkSuite(
        datasets=synth,
        reference_datasets=real,
        benchmarks={
            "pitch": PitchBenchmark,
            "hubert_token_sr": MHubert147TokenSRBenchmark,
        },
        multilingual=False,
        write_to_file=str(OUTPUT_DIR / "ttsds_fleurs_results.csv"),
        skip_errors=True,
        include_environment=False,
        category_weights={
            BenchmarkCategory.PROSODY: 1.0,
            BenchmarkCategory.SPEAKER: 0.0,
            BenchmarkCategory.INTELLIGIBILITY: 0.0,
            BenchmarkCategory.GENERIC: 0.0,
        },
    )

    results = suite.run()
    print("\n=== RESULTS ===")
    agg = suite.get_aggregated_results()
    print(agg)

    if hasattr(agg, "to_dict"):
        with open(OUTPUT_DIR / "aggregated.json", "w") as f:
            json.dump(agg.to_dict(), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
