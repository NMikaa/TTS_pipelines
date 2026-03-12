"""
Get train/val/test splits from the downloaded manifests.

The HuggingFace dataset (NMikka/Common-Voice-Geo-Cleaned) ships with fixed splits:
  - train: 20,300 samples
  - eval: 1,001 samples
  - test: 120 samples (best quality, speaker references)

All pipelines MUST use get_splits() for fair comparison.
"""

import json
from pathlib import Path
from typing import List, Tuple


def get_splits(data_dir: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Get the fixed train/eval/test splits.

    Args:
        data_dir: Path to data directory with manifests.

    Returns:
        (train_entries, eval_entries, test_entries) — lists of entry dicts.
    """
    train = _load_manifest(data_dir, "train")
    val = _load_manifest(data_dir, "eval")
    test = _load_manifest(data_dir, "test")

    return train, val, test


def _load_manifest(data_dir: str, split: str) -> List[dict]:
    manifest_path = Path(data_dir) / f"{split}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run `python -m shared.data.download --output-dir {data_dir}` first."
        )

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries
