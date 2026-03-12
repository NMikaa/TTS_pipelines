"""
Load all dataset entries from manifests.

This provides prepare_dataset() which loads all entries across all splits.
For split-specific loading, use get_splits() from shared.data.splits instead.
"""

import json
from pathlib import Path
from typing import List


def prepare_dataset(data_dir: str) -> List[dict]:
    """Load all entries from train + eval + test manifests.

    Args:
        data_dir: Path to data directory containing *_manifest.json files.

    Returns:
        List of all entry dicts across all splits.
    """
    data_path = Path(data_dir)
    all_entries = []

    for manifest_name in ["train_manifest.json", "eval_manifest.json", "test_manifest.json"]:
        manifest_path = data_path / manifest_name
        if not manifest_path.exists():
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    if not all_entries:
        raise FileNotFoundError(
            f"No manifests found in {data_dir}. "
            f"Run `python -m shared.data.download --output-dir {data_dir}` first."
        )

    return all_entries
