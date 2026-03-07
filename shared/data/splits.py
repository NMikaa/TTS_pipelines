"""
Deterministic hash-based train/val/test splits.

Uses MD5 hash of clip ID to ensure the same split regardless of file ordering.
Split: 90% train, 5% val, 5% test.
"""

import hashlib
from typing import List, Tuple

from .prepare import prepare_dataset


def _hash_split(clip_id: str) -> str:
    """Assign a split based on MD5 hash of clip ID."""
    h = hashlib.md5(clip_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 5:
        return "test"
    elif bucket < 10:
        return "val"
    else:
        return "train"


def get_splits(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Get deterministic train/val/test split IDs.

    Args:
        data_dir: Path to data directory.

    Returns:
        (train_ids, val_ids, test_ids) — lists of clip IDs.
    """
    entries = prepare_dataset(data_dir)

    train_ids, val_ids, test_ids = [], [], []
    for entry in entries:
        split = _hash_split(entry["id"])
        if split == "train":
            train_ids.append(entry["id"])
        elif split == "val":
            val_ids.append(entry["id"])
        else:
            test_ids.append(entry["id"])

    return train_ids, val_ids, test_ids
