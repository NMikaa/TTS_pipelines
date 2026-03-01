"""
Dataset for Pocket TTS training with pre-computed latents.

Supports two modes:
1. Pre-computed latents (fast): loads .pt files from precompute_latents.py
2. On-the-fly encoding (slow): encodes audio through Mimi during training
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class LatentDataset(Dataset):
    """Dataset loading pre-computed latents.

    Expects a directory with:
        - metadata.json: list of {latent_path, text, speaker_id, num_frames}
        - 000000.pt, 000001.pt, ...: latent tensors [S, 32]
    """

    def __init__(
        self,
        latents_dir: str,
        max_frames: int = 188,  # 15s * 12.5 fps
        min_frames: int = 13,   # ~1s
    ):
        self.latents_dir = Path(latents_dir)

        with open(self.latents_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Filter by length
        self.metadata = [
            m for m in self.metadata
            if min_frames <= m["num_frames"] <= max_frames
        ]
        print(f"Loaded {len(self.metadata)} samples from {latents_dir}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        meta = self.metadata[idx]
        latent = torch.load(
            self.latents_dir / meta["latent_path"],
            map_location="cpu",
            weights_only=True,
        )
        return {
            "latent": latent,           # [S, 32]
            "text": meta["text"],
            "num_frames": meta["num_frames"],
            "speaker_id": meta.get("speaker_id", "unknown"),
        }


class BucketSampler(Sampler):
    """Groups samples by similar length to minimize padding waste.

    Sorts samples into buckets by latent length, then yields
    batches from the same bucket.
    """

    def __init__(self, dataset: LatentDataset, batch_size: int, num_buckets: int = 10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets

    def __iter__(self):
        # Sort indices by length
        indices = list(range(len(self.dataset)))
        lengths = [self.dataset.metadata[i]["num_frames"] for i in indices]
        sorted_indices = sorted(indices, key=lambda i: lengths[i])

        # Create buckets
        bucket_size = max(1, len(sorted_indices) // self.num_buckets)
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i : i + bucket_size]
            random.shuffle(bucket)
            buckets.append(bucket)

        # Shuffle bucket order
        random.shuffle(buckets)

        # Yield batches from buckets
        all_indices = []
        for bucket in buckets:
            all_indices.extend(bucket)

        # Yield complete batches
        for i in range(0, len(all_indices) - self.batch_size + 1, self.batch_size):
            yield all_indices[i : i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


def collate_latent_batch(batch: list[dict]) -> dict:
    """Collate variable-length latents with padding and masking.

    Returns:
        latents: [B, max_S, 32] padded latent tensor
        mask: [B, max_S] boolean mask (True = valid)
        texts: list of text strings
        num_frames: [B] actual lengths
    """
    batch_size = len(batch)
    max_frames = max(b["num_frames"] for b in batch)
    latent_dim = batch[0]["latent"].shape[-1]

    latents = torch.zeros(batch_size, max_frames, latent_dim)
    mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["num_frames"]
        latents[i, :L] = b["latent"][:L]
        mask[i, :L] = True

    return {
        "latents": latents,
        "mask": mask,
        "texts": [b["text"] for b in batch],
        "num_frames": torch.tensor([b["num_frames"] for b in batch]),
    }


def create_dataloader(
    latents_dir: str,
    batch_size: int,
    num_workers: int = 4,
    max_frames: int = 188,
    use_bucketing: bool = True,
) -> DataLoader:
    """Create DataLoader with optional length bucketing."""
    dataset = LatentDataset(latents_dir, max_frames=max_frames)

    if use_bucketing:
        sampler = BucketSampler(dataset, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_latent_batch,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_latent_batch,
            pin_memory=True,
            drop_last=True,
        )
