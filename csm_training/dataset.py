"""
Dataset and data loading for CSM-1B streaming inference training.

Supports two data sources:
1. HuggingFace sharded datasets (shard_XXXX format)
2. Voice actor JSONL manifests

Key feature: 2-word look-ahead scheme for streaming training,
where the model learns to generate audio word-by-word with a
growing text prefix.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from .alignment import AlignmentProcessor

FRAME = 1920  # 24kHz * 0.08s
TARGET_SR = 24_000
MAX_AUDIO = 24_000 * 120
MAX_TEXT = 4096
CUTOFFS_LEN = 128


class CSMDataLoader:
    """Load and prepare datasets for CSM-1B training."""

    def __init__(
        self,
        shards_path: Optional[str] = None,
        voice_actor_path: Optional[str] = None,
        list_of_speakers: Optional[List[str]] = None,
    ):
        self.target_sr = TARGET_SR
        self.list_of_speakers = list_of_speakers or []
        self.voice_actor_path = voice_actor_path
        self.ds = self._load_data(shards_path)
        if self.list_of_speakers:
            self.ds = self._filter_speakers(self.ds, self.list_of_speakers)
        self.alignment_dict: Optional[Dict] = None

    def _load_data(self, shards_path: Optional[str]) -> Dataset:
        """Load data from HF shards or JSONL manifest."""
        if shards_path:
            datasets = [
                load_from_disk(f"{shards_path}{str(i).zfill(4)}")
                for i in range(26)
            ]
            ds = concatenate_datasets(datasets)
            ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sr))
        elif self.voice_actor_path:
            import pandas as pd

            records = []
            with open(self.voice_actor_path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            df = pd.DataFrame(records)
            df["audio"] = df["audio_filepath"]
            df["path"] = df["audio_filepath"]
            df["source"] = "0"
            ds = Dataset.from_pandas(df, preserve_index=False)
            ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        else:
            raise ValueError("Must provide either shards_path or voice_actor_path")

        return ds

    @staticmethod
    def _filter_speakers(ds: Dataset, speakers: List[str]) -> Dataset:
        """Filter dataset to keep only specified speakers."""
        if not speakers:
            return ds
        speaker_set = set(speakers)
        return ds.filter(
            lambda source: source in speaker_set,
            input_columns="source",
        )

    def process_alignments(
        self, alignment_base_path: str, min_duration: float = 0.08
    ) -> Dict:
        """Process CTM alignments for all audio in the dataset."""
        if "path" not in self.ds.column_names:
            raise ValueError("Dataset must have a 'path' column")

        audio_paths = self.ds["path"]
        print(f"Found {len(audio_paths)} audio files in dataset")

        # Extract durations from loaded audio
        print("Extracting audio durations...")
        audio_durations = {}
        for item in tqdm(self.ds, desc="Getting durations"):
            path = item["path"]
            arr = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            audio_durations[path] = len(arr) / sr

        processor = AlignmentProcessor(alignment_base_path)
        self.alignment_dict = processor.process_paths_with_durations(
            audio_paths=audio_paths,
            audio_durations=audio_durations,
            min_duration=min_duration,
        )
        print(f"Aligned: {len(self.alignment_dict)}/{len(audio_paths)} files")
        return self.alignment_dict

    def load_alignments(self, path: str):
        """Load pre-computed alignments from JSON."""
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Alignment JSON not found: {path}\n"
                f"Generate it first with: python scripts/generate_alignments.py"
            )
        with open(path, "r", encoding="utf-8") as f:
            self.alignment_dict = json.load(f)
        print(f"Loaded alignments for {len(self.alignment_dict)} files from: {path}")

    def save_alignments(self, path: str):
        """Save alignment dictionary to JSON."""
        if self.alignment_dict is None:
            raise ValueError("No alignments to save. Run process_alignments() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.alignment_dict, f, indent=2, ensure_ascii=False)
        print(f"Alignments saved to: {path}")

    def split_train_test(
        self, test_fraction: float = 0.05, seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """Split into train/test sets."""
        split = self.ds.train_test_split(test_size=test_fraction, seed=seed)
        print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")
        return split["train"], split["test"]


class TTSTrainingDataset(TorchDataset):
    """PyTorch Dataset wrapping HF Dataset for CSM-1B training.

    Uses a 2-word look-ahead scheme where each training example has:
    1. A context turn (shifted audio from same speaker) - masked from loss
    2. Current turn with word-by-word audio generation with growing text prefix
    """

    def __init__(self, dataset: Dataset, alignment_dict: Dict, processor):
        self.ds = dataset.sort("source")
        self.ds = self._attach_shifted_audio(shift=1)
        self.alignment_dict = alignment_dict
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def _attach_shifted_audio(self, shift: int = 1) -> Dataset:
        """Attach right-wrapped shifted audio/text columns for context."""
        n = len(self.ds)
        k = shift % n
        idx_shifted = np.roll(np.arange(n), k)

        shifted = (
            self.ds.select(idx_shifted)
            .select_columns(["audio", "text"])
            .rename_column("audio", "shifted_audio")
            .rename_column("text", "shifted_text")
        )
        return concatenate_datasets([self.ds, shifted], axis=1)

    @staticmethod
    def _sec_to_frame_idx(t_sec: float, sr: int) -> int:
        return int(round(t_sec * sr / FRAME)) * FRAME

    def _slice_word_segments(
        self, y: np.ndarray, sr: int, words: List[Dict]
    ) -> List[np.ndarray]:
        """Per-word audio segments, snapped to MiMi frames (>=1 frame each)."""
        segs = []
        n = len(y)
        for w in words:
            s = max(0, self._sec_to_frame_idx(float(w["start"]), sr))
            e = max(s + FRAME, self._sec_to_frame_idx(float(w["end"]), sr))
            e = min(e, n)
            segs.append(y[s:e].copy())
        return segs

    def _build_messages(
        self, alignment_dict: Dict, current_item: Dict
    ) -> List[Dict[str, Any]]:
        """Build chat messages for processor.apply_chat_template.

        Structure:
          [context_turn (masked), current_turn_word_0, word_1, ..., word_N]

        The current turn uses a 2-word look-ahead: first message has
        up to 3 words of text + first word's audio, then each subsequent
        message appends one new look-ahead word + that word's audio.
        """
        cur_path_id = str(current_item["path"])
        cur_source = str(current_item.get("source", "0"))
        assert cur_path_id in alignment_dict, f"No alignment for: {cur_path_id}"

        y_cur = current_item["audio"]["array"]
        words = alignment_dict[cur_path_id]
        word_tokens = [w["word"] for w in words]
        segs = self._slice_word_segments(y_cur, TARGET_SR, words)

        # Context turn (shifted audio from same/nearby speaker)
        y_ctx = current_item["shifted_audio"]["array"]
        ctx_text = str(current_item["shifted_text"]).strip()

        messages = [{
            "role": cur_source,
            "content": [
                {"type": "text", "text": ctx_text},
                {"type": "audio", "path": y_ctx},
            ],
        }]

        # Current turn with 2-word look-ahead
        n = len(word_tokens)
        if n > 0:
            first_k = min(3, n)
            messages.append({
                "role": cur_source,
                "content": [
                    {"type": "text", "text": " ".join(word_tokens[:first_k])},
                    {"type": "audio", "path": segs[0]},
                ],
            })

            for t in range(1, n):
                text = " " + word_tokens[t + 2] if t + 2 < n else ""
                messages.append({
                    "role": cur_source,
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "audio", "path": segs[t]},
                    ],
                })

        return messages

    @staticmethod
    def _mask_first_audio_run(
        labels: torch.Tensor, audio_token_id: int
    ) -> torch.Tensor:
        """Mask the first consecutive run of audio frames with -100.

        This ensures the context audio turn is not included in the loss.
        """
        if labels.dim() == 1:
            is_audio = (labels == audio_token_id) | (labels == -101)
            idxs = torch.nonzero(is_audio, as_tuple=False)
            if idxs.numel() == 0:
                return labels
            start = idxs[0, 0].item()
            end = start
            T = labels.shape[0]
            while end < T and (labels[end] == audio_token_id or labels[end] == -101):
                end += 1
            labels[start:end] = -100
        elif labels.dim() == 2:
            B, T = labels.shape
            for b in range(B):
                row = labels[b]
                is_audio = (row == audio_token_id) | (row == -101)
                idxs = torch.nonzero(is_audio, as_tuple=False)
                if idxs.numel() == 0:
                    continue
                start = idxs[0, 0].item()
                end = start
                while end < T and (row[end] == audio_token_id or row[end] == -101):
                    end += 1
                row[start:end] = -100
        return labels

    @staticmethod
    def _pad_cutoffs(cutoffs: torch.Tensor, max_len: int = CUTOFFS_LEN) -> torch.Tensor:
        pad_len = max_len - cutoffs.shape[1]
        if pad_len > 0:
            pad = torch.full((1, pad_len), -1, dtype=cutoffs.dtype)
            cutoffs = torch.cat([cutoffs, pad], dim=1)
        return cutoffs

    def __getitem__(self, idx):
        """Build a single training example with context masking."""
        item = self.ds[idx]
        messages = self._build_messages(self.alignment_dict, item)

        # Pad short audio segments to at least one frame
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "audio":
                    arr = content["path"]
                    if len(arr) < FRAME:
                        content["path"] = np.pad(
                            arr, (0, FRAME - len(arr)), mode="constant"
                        )

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length",
                "max_length": MAX_TEXT,
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": TARGET_SR,
                "max_length": MAX_AUDIO,
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )

        batch["input_values_cutoffs"] = self._pad_cutoffs(
            batch["input_values_cutoffs"], CUTOFFS_LEN
        )

        required_keys = [
            "input_ids", "attention_mask", "labels",
            "input_values", "input_values_cutoffs",
        ]
        example = {key: batch[key][0] for key in required_keys if key in batch}

        # Mask context audio from loss
        example["labels"] = self._mask_first_audio_run(
            example["labels"], self.processor.audio_token_id
        )
        return example


def filter_errored_samples(ds: TorchDataset) -> TorchDataset:
    """Filter out samples that throw errors during __getitem__."""
    from torch.utils.data import Subset

    valid_indices = []
    removed = 0
    for i in tqdm(range(len(ds)), desc="Validating samples"):
        try:
            _ = ds[i]
            valid_indices.append(i)
        except Exception:
            removed += 1

    print(f"Removed {removed} errored samples, {len(valid_indices)} remaining")
    return Subset(ds, valid_indices)
