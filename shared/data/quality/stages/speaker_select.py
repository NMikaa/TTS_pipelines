"""Stage: Speaker selection by total audio duration.

Keeps speakers with at least min_speaker_duration_sec total audio.
Follows the Catalan TTS paper threshold of >= 1400s per speaker.
Falls back to ECAPA-TDNN clustering if no speaker IDs in manifest.
"""

from collections import defaultdict, Counter

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from ..audio_io import load, info as audio_info
from ..config import PipelineContext, MIN_SPEAKER_DURATION_SEC, MAX_SPEAKERS

NAME = "speaker_select"
DESCRIPTION = f"Keep speakers with >= {MIN_SPEAKER_DURATION_SEC}s total audio"


def run(entries, ctx: PipelineContext):
    has_speaker_ids = all(
        e.get("speaker_id") and e["speaker_id"] != "unknown"
        for e in entries
    )

    if has_speaker_ids:
        return _select_by_manifest(entries, ctx)
    else:
        return _select_by_clustering(entries, ctx)


def _select_by_manifest(entries, ctx):
    speaker_duration = defaultdict(float)
    speaker_counts = Counter()

    for e in entries:
        spk = e["speaker_id"]
        dur = e.get("duration", 0)
        if dur <= 0:
            try:
                ai = audio_info(e["audio_path"])
                dur = ai.num_frames / ai.sample_rate
            except Exception:
                dur = 0
        speaker_duration[spk] += dur
        speaker_counts[spk] += 1

    ctx.logger.info(
        f"Found {len(speaker_counts)} speakers. "
        f"Duration (s): {dict(sorted(speaker_duration.items(), key=lambda x: -x[1]))}"
    )

    valid_speakers = {
        spk for spk, dur in speaker_duration.items()
        if dur >= MIN_SPEAKER_DURATION_SEC
    }

    if len(valid_speakers) > MAX_SPEAKERS:
        top = sorted(speaker_duration.items(), key=lambda x: -x[1])[:MAX_SPEAKERS]
        valid_speakers = {spk for spk, _ in top}

    kept = [e for e in entries if e["speaker_id"] in valid_speakers]
    dropped = len(entries) - len(kept)

    ctx.logger.info(
        f"Speaker select: {len(kept)}/{len(entries)} kept "
        f"(dropped {dropped}, {len(valid_speakers)} speakers, "
        f"threshold={MIN_SPEAKER_DURATION_SEC}s)"
    )
    return kept


def _select_by_clustering(entries, ctx):
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import KMeans

    ctx.logger.info("No speaker IDs — extracting ECAPA-TDNN embeddings for clustering")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": ctx.device},
    )

    embeddings = []
    valid_entries = []
    for entry in tqdm(entries, desc="Speaker embeddings"):
        try:
            waveform, sr = load(entry["audio_path"])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            emb = classifier.encode_batch(waveform).squeeze().cpu().numpy()
            embeddings.append(emb)
            valid_entries.append(entry)
        except Exception:
            pass

    if not embeddings:
        ctx.logger.warning("No embeddings extracted!")
        return entries

    embeddings_np = np.stack(embeddings)
    n_clusters = min(MAX_SPEAKERS, max(2, len(embeddings_np) // 50))

    ctx.logger.info(f"Clustering {len(embeddings_np)} embeddings into {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)

    cluster_counts = Counter(labels)
    valid_clusters = {c for c, count in cluster_counts.items() if count >= 50}

    kept = []
    for entry, label in zip(valid_entries, labels):
        if label in valid_clusters:
            entry_copy = entry.copy()
            entry_copy["cluster_speaker_id"] = int(label)
            kept.append(entry_copy)

    dropped = len(entries) - len(kept)
    ctx.logger.info(
        f"Speaker select: {len(kept)}/{len(entries)} kept "
        f"(dropped {dropped}, {len(valid_clusters)} clusters)"
    )
    return kept
