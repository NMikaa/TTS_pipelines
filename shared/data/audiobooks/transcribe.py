"""
Multi-GPU ASR transcription using Meta Omnilingual ASR 7B.

Runs two model instances (one per GPU) for ~2x throughput.
Each instance processes a shard of audio chunks.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
LANG = "kat_Geor"


def _transcribe_shard(
    gpu_id: int,
    audio_paths: list[str],
    output_path: str,
    batch_size: int,
):
    """Worker: load model on one GPU and transcribe a shard of audio files."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(message)s",
    )
    log = logging.getLogger(f"transcribe_gpu{gpu_id}")

    log.info(f"Loading omniASR_LLM_7B on GPU {gpu_id}...")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B", device="cuda:0")

    log.info(f"Transcribing {len(audio_paths)} files (batch_size={batch_size})...")

    results = {}
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[batch_start:batch_start + batch_size]
        batch_langs = [LANG] * len(batch_paths)

        try:
            texts = pipeline.transcribe(batch_paths, lang=batch_langs, batch_size=len(batch_paths))
            for path, text in zip(batch_paths, texts):
                results[Path(path).stem] = text
        except Exception as e:
            log.error(f"Batch failed: {e}. Falling back to single-file mode.")
            for path in batch_paths:
                try:
                    text = pipeline.transcribe([path], lang=[LANG], batch_size=1)
                    results[Path(path).stem] = text[0]
                except Exception as e2:
                    log.error(f"Failed on {path}: {e2}")
                    results[Path(path).stem] = ""

        done = min(batch_start + batch_size, len(audio_paths))
        if done % (batch_size * 5) == 0 or done == len(audio_paths):
            log.info(f"  Progress: {done}/{len(audio_paths)}")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f"Done. Saved {len(results)} transcriptions to {output_path}")


def transcribe_chunks(
    chunk_dir: str,
    output_path: str,
    num_gpus: int = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, str]:
    """
    Transcribe all WAV chunks in a directory using multi-GPU inference.

    Args:
        chunk_dir: Directory containing chunk WAV files.
        output_path: Path to save transcription JSON.
        num_gpus: Number of GPUs to use. Auto-detected if None.
        batch_size: Batch size per GPU.

    Returns:
        Dict mapping chunk_id -> transcribed text.
    """
    chunk_path = Path(chunk_dir)
    wav_files = sorted(chunk_path.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {chunk_dir}")

    audio_paths = [str(f) for f in wav_files]
    logger.info(f"Found {len(audio_paths)} chunks to transcribe")

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, len(audio_paths))

    logger.info(f"Using {num_gpus} GPU(s)")

    if num_gpus <= 1:
        # Single GPU - run directly
        _transcribe_shard(0, audio_paths, output_path, batch_size)
    else:
        # Sort by file size (proxy for duration) for balanced sharding
        audio_paths.sort(key=lambda p: os.path.getsize(p))

        # Round-robin shard assignment for balanced duration
        shards = [[] for _ in range(num_gpus)]
        for i, path in enumerate(audio_paths):
            shards[i % num_gpus].append(path)

        out_dir = Path(output_path).parent
        shard_outputs = [str(out_dir / f"_transcriptions_shard_{i}.json") for i in range(num_gpus)]

        # Launch workers
        mp.set_start_method("spawn", force=True)
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_transcribe_shard,
                args=(gpu_id, shards[gpu_id], shard_outputs[gpu_id], batch_size),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge results
        merged = {}
        for shard_path in shard_outputs:
            if Path(shard_path).exists():
                with open(shard_path, "r", encoding="utf-8") as f:
                    merged.update(json.load(f))
                Path(shard_path).unlink()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    # Load and return
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)
