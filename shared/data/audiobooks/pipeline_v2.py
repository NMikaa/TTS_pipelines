#!/usr/bin/env python3
"""
Georgian speech corpus pipeline v2.

Flow: VAD (large chunks) → ASR → CTC forced alignment → sentence split → filter → manifest

Key features:
- VAD chunks are large (30-60s) — just for ASR batching, NOT final segments
- CTC forced alignment gives word-level timestamps
- Final splits at sentence boundaries (pauses + punctuation)
- Produces natural sentence-length segments (3-25s)
- Multi-GPU ASR with persistent workers
- Optional reference text alignment for punctuation transfer

Usage:
    python -m shared.data.audiobooks.pipeline_v2 \
        --wav "input.wav" \
        --output-dir ./data/output
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from rapidfuzz.distance import Levenshtein as _rf_lev
logger = logging.getLogger("pipeline_v2")

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000       # FastConformer and Silero VAD both want 16kHz
NEMO_MODEL_PATH = "/root/TTS_pipelines/fast_conformer_georgian.nemo"
TARGET_SR = 22050         # Output sample rate for TTS training
MIN_SEGMENT_SEC = 3.0     # Minimum final segment duration
MAX_SEGMENT_SEC = 25.0    # Maximum final segment duration
SENTENCE_PAUSE_SEC = 0.4  # Pause threshold for sentence boundary
WORD_PAUSE_SEC = 0.15     # Pause threshold for word boundary

# VAD settings for initial large chunking
VAD_CHUNK_MIN_SEC = 3.0
VAD_CHUNK_MAX_SEC = 35.0   # omniASR caps at 40s — keep margin
VAD_MIN_SILENCE_MS = 400
VAD_SPEECH_PAD_MS = 100

# Filter thresholds
MIN_GEORGIAN_RATIO = 0.80
MIN_CHARS_PER_SEC = 3.0
MAX_CHARS_PER_SEC = 25.0
MIN_TEXT_CHARS = 10


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class WordTiming:
    word: str
    start_sec: float
    end_sec: float
    score: float  # alignment confidence


@dataclass
class Segment:
    segment_id: str
    text: str             # final text (punctuated if available)
    asr_text: str         # raw ASR text
    book_text: str        # matched book text (empty if mismatched)
    start_sec: float      # in original audio
    end_sec: float
    duration_sec: float
    word_timings: list     # list of WordTiming
    cer: float
    speaker_id: str


# ─── NeMo model loading ───────────────────────────────────────────────────────

_nemo_model = None

def _load_nemo_model(device: str = "cuda:0"):
    """Load Georgian FastConformer CTC model (cached)."""
    global _nemo_model
    if _nemo_model is None:
        import nemo.collections.asr as nemo_asr
        logger.info(f"Loading Georgian FastConformer from {NEMO_MODEL_PATH}...")
        _nemo_model = nemo_asr.models.ASRModel.restore_from(
            NEMO_MODEL_PATH, map_location=device
        )
        _nemo_model.eval()
        logger.info("  Model loaded.")
    return _nemo_model


def _free_nemo_model():
    global _nemo_model
    if _nemo_model is not None:
        del _nemo_model
        _nemo_model = None
        torch.cuda.empty_cache()


# ─── Step 1: Load audio ──────────────────────────────────────────────────────

def load_audio(wav_path: str) -> tuple[torch.Tensor, int]:
    """Load audio, convert to mono, return (waveform, sample_rate)."""
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    return torchaudio.transforms.Resample(orig_sr, target_sr)(waveform)


# ─── Step 2: VAD chunking (large chunks for ASR) ─────────────────────────────

def vad_chunk(waveform_16k: torch.Tensor) -> list[dict]:
    """
    Run Silero VAD to get speech regions. Merge into chunks up to VAD_CHUNK_MAX_SEC.
    These are NOT final segments — just for batching ASR.
    """
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]

    wav = waveform_16k.squeeze()
    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        min_speech_duration_ms=500,
    )

    # Convert to seconds
    segments = [
        {"start": ts["start"] / SAMPLE_RATE, "end": ts["end"] / SAMPLE_RATE}
        for ts in timestamps
    ]

    if not segments:
        return []

    # Merge close segments, respecting max duration
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        merged_duration = seg["end"] - merged[-1]["start"]
        if merged_duration <= VAD_CHUNK_MAX_SEC and gap < 1.5:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Split any chunks still over the max (shouldn't happen often, but safety)
    final = []
    for chunk in merged:
        dur = chunk["end"] - chunk["start"]
        if dur <= VAD_CHUNK_MAX_SEC:
            if dur >= VAD_CHUNK_MIN_SEC:
                final.append(chunk)
            elif final:
                # Merge short chunk into previous if combined still fits
                if final[-1]["end"] - final[-1]["start"] + dur <= VAD_CHUNK_MAX_SEC:
                    final[-1]["end"] = chunk["end"]
                else:
                    final.append(chunk)  # keep even if short
            else:
                final.append(chunk)
        else:
            # Re-run VAD with tighter settings on this region
            start_sample = int(chunk["start"] * SAMPLE_RATE)
            end_sample = int(chunk["end"] * SAMPLE_RATE)
            sub_wav = wav[start_sample:end_sample]

            sub_ts = get_speech_timestamps(
                sub_wav, model,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=200,  # tighter
                speech_pad_ms=50,
                min_speech_duration_ms=500,
            )

            if len(sub_ts) <= 1:
                # Can't split — just take MAX_SEC windows
                offset = 0.0
                while offset < dur:
                    end = min(offset + VAD_CHUNK_MAX_SEC, dur)
                    final.append({
                        "start": chunk["start"] + offset,
                        "end": chunk["start"] + end,
                    })
                    offset = end
            else:
                # Re-merge sub-segments respecting max
                sub_segs = [
                    {"start": chunk["start"] + ts["start"] / SAMPLE_RATE,
                     "end": chunk["start"] + ts["end"] / SAMPLE_RATE}
                    for ts in sub_ts
                ]
                current = sub_segs[0].copy()
                for ss in sub_segs[1:]:
                    if ss["end"] - current["start"] <= VAD_CHUNK_MAX_SEC:
                        current["end"] = ss["end"]
                    else:
                        final.append(current)
                        current = ss.copy()
                final.append(current)

    return final


# ─── Step 3: ASR transcription ───────────────────────────────────────────────

def _asr_worker(
    gpu_id: int,
    chunk_paths: list[str],
    chunk_indices: list[int],
    output_json_path: str,
):
    """Worker process: load omniASR on one GPU and transcribe assigned chunks."""
    import json as _json
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(message)s",
    )
    log = _logging.getLogger(f"asr_gpu{gpu_id}")

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    log.info(f"Loading omniASR_LLM_7B on cuda:{gpu_id} ({len(chunk_paths)} chunks)...")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B", device=f"cuda:{gpu_id}")

    results = {}
    batch_size = 32  # ~14GB model (bf16) + ~32GB activations → ~46GB of 49GB
    for i in range(0, len(chunk_paths), batch_size):
        batch = chunk_paths[i : i + batch_size]
        batch_idx = chunk_indices[i : i + batch_size]
        try:
            texts = pipeline.transcribe(batch, lang=["kat_Geor"] * len(batch), batch_size=len(batch))
            for idx, text in zip(batch_idx, texts):
                results[str(idx)] = text if text else ""
        except Exception as e:
            log.error(f"Batch failed: {e}, falling back to single-file")
            for p, idx in zip(batch, batch_idx):
                try:
                    text = pipeline.transcribe([p], lang=["kat_Geor"], batch_size=1)
                    results[str(idx)] = text[0] if text else ""
                except Exception as e2:
                    log.error(f"  Failed chunk {idx}: {e2}")
                    results[str(idx)] = ""

        done = min(i + batch_size, len(chunk_paths))
        if done % 10 == 0 or done == len(chunk_paths):
            log.info(f"  Progress: {done}/{len(chunk_paths)}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, ensure_ascii=False)

    log.info(f"Done. {len(results)} transcriptions saved.")


def transcribe_chunks_asr(
    waveform_mono: torch.Tensor,
    sr: int,
    vad_chunks: list[dict],
    num_gpus: int = 2,
) -> list[str]:
    """Transcribe VAD chunks using omniASR on multiple GPUs in parallel.

    waveform_mono must be (1, samples) mono at original sample rate.
    Uses mp.Process to run one omniASR instance per GPU.
    """
    import torch.multiprocessing as mp
    import json as _json

    num_gpus = min(num_gpus, torch.cuda.device_count(), max(1, len(vad_chunks)))

    # Save all chunks as temp WAV files
    tmp_dir = "/tmp/_asr_chunks"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_paths = []
    for i, chunk in enumerate(vad_chunks):
        start_sample = int(chunk["start"] * sr)
        end_sample = min(int(chunk["end"] * sr), waveform_mono.shape[1])
        chunk_wav = waveform_mono[:, start_sample:end_sample]
        tmp_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        torchaudio.save(tmp_path, chunk_wav, sr)
        tmp_paths.append(tmp_path)

    if num_gpus <= 1:
        # Single GPU fallback
        shard_out = os.path.join(tmp_dir, "_shard_0.json")
        _asr_worker(0, tmp_paths, list(range(len(tmp_paths))), shard_out)
        with open(shard_out) as f:
            results = _json.load(f)
        os.unlink(shard_out)
    else:
        # Round-robin shard by file size for balanced workload
        sizes = [os.path.getsize(p) for p in tmp_paths]
        indexed = sorted(enumerate(tmp_paths), key=lambda x: sizes[x[0]])

        shard_paths = [[] for _ in range(num_gpus)]
        shard_indices = [[] for _ in range(num_gpus)]
        for rank, (orig_idx, path) in enumerate(indexed):
            gpu = rank % num_gpus
            shard_paths[gpu].append(path)
            shard_indices[gpu].append(orig_idx)

        shard_outputs = [os.path.join(tmp_dir, f"_shard_{g}.json") for g in range(num_gpus)]

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

        logger.info(f"Launching ASR on {num_gpus} GPUs: {[len(s) for s in shard_paths]} chunks each")

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_asr_worker,
                args=(gpu_id, shard_paths[gpu_id], shard_indices[gpu_id], shard_outputs[gpu_id]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Merge results
        results = {}
        for shard_out in shard_outputs:
            if os.path.exists(shard_out):
                with open(shard_out) as f:
                    results.update(_json.load(f))
                os.unlink(shard_out)

    # Reconstruct ordered list
    texts = []
    for i in range(len(vad_chunks)):
        texts.append(results.get(str(i), ""))

    # Cleanup temp WAV files
    for p in tmp_paths:
        if os.path.exists(p):
            os.unlink(p)

    return texts


# ─── Step 4: CTC forced alignment (Georgian FastConformer) ────────────────────

def _get_ctc_emission(model, waveform_16k: torch.Tensor, device: str) -> torch.Tensor:
    """Get CTC log-probabilities from NeMo FastConformer."""
    with torch.inference_mode():
        processed, processed_len = model.preprocessor(
            input_signal=waveform_16k.to(device),
            length=torch.tensor([waveform_16k.shape[1]], device=device),
        )
        encoded, encoded_len = model.encoder(
            audio_signal=processed, length=processed_len
        )
        log_probs = model.ctc_decoder(encoder_output=encoded)
    return log_probs[0].cpu()  # (T, V)


def ctc_align_chunk(
    waveform_16k: torch.Tensor,
    text: str,
    model,
    device: str = "cuda:0",
) -> list[WordTiming]:
    """
    CTC forced alignment using Georgian FastConformer.
    Native Georgian BPE tokens — no romanization needed.
    Returns word-level timings.
    """
    if not text.strip():
        return []

    tokenizer = model.tokenizer
    words = text.split()

    # Tokenize each word separately to track boundaries
    word_token_ranges = []
    all_tokens = []
    for word in words:
        toks = tokenizer.text_to_ids(word)
        start = len(all_tokens)
        all_tokens.extend(toks)
        word_token_ranges.append((start, len(all_tokens)))

    if not all_tokens:
        return []

    # Get CTC emission
    emission = _get_ctc_emission(model, waveform_16k, device)
    T, V = emission.shape
    frame_dur = waveform_16k.shape[1] / SAMPLE_RATE / T

    # Rearrange: NeMo puts blank at last index (V-1),
    # torchaudio forced_align expects blank at index 0
    blank_id = V - 1
    reordered = torch.zeros_like(emission)
    reordered[:, 0] = emission[:, blank_id]      # blank -> 0
    reordered[:, 1:] = emission[:, :blank_id]     # shift others +1

    # Adjust token IDs (+1 for the shift)
    adjusted_tokens = [t + 1 for t in all_tokens]
    tokens_tensor = torch.tensor([adjusted_tokens], dtype=torch.int32)

    try:
        aligned_seq, scores_seq = torchaudio.functional.forced_align(
            reordered.unsqueeze(0), tokens_tensor, blank=0
        )
    except Exception as e:
        logger.warning(f"CTC alignment failed: {e}")
        return []

    aligned_seq = aligned_seq[0].tolist()

    # Group consecutive same-token frames into spans
    token_spans = []
    current_val = aligned_seq[0]
    start_frame = 0
    for f in range(1, len(aligned_seq)):
        if aligned_seq[f] != current_val:
            if current_val != 0:  # not blank
                token_spans.append({
                    "start": start_frame,
                    "end": f,
                })
            current_val = aligned_seq[f]
            start_frame = f
    if current_val != 0:
        token_spans.append({"start": start_frame, "end": len(aligned_seq)})

    # Map token spans to words
    span_idx = 0
    word_timings = []
    for word, (tok_start, tok_end) in zip(words, word_token_ranges):
        n_toks = tok_end - tok_start
        if n_toks == 0 or span_idx >= len(token_spans):
            continue

        word_start = token_spans[span_idx]["start"]
        end_idx = min(span_idx + n_toks - 1, len(token_spans) - 1)
        word_end = token_spans[end_idx]["end"]
        span_idx += n_toks

        word_timings.append(WordTiming(
            word=word,
            start_sec=word_start * frame_dur,
            end_sec=word_end * frame_dur,
            score=1.0,
        ))

    return word_timings


# ─── Step 5: Sentence splitting ──────────────────────────────────────────────

def _is_sentence_end(word: str) -> bool:
    """Check if word ends with sentence-ending punctuation."""
    return bool(re.search(r'[.!?;:…]$', word))


def split_into_segments(
    word_timings: list[WordTiming],
    chunk_offset: float,
    book_id: str,
    segment_counter: int,
    has_book_text: bool = False,
) -> tuple[list[Segment], int]:
    """
    Split word timings into sentence-level segments.

    Strategy:
    - Primary: split at sentence-ending punctuation + pause > 0.3s
    - Secondary: split at any pause > SENTENCE_PAUSE_SEC (0.4s) if segment too long
    - Enforce MIN_SEGMENT_SEC / MAX_SEGMENT_SEC bounds
    """
    if not word_timings:
        return [], segment_counter

    segments = []
    current_words = []
    current_start = None

    for i, wt in enumerate(word_timings):
        # Offset to original audio timeline
        wt_abs = WordTiming(
            word=wt.word,
            start_sec=wt.start_sec + chunk_offset,
            end_sec=wt.end_sec + chunk_offset,
            score=wt.score,
        )

        if current_start is None:
            current_start = wt_abs.start_sec
        current_words.append(wt_abs)

        current_duration = wt_abs.end_sec - current_start

        # Look at gap to next word
        if i < len(word_timings) - 1:
            next_wt = word_timings[i + 1]
            next_start = next_wt.start_sec + chunk_offset
            pause = next_start - wt_abs.end_sec
        else:
            pause = 999.0  # end of chunk — always split

        should_split = False

        # Rule 1: sentence-ending punct + meaningful pause
        if _is_sentence_end(wt.word) and pause > 0.2 and current_duration >= MIN_SEGMENT_SEC:
            should_split = True

        # Rule 2: long pause even without punctuation
        elif pause > SENTENCE_PAUSE_SEC and current_duration >= MIN_SEGMENT_SEC:
            should_split = True

        # Rule 3: segment too long — force split at next pause
        elif current_duration > MAX_SEGMENT_SEC and pause > WORD_PAUSE_SEC:
            should_split = True

        # Rule 4: end of chunk
        elif i == len(word_timings) - 1 and current_duration >= MIN_SEGMENT_SEC:
            should_split = True

        if should_split and current_words:
            text = " ".join(w.word for w in current_words)
            seg = Segment(
                segment_id=f"{book_id}_seg_{segment_counter:05d}",
                text=text,
                asr_text=text,  # will be updated if book text available
                book_text="",
                start_sec=current_start,
                end_sec=current_words[-1].end_sec,
                duration_sec=current_words[-1].end_sec - current_start,
                word_timings=[asdict(w) for w in current_words],
                cer=0.0,
                speaker_id="",
            )
            segments.append(seg)
            segment_counter += 1
            current_words = []
            current_start = None

    # Handle leftover words
    if current_words:
        duration = current_words[-1].end_sec - current_words[0].start_sec
        if duration >= MIN_SEGMENT_SEC / 2 and segments:
            # Merge with previous if too short
            if duration < MIN_SEGMENT_SEC and segments:
                prev = segments[-1]
                merged_text = prev.text + " " + " ".join(w.word for w in current_words)
                merged_dur = current_words[-1].end_sec - prev.start_sec
                if merged_dur <= MAX_SEGMENT_SEC:
                    segments[-1] = Segment(
                        segment_id=prev.segment_id,
                        text=merged_text,
                        asr_text=merged_text,
                        book_text="",
                        start_sec=prev.start_sec,
                        end_sec=current_words[-1].end_sec,
                        duration_sec=merged_dur,
                        word_timings=prev.word_timings + [asdict(w) for w in current_words],
                        cer=0.0,
                        speaker_id="",
                    )
                else:
                    # Just make it its own segment even if short
                    text = " ".join(w.word for w in current_words)
                    segments.append(Segment(
                        segment_id=f"{book_id}_seg_{segment_counter:05d}",
                        text=text, asr_text=text, book_text="",
                        start_sec=current_words[0].start_sec,
                        end_sec=current_words[-1].end_sec,
                        duration_sec=duration,
                        word_timings=[asdict(w) for w in current_words],
                        cer=0.0, speaker_id="",
                    ))
                    segment_counter += 1
            else:
                text = " ".join(w.word for w in current_words)
                segments.append(Segment(
                    segment_id=f"{book_id}_seg_{segment_counter:05d}",
                    text=text, asr_text=text, book_text="",
                    start_sec=current_words[0].start_sec,
                    end_sec=current_words[-1].end_sec,
                    duration_sec=duration,
                    word_timings=[asdict(w) for w in current_words],
                    cer=0.0, speaker_id="",
                ))
                segment_counter += 1

    return segments, segment_counter


# ─── Step 6: Book text alignment & punctuation transfer ──────────────────────

def _normalize(text: str) -> str:
    """Strip punctuation for matching."""
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _levenshtein_distance(s1: str, s2: str) -> int:
    return _rf_lev.distance(s1, s2)


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _rf_lev.normalized_distance(ref, hyp)


def align_segments_to_book(
    segments: list[Segment],
    book_text: str,
) -> tuple[list[Segment], bool]:
    """
    Align segment ASR texts to book text for punctuation transfer.
    Returns (updated_segments, text_matched).
    """
    if not book_text or not book_text.strip():
        return segments, False

    # Clean book text
    clean_book = re.sub(r'^#+\s+', '', book_text, flags=re.MULTILINE)
    clean_book = re.sub(r'\s+', ' ', clean_book).strip()
    norm_book = _normalize(book_text)

    # Quick mismatch probe: check first 5 segments
    probe_cers = []
    for seg in segments[:min(5, len(segments))]:
        if not seg.asr_text:
            continue
        norm_asr = _normalize(seg.asr_text)
        # Simple sliding window search
        best_cer = 1.0
        for start in range(0, min(len(norm_book), len(norm_asr) * 10), len(norm_asr) // 2 + 1):
            end = start + len(norm_asr)
            if end > len(norm_book):
                break
            c = _cer(norm_book[start:end], norm_asr)
            best_cer = min(best_cer, c)
        probe_cers.append(best_cer)

    avg_probe_cer = sum(probe_cers) / len(probe_cers) if probe_cers else 1.0
    if avg_probe_cer > 0.4:
        logger.warning(f"Text mismatch detected (avg probe CER={avg_probe_cer:.3f}). Using ASR-only.")
        return segments, False

    logger.info(f"Text match OK (avg probe CER={avg_probe_cer:.3f}). Aligning for punctuation.")

    # Full alignment: sequential search through book
    cursor = 0
    for seg in segments:
        if not seg.asr_text:
            continue

        norm_asr = _normalize(seg.asr_text)
        if not norm_asr:
            continue

        # Search window
        search_start = max(0, cursor - 200)
        search_end = min(len(norm_book), cursor + len(norm_asr) * 5 + 1000)

        best_cer = 1.0
        best_start = cursor
        best_end = min(cursor + len(norm_asr), len(norm_book))

        # N-gram anchor search
        step = max(1, len(norm_asr) // 10)
        for start in range(search_start, search_end, step):
            for factor in [0.9, 1.0, 1.1]:
                end = start + int(len(norm_asr) * factor)
                if end > len(norm_book):
                    break
                c = _cer(norm_book[start:end], norm_asr)
                if c < best_cer:
                    best_cer = c
                    best_start = start
                    best_end = end

        # Map normalized positions back to clean book (approximate)
        ratio = len(clean_book) / max(len(norm_book), 1)
        orig_start = int(best_start * ratio)
        orig_end = int(best_end * ratio)

        # Snap to word boundaries
        while orig_start > 0 and not clean_book[orig_start - 1].isspace():
            orig_start -= 1
        while orig_end < len(clean_book) and not clean_book[orig_end - 1:orig_end].isspace():
            orig_end += 1

        orig_start = max(0, orig_start)
        orig_end = min(len(clean_book), orig_end)

        matched_book = clean_book[orig_start:orig_end].strip()
        seg.book_text = matched_book
        seg.cer = best_cer

        # Transfer punctuation: word-level
        if best_cer < 0.3 and matched_book:
            seg.text = _transfer_punct(seg.asr_text, matched_book)
        # else keep ASR text

        if best_cer < 0.5:
            cursor = best_end

    cers = [s.cer for s in segments if s.asr_text and s.book_text]
    if cers:
        logger.info(f"Alignment: avg CER={sum(cers)/len(cers):.3f}, "
                     f"good (<0.2): {sum(1 for c in cers if c < 0.2)}/{len(cers)}")

    return segments, True


def _transfer_punct(asr_text: str, book_text: str) -> str:
    """Transfer punctuation from book text to ASR words."""
    asr_words = asr_text.split()
    book_words = book_text.split()
    if not asr_words or not book_words:
        return asr_text

    # Simple word-level DP alignment
    n, m = len(asr_words), len(book_words)
    asr_s = [re.sub(r'[^\w]', '', w, flags=re.UNICODE).lower() for w in asr_words]
    book_s = [re.sub(r'[^\w]', '', w, flags=re.UNICODE).lower() for w in book_words]

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if asr_s[i-1] == book_s[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    # Backtrace
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if asr_s[i-1] == book_s[j-1] else 1
        if dp[i][j] == dp[i-1][j-1] + cost:
            pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    result = list(asr_words)
    for ai, bi in pairs:
        a_stripped = asr_s[ai]
        b_stripped = book_s[bi]
        if a_stripped and b_stripped:
            word_cer = _levenshtein_distance(a_stripped, b_stripped) / max(len(a_stripped), len(b_stripped))
            if word_cer < 0.3:
                result[ai] = book_words[bi]

    return ' '.join(result)


# ─── Step 7: Filter ──────────────────────────────────────────────────────────

def _georgian_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    geo = sum(1 for c in alpha if '\u10a0' <= c <= '\u10ff' or '\u2d00' <= c <= '\u2d2f')
    return geo / len(alpha)


def filter_segments(segments: list[Segment]) -> tuple[list[Segment], dict]:
    """Apply quality filters. Returns (passed_segments, summary)."""
    passed = []
    reasons = {}

    for seg in segments:
        reason = ""
        text = seg.asr_text

        if len(text.strip()) < MIN_TEXT_CHARS:
            reason = "short_text"
        elif seg.duration_sec < MIN_SEGMENT_SEC:
            reason = "short_duration"
        elif seg.duration_sec > MAX_SEGMENT_SEC * 1.1:
            reason = "long_duration"
        elif _georgian_ratio(text) < MIN_GEORGIAN_RATIO:
            reason = "low_georgian"
        elif seg.duration_sec > 0:
            cps = len(text) / seg.duration_sec
            if cps < MIN_CHARS_PER_SEC:
                reason = "low_cps"
            elif cps > MAX_CHARS_PER_SEC:
                reason = "high_cps"

        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            passed.append(seg)

    logger.info(f"Filter: {len(passed)}/{len(segments)} passed ({100*len(passed)/max(len(segments),1):.1f}%)")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  Rejected: {r}: {c}")

    return passed, {"total": len(segments), "passed": len(passed), "reasons": reasons}


# ─── Step 8: Save segments as WAV + manifest ─────────────────────────────────

def save_segments(
    segments: list[Segment],
    waveform: torch.Tensor,
    sr: int,
    output_dir: Path,
) -> list[dict]:
    """Save each segment as a WAV file and build manifest entries."""
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Resample to target SR for TTS
    if sr != TARGET_SR:
        waveform_out = resample(waveform, sr, TARGET_SR)
        out_sr = TARGET_SR
    else:
        waveform_out = waveform
        out_sr = sr

    manifest = []
    for seg in segments:
        start_sample = int(seg.start_sec * out_sr)
        end_sample = min(int(seg.end_sec * out_sr), waveform_out.shape[1])
        chunk_wav = waveform_out[:, start_sample:end_sample]

        wav_path = wavs_dir / f"{seg.segment_id}.wav"
        torchaudio.save(str(wav_path), chunk_wav, out_sr)

        manifest.append({
            "id": seg.segment_id,
            "audio_filepath": str(wav_path),
            "text": seg.text,
            "asr_text": seg.asr_text,
            "book_text": seg.book_text,
            "speaker_id": seg.speaker_id,
            "duration": seg.duration_sec,
            "cer": seg.cer,
        })

    return manifest


# ─── Main pipeline ────────────────────────────────────────────────────────────

def extract_narrator(wav_path: str) -> str:
    """Extract narrator from filename: title_author_narrator.wav"""
    stem = Path(wav_path).stem
    parts = stem.split('_')
    return parts[-1] if len(parts) >= 3 else "unknown"


def find_book_text(parquet_path: str, wav_path: str) -> str:
    """Try to find book text from parquet matching the wav filename."""
    df = pd.read_parquet(parquet_path)
    stem = Path(wav_path).stem
    title = stem.split('_')[0] if '_' in stem else stem

    # Try exact title match first
    match = df[df['title'] == title]
    if len(match) == 0:
        # Try contains
        match = df[df['title'].str.contains(title, na=False)]
    if len(match) == 0:
        logger.warning(f"No book text found for '{title}'")
        return ""
    return match.iloc[0]['text']


def process_book(
    wav_path: str,
    output_dir: str,
    parquet_path: str = None,
    book_text: str = None,
    gpu_align: int = 0,
) -> dict:
    """
    Process one long-form audio file end-to-end.

    Returns stats dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(wav_path).stem
    book_id = re.sub(r'[^\w]', '_', stem.split('_')[0], flags=re.UNICODE)[:60]
    narrator = extract_narrator(wav_path)

    logger.info(f"Processing: {stem}")
    logger.info(f"  Book ID: {book_id}, Narrator: {narrator}")

    t_total = time.time()

    # ── Load audio ──
    logger.info("Step 1: Loading audio...")
    t0 = time.time()
    waveform, sr = load_audio(wav_path)
    total_duration = waveform.shape[1] / sr
    waveform_16k = resample(waveform, sr, SAMPLE_RATE)
    logger.info(f"  Duration: {total_duration/60:.1f}min, SR: {sr}")
    load_time = time.time() - t0

    # ── VAD chunking ──
    logger.info("Step 2: VAD chunking (large chunks for ASR)...")
    t0 = time.time()
    vad_chunks = vad_chunk(waveform_16k)
    vad_duration = sum(c["end"] - c["start"] for c in vad_chunks)
    logger.info(f"  {len(vad_chunks)} VAD chunks, {vad_duration/60:.1f}min speech")
    vad_time = time.time() - t0

    # ── ASR transcription (dual-GPU) ──
    logger.info("Step 3: ASR transcription (multi-GPU)...")
    t0 = time.time()
    # Use mono waveform at original SR (omniASR handles resampling internally)
    num_gpus = torch.cuda.device_count()
    asr_texts = transcribe_chunks_asr(waveform, sr, vad_chunks, num_gpus=num_gpus)
    asr_time = time.time() - t0
    logger.info(f"  Transcribed {len(asr_texts)} chunks in {asr_time:.1f}s")

    # ── CTC forced alignment ──
    logger.info("Step 4: CTC forced alignment for word timestamps...")
    t0 = time.time()

    device = f"cuda:{gpu_align}"
    align_model = _load_nemo_model(device)

    all_segments = []
    seg_counter = 0

    for i, (chunk, text) in enumerate(zip(vad_chunks, asr_texts)):
        if not text.strip():
            continue

        # Extract chunk audio at 16kHz
        start_sample = int(chunk["start"] * SAMPLE_RATE)
        end_sample = int(chunk["end"] * SAMPLE_RATE)
        chunk_16k = waveform_16k[:, start_sample:end_sample]

        # CTC align using Georgian FastConformer
        word_timings = ctc_align_chunk(
            chunk_16k, text, align_model, device
        )

        if not word_timings:
            continue

        # Split into segments
        chunk_segments, seg_counter = split_into_segments(
            word_timings,
            chunk_offset=chunk["start"],
            book_id=book_id,
            segment_counter=seg_counter,
        )

        for seg in chunk_segments:
            seg.speaker_id = narrator

        all_segments.extend(chunk_segments)

    _free_nemo_model()

    align_time = time.time() - t0
    logger.info(f"  {len(all_segments)} raw segments from CTC alignment in {align_time:.1f}s")

    if not all_segments:
        logger.error("No segments produced!")
        return {"book_id": book_id, "status": "no_segments"}

    # ── Book text alignment (punctuation transfer) ──
    logger.info("Step 5: Book text alignment & punctuation transfer...")
    t0 = time.time()

    if book_text is None and parquet_path:
        book_text = find_book_text(parquet_path, wav_path)

    all_segments, text_matched = align_segments_to_book(all_segments, book_text or "")
    book_time = time.time() - t0

    # ── Filter ──
    logger.info("Step 6: Quality filtering...")
    t0 = time.time()
    passed_segments, filter_summary = filter_segments(all_segments)
    filter_time = time.time() - t0

    # ── Save ──
    logger.info("Step 7: Saving segments...")
    t0 = time.time()
    book_out = output_dir / book_id
    manifest = save_segments(passed_segments, waveform, sr, book_out)
    save_time = time.time() - t0

    # Write manifest
    manifest_path = book_out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Also write full manifest as JSON for inspection
    with open(book_out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t_total
    kept_duration = sum(s.duration_sec for s in passed_segments)
    durations = [s.duration_sec for s in passed_segments]

    stats = {
        "book_id": book_id,
        "narrator": narrator,
        "status": "complete",
        "wav_path": wav_path,
        "total_audio_min": total_duration / 60,
        "speech_min": vad_duration / 60,
        "raw_segments": len(all_segments),
        "passed_segments": len(passed_segments),
        "kept_duration_min": kept_duration / 60,
        "yield_pct": 100 * kept_duration / total_duration if total_duration > 0 else 0,
        "text_matched": text_matched,
        "filter_summary": filter_summary,
        "duration_stats": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "mean": float(np.mean(durations)) if durations else 0,
            "median": float(np.median(durations)) if durations else 0,
        },
        "timing": {
            "load_sec": load_time,
            "vad_sec": vad_time,
            "asr_sec": asr_time,
            "align_sec": align_time,
            "book_align_sec": book_time,
            "filter_sec": filter_time,
            "save_sec": save_time,
            "total_sec": total_time,
        },
        "manifest_path": str(manifest_path),
    }

    with open(book_out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {book_id}")
    logger.info(f"  Audio: {total_duration/60:.1f}min total, {vad_duration/60:.1f}min speech")
    logger.info(f"  Segments: {len(all_segments)} raw → {len(passed_segments)} passed")
    logger.info(f"  Kept: {kept_duration/60:.1f}min ({stats['yield_pct']:.1f}%)")
    if durations:
        logger.info(f"  Durations: {min(durations):.1f}-{max(durations):.1f}s, mean={np.mean(durations):.1f}s")
    logger.info(f"  Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"{'='*60}")

    return stats


# ─── Persistent ASR workers for batch processing ────────────────────────────

def _persistent_asr_worker(gpu_id: int, task_queue, result_queue):
    """
    Long-lived ASR worker: loads model once, processes jobs from queue.
    Sends (book_idx, chunk_idx, text) tuples back.
    Exits when it receives None sentinel.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(message)s",
    )
    log = _logging.getLogger(f"asr_gpu{gpu_id}")

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    log.info(f"Loading omniASR_LLM_7B on cuda:{gpu_id} (persistent worker)...")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B", device=f"cuda:{gpu_id}")
    log.info("Model loaded. Waiting for jobs...")

    while True:
        job = task_queue.get()
        if job is None:  # shutdown sentinel
            log.info("Shutdown signal received.")
            break

        book_idx, chunk_paths, chunk_indices = job
        log.info(f"Book {book_idx}: transcribing {len(chunk_paths)} chunks...")

        batch_size = 32
        for i in range(0, len(chunk_paths), batch_size):
            batch = chunk_paths[i : i + batch_size]
            batch_idx = chunk_indices[i : i + batch_size]
            try:
                texts = pipeline.transcribe(
                    batch, lang=["kat_Geor"] * len(batch), batch_size=len(batch)
                )
                for idx, text in zip(batch_idx, texts):
                    result_queue.put((book_idx, idx, text if text else ""))
            except Exception as e:
                log.error(f"Batch failed: {e}, falling back to single-file")
                for p, idx in zip(batch, batch_idx):
                    try:
                        text = pipeline.transcribe([p], lang=["kat_Geor"], batch_size=1)
                        result_queue.put((book_idx, idx, text[0] if text else ""))
                    except Exception as e2:
                        log.error(f"  Failed chunk {idx}: {e2}")
                        result_queue.put((book_idx, idx, ""))

        # Signal that this book's shard is done for this worker
        result_queue.put((book_idx, -1, f"DONE_GPU_{gpu_id}"))
        log.info(f"Book {book_idx}: done.")


def process_all_books(
    wav_paths: list[str],
    output_dir: str,
    parquet_path: str = None,
    num_gpus: int = None,
) -> list[dict]:
    """
    Process all audio files with persistent ASR workers.

    ASR models are loaded ONCE and reused across all books.
    Pipeline: main process does VAD → dispatches ASR to workers →
              collects results → CTC align → book align → filter → save.
    """
    import torch.multiprocessing as mp

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, 2)  # omniASR is large, 2 GPUs max

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet once
    parquet_df = None
    if parquet_path and os.path.exists(parquet_path):
        parquet_df = pd.read_parquet(parquet_path)

    # ── Start persistent ASR workers ──
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = []
    for gpu_id in range(num_gpus):
        w = mp.Process(target=_persistent_asr_worker, args=(gpu_id, task_queue, result_queue))
        w.start()
        workers.append(w)

    logger.info(f"Started {num_gpus} persistent ASR workers.")

    # ── Load NeMo alignment model in main process ──
    align_device = "cuda:0"
    align_model = _load_nemo_model(align_device)

    # Helper to look up book text
    def _find_text(wav_path):
        if parquet_df is None:
            return ""
        stem = Path(wav_path).stem
        title = stem.split('_')[0] if '_' in stem else stem
        match = parquet_df[parquet_df['title'] == title]
        if len(match) == 0:
            match = parquet_df[parquet_df['title'].str.contains(title, na=False)]
        if len(match) == 0:
            return ""
        return match.iloc[0]['text']

    tmp_dir = "/tmp/_asr_chunks"
    os.makedirs(tmp_dir, exist_ok=True)

    all_stats = []

    # ── Pipelined VAD: precompute next book while ASR runs ──
    from concurrent.futures import ThreadPoolExecutor

    def _prepare_book(wav_path, book_idx):
        """Load audio + VAD in background thread. Returns prepared data."""
        waveform, sr = load_audio(wav_path)
        total_duration = waveform.shape[1] / sr
        waveform_16k = resample(waveform, sr, SAMPLE_RATE)
        vad_chunks = vad_chunk(waveform_16k)
        vad_duration = sum(c["end"] - c["start"] for c in vad_chunks)

        # Save temp WAVs
        book_tmp_dir = os.path.join(tmp_dir, f"book_{book_idx}")
        os.makedirs(book_tmp_dir, exist_ok=True)
        tmp_paths = []
        for i, chunk in enumerate(vad_chunks):
            start_sample = int(chunk["start"] * sr)
            end_sample = min(int(chunk["end"] * sr), waveform.shape[1])
            chunk_wav = waveform[:, start_sample:end_sample]
            tmp_path = os.path.join(book_tmp_dir, f"chunk_{i:04d}.wav")
            torchaudio.save(tmp_path, chunk_wav, sr)
            tmp_paths.append(tmp_path)

        return {
            "waveform": waveform, "sr": sr, "waveform_16k": waveform_16k,
            "total_duration": total_duration, "vad_chunks": vad_chunks,
            "vad_duration": vad_duration, "tmp_paths": tmp_paths,
            "book_tmp_dir": book_tmp_dir,
        }

    # Filter to books that need processing
    pending = []
    for book_idx, wav_path in enumerate(wav_paths):
        stem = Path(wav_path).stem
        book_id = re.sub(r'[^\w]', '_', stem.split('_')[0], flags=re.UNICODE)[:60]
        book_out = output_dir / book_id
        if (book_out / "manifest.jsonl").exists():
            logger.info(f"[{book_idx+1}/{len(wav_paths)}] SKIP {book_id} (already done)")
            continue
        pending.append((book_idx, wav_path))

    logger.info(f"{len(pending)} books to process ({len(wav_paths) - len(pending)} already done)")

    executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None

    # Kick off prefetch for first book
    if pending:
        first_idx, first_path = pending[0]
        prefetch_future = executor.submit(_prepare_book, first_path, first_idx)

    try:
      for pi, (book_idx, wav_path) in enumerate(pending):
        stem = Path(wav_path).stem
        book_id = re.sub(r'[^\w]', '_', stem.split('_')[0], flags=re.UNICODE)[:60]
        book_out = output_dir / book_id
        narrator = extract_narrator(wav_path)

        logger.info(f"\n[{book_idx+1}/{len(wav_paths)}] Processing: {stem}")
        t_book = time.time()

        # ── Get prepared data (from prefetch or compute now) ──
        t0 = time.time()
        if prefetch_future is not None:
            prepared = prefetch_future.result()
            prefetch_future = None
        else:
            prepared = _prepare_book(wav_path, book_idx)
        prep_time = time.time() - t0

        waveform = prepared["waveform"]
        sr = prepared["sr"]
        waveform_16k = prepared["waveform_16k"]
        total_duration = prepared["total_duration"]
        vad_chunks = prepared["vad_chunks"]
        vad_duration = prepared["vad_duration"]
        tmp_paths = prepared["tmp_paths"]
        book_tmp_dir = prepared["book_tmp_dir"]

        logger.info(f"  Duration: {total_duration/60:.1f}min, "
                     f"VAD: {len(vad_chunks)} chunks, {vad_duration/60:.1f}min speech "
                     f"(prep {prep_time:.0f}s)")

        if not vad_chunks:
            logger.warning(f"  No speech detected, skipping.")
            all_stats.append({"book_id": book_id, "status": "no_speech"})
            continue

        # ── Kick off prefetch for NEXT book while we do ASR ──
        if pi + 1 < len(pending):
            next_idx, next_path = pending[pi + 1]
            prefetch_future = executor.submit(_prepare_book, next_path, next_idx)

        # ── Dispatch to ASR workers ──
        t0 = time.time()
        sizes = [os.path.getsize(p) for p in tmp_paths]
        indexed = sorted(enumerate(tmp_paths), key=lambda x: sizes[x[0]])

        shard_paths = [[] for _ in range(num_gpus)]
        shard_indices = [[] for _ in range(num_gpus)]
        for rank, (orig_idx, path) in enumerate(indexed):
            gpu = rank % num_gpus
            shard_paths[gpu].append(path)
            shard_indices[gpu].append(orig_idx)

        for gpu_id in range(num_gpus):
            task_queue.put((book_idx, shard_paths[gpu_id], shard_indices[gpu_id]))

        # Collect results
        asr_results = {}
        done_signals = 0
        while done_signals < num_gpus:
            bi, ci, text = result_queue.get()
            if ci == -1:
                done_signals += 1
            else:
                asr_results[ci] = text

        asr_texts = [asr_results.get(i, "") for i in range(len(vad_chunks))]
        asr_time = time.time() - t0

        # Cleanup temp files
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)
        try:
            os.rmdir(book_tmp_dir)
        except OSError:
            pass

        logger.info(f"  ASR: {len(asr_texts)} chunks in {asr_time:.0f}s")

        # ── CTC forced alignment ──
        t0 = time.time()
        all_segments = []
        seg_counter = 0

        for i, (chunk, text) in enumerate(zip(vad_chunks, asr_texts)):
            if not text.strip():
                continue
            start_sample = int(chunk["start"] * SAMPLE_RATE)
            end_sample = int(chunk["end"] * SAMPLE_RATE)
            chunk_16k = waveform_16k[:, start_sample:end_sample]

            word_timings = ctc_align_chunk(chunk_16k, text, align_model, align_device)
            if not word_timings:
                continue

            chunk_segments, seg_counter = split_into_segments(
                word_timings, chunk_offset=chunk["start"],
                book_id=book_id, segment_counter=seg_counter,
            )
            for seg in chunk_segments:
                seg.speaker_id = narrator
            all_segments.extend(chunk_segments)

        align_time = time.time() - t0
        logger.info(f"  CTC align: {len(all_segments)} segments in {align_time:.0f}s")

        if not all_segments:
            logger.warning(f"  No segments produced, skipping.")
            all_stats.append({"book_id": book_id, "status": "no_segments"})
            continue

        # ── Book text alignment ──
        t0 = time.time()
        book_text = _find_text(wav_path)
        all_segments, text_matched = align_segments_to_book(all_segments, book_text)
        book_time = time.time() - t0

        # ── Filter ──
        passed_segments, filter_summary = filter_segments(all_segments)

        # ── Save ──
        t0 = time.time()
        manifest = save_segments(passed_segments, waveform, sr, book_out)
        save_time = time.time() - t0

        manifest_path = book_out / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total_time = time.time() - t_book
        kept_duration = sum(s.duration_sec for s in passed_segments)

        stats = {
            "book_id": book_id,
            "narrator": narrator,
            "status": "complete",
            "wav_path": wav_path,
            "total_audio_min": total_duration / 60,
            "speech_min": vad_duration / 60,
            "raw_segments": len(all_segments),
            "passed_segments": len(passed_segments),
            "kept_duration_min": kept_duration / 60,
            "yield_pct": 100 * kept_duration / total_duration if total_duration > 0 else 0,
            "text_matched": text_matched,
            "timing": {
                "prep_sec": prep_time,
                "asr_sec": asr_time, "align_sec": align_time,
                "book_align_sec": book_time, "save_sec": save_time,
                "total_sec": total_time,
            },
        }
        with open(book_out / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        all_stats.append(stats)

        logger.info(f"  DONE: {len(passed_segments)} segments, "
                     f"{kept_duration/60:.1f}min kept ({stats['yield_pct']:.0f}%), "
                     f"{total_time:.0f}s total")

    finally:
        # ── Shutdown (always runs, even on crash) ──
        executor.shutdown(wait=False)
        for _ in range(num_gpus):
            task_queue.put(None)
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                w.kill()

        _free_nemo_model()

    # ── Summary ──
    completed = [s for s in all_stats if s.get("status") == "complete"]
    total_kept = sum(s["kept_duration_min"] for s in completed)
    total_audio = sum(s["total_audio_min"] for s in completed)
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE: {len(completed)}/{len(wav_paths)} books processed")
    logger.info(f"  Total audio: {total_audio/60:.1f}h, kept: {total_kept/60:.1f}h ({100*total_kept/max(total_audio,1):.0f}%)")
    logger.info(f"{'='*60}")

    # Save global summary
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    return all_stats


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Georgian speech corpus pipeline v2")
    parser.add_argument("--wav", type=str, default=None, help="Path to single WAV file")
    parser.add_argument("--wav-dir", type=str, default=None, help="Directory of WAV files (batch mode)")
    parser.add_argument("--parquet", type=str, default=None, help="Path to parquet with reference texts")
    parser.add_argument("--output-dir", type=str, default="./data/corpus_v2")
    parser.add_argument("--gpu-align", type=int, default=0, help="GPU for CTC alignment")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "pipeline.log"), mode="a"),
        ],
        force=True,
    )

    if args.wav_dir:
        # Batch mode: process all WAVs in directory
        wav_dir = Path(args.wav_dir)
        wav_paths = sorted(str(p) for p in wav_dir.glob("*.wav"))
        logger.info(f"Found {len(wav_paths)} WAV files in {wav_dir}")

        all_stats = process_all_books(
            wav_paths=wav_paths,
            output_dir=args.output_dir,
            parquet_path=args.parquet,
        )
        print(json.dumps({"books_processed": len(all_stats)}, indent=2))

    elif args.wav:
        # Single book mode
        stats = process_book(
            wav_path=args.wav,
            output_dir=args.output_dir,
            parquet_path=args.parquet,
            gpu_align=args.gpu_align,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    else:
        parser.error("Provide either --wav or --wav-dir")


if __name__ == "__main__":
    main()
