# Georgian Speech Corpus Pipeline

## Overview
Pipeline for building a large-scale parallel audio-text corpus for TTS training from long-form Georgian speech recordings.

## Pipeline
1. **VAD chunking** — Silero VAD splits long recordings into segments (≤35s) for ASR batching
2. **ASR transcription** — Multi-GPU Meta Omnilingual ASR (7B, ~2% CER on Georgian)
3. **CTC forced alignment** — Georgian FastConformer for word-level timestamps
4. **Sentence splitting** — Split at sentence boundaries using pauses + punctuation
5. **Text alignment** — Optional alignment to reference text for punctuation transfer
6. **Quality filtering** — Georgian ratio, chars/sec, duration bounds

## Hardware
2x RTX A6000 (96 GB VRAM total) — dual-GPU ASR with batched inference.
