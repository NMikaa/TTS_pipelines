# Georgian TTS Clean Dataset

Cleaned and filtered subset of Mozilla Common Voice Georgian (~71k raw clips) for TTS fine-tuning.

## Quality Pipeline

6-stage modular pipeline (`shared/data/quality/`), inspired by the Catalan TTS paper (arXiv 2410.13357) and Emilia:

1. **Standardize** — Resample to 24kHz mono, normalize loudness to -23 LUFS, filter duration [0.5s, 30s]
2. **Enhance** — VoiceFixer restoration + sox spectral noise subtraction
3. **NISQA filter** — NISQA MOS scoring (threshold=0.0, scored only, no filtering at this stage)
4. **Duration outlier** — Emilia-style IQR filter on per-character duration (catches bad transcripts, tempo anomalies)
5. **Transcript verify** — Round-trip CER via Meta Omnilingual ASR 7B (`omniASR_LLM_7B_v2`), drop if CER > 0.20
6. **Speaker select** — Keep speakers with >= 1400s total audio

### Pipeline Results

| Stage | Kept | Dropped | Notes |
|-------|------|---------|-------|
| Raw input | 40,769 | — | Common Voice Georgian |
| Standardize | 40,769 | 0 | All within duration bounds |
| Enhance | 40,769 | 0 | VoiceFixer + sox noisered |
| NISQA filter | 40,769 | 0 | Scored only (threshold=0.0) |
| Duration outlier | 39,437 | 1,332 | IQR on char_duration |
| Transcript verify | 38,997 | 440 | CER <= 0.20 |
| Speaker select | 38,997 | 0 | 14 speakers, all >= 1400s |

## Primary Dataset (NISQA MOS > 3.0)

Post-pipeline, an additional quality filter was applied:
- **NISQA MOS > 3.0** — drops noisy/low-quality recordings
- **Speaker duration >= 10 min** — drops speakers 9 and 13 (insufficient data after MOS filtering)

Result: **21,421 samples, 12 speakers, 35.0 hours**

## Data Splits

| Split | Samples | Duration | Purpose | Selection method |
|-------|---------|----------|---------|-----------------|
| `train_manifest.json` | 20,300 | 33.2h | Model training | Remainder after eval + refs |
| `eval_manifest.json` | 1,001 | 1.6h | Loss monitoring, overfitting detection | Random stratified by speaker (~4.7% per speaker) |
| `speaker_refs_manifest.json` | 120 | 0.18h | Speaker similarity evaluation | Top 10 by MOS per speaker (12 speakers x 10) |

- **Train and eval** have matched NISQA MOS distribution (mean ~3.95, std ~0.55)
- **Speaker refs** are highest-quality clips (MOS >= 4.37, mean 4.92) for clean speaker identity anchors
- All 12 speakers represented proportionally across all splits

## Speaker Statistics (after MOS > 3.0 filter)

| Speaker | Train | Eval | Refs | Total Hours | Avg MOS | Avg CER |
|---------|-------|------|------|-------------|---------|---------|
| 1 | 5,407 | 266 | 10 | 8.80 | 3.60 | 0.039 |
| 4 | 3,078 | 152 | 10 | 5.30 | 4.38 | 0.035 |
| 3 | 2,821 | 139 | 10 | 5.26 | 4.09 | 0.027 |
| 5 | 2,464 | 121 | 10 | 3.59 | 4.07 | 0.035 |
| 6 | 1,473 | 73 | 10 | 2.85 | 4.19 | 0.041 |
| 8 | 1,067 | 53 | 10 | 2.11 | 4.05 | 0.031 |
| 2 | 1,100 | 54 | 10 | 1.81 | 3.70 | 0.045 |
| 7 | 1,068 | 53 | 10 | 1.80 | 3.88 | 0.030 |
| 11 | 509 | 25 | 10 | 0.97 | 3.83 | 0.033 |
| 12 | 569 | 28 | 10 | 0.97 | 4.01 | 0.046 |
| 10 | 438 | 22 | 10 | 0.84 | 3.50 | 0.062 |
| 14 | 306 | 15 | 10 | 0.68 | 3.99 | 0.039 |

## Manifest Format (JSONL)

Each line is a JSON object:
```json
{
  "id": "common_voice_ka_37382284",
  "audio_path": "data/clean/audio_clean/common_voice_ka_37382284.wav",
  "text": "Georgian text here",
  "speaker_id": "14",
  "duration": 9.468,
  "nisqa_mos": 4.592,
  "nisqa_noisiness": 4.549,
  "nisqa_discontinuity": 4.424,
  "nisqa_coloration": 4.196,
  "nisqa_loudness": 4.422,
  "char_duration": 0.093,
  "asr_cer": 0.018,
  "asr_text": "ASR transcription for verification"
}
```

## Audio Format
- Sample rate: 24,000 Hz
- Channels: Mono
- Loudness: -23 LUFS normalized
- Format: WAV (16-bit PCM)
- Duration: 0.5s - 30s per clip

## S3 Location

Bucket: `ttsopensource`, region: `eu-central-1`, prefix: `tts-georgian/`

| File | Description |
|------|-------------|
| `clean_audio.tar.gz` | All 38,997 clean audio files (24kHz WAV) |
| `clean_manifest.json` | Full pipeline output manifest |
| `nisqa_scores.json` | NISQA scores for all entries |
| `primary_manifest.json` | MOS > 3.0 + speaker filtered (21,421 entries) |
| `train_manifest.json` | Training split (20,300 entries) |
| `eval_manifest.json` | Evaluation split (1,001 entries) |
| `speaker_refs_manifest.json` | Speaker reference clips (120 entries) |
