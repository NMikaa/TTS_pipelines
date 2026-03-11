---
language:
- ka
license: cc0-1.0
task_categories:
- text-to-speech
tags:
- georgian
- tts
- common-voice
- speech-synthesis
size_categories:
- 10K<n<100K
---

# Common Voice Georgian — Cleaned for TTS

A high-quality subset of [Mozilla Common Voice Georgian](https://commonvoice.mozilla.org/en/datasets) cleaned and filtered specifically for text-to-speech fine-tuning.

## Dataset Summary

| | |
|---|---|
| **Samples** | 21,421 |
| **Duration** | 35.0 hours |
| **Speakers** | 12 |
| **Sample rate** | 24 kHz mono WAV |
| **Language** | Georgian (kat) |
| **Source** | Mozilla Common Voice 19.0 |
| **License** | CC-0 (public domain) |

## Quality Pipeline

The dataset was cleaned from ~71K raw Common Voice recordings through a 6-stage pipeline:

1. **Standardize** — Resample to 24 kHz mono, normalize loudness to −23 LUFS, filter duration to [0.5s, 30s]
2. **Enhance** — VoiceFixer audio restoration + Sox spectral noise subtraction
3. **NISQA Filter** — NISQA MOS ≥ 3.0 (neural speech quality assessment)
4. **Duration Outlier** — IQR-based character duration filter (removes misaligned/rushed/slow speech)
5. **Transcript Verify** — Round-trip ASR (Meta Omnilingual 7B, 1.9% CER on Georgian) with CER ≤ 0.20 threshold
6. **Speaker Select** — Keep speakers with ≥ 1,400 seconds total audio

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Common Voice clip ID |
| `audio` | Audio | 24 kHz mono WAV |
| `text` | string | Georgian transcript |
| `speaker_id` | string | Anonymized speaker ID (0–11) |
| `duration` | float | Duration in seconds |

## Speaker Distribution

| Speaker | Samples | Duration |
|---------|---------|----------|
| 0 | 5,683 | 8.8h |
| 1 | 1,164 | 1.8h |
| 2 | 2,970 | 5.3h |
| 3 | 3,240 | 5.3h |
| 4 | 2,595 | 3.6h |
| 5 | 1,556 | 2.8h |
| 6 | 1,131 | 1.8h |
| 7 | 1,130 | 2.1h |
| 8 | 470 | 0.8h |
| 9 | 544 | 1.0h |
| 10 | 607 | 1.0h |
| 11 | 331 | 0.7h |

## Statistics

- **Duration**: min 2.4s, mean 5.9s, max 10.6s

## Usage

```python
from datasets import load_dataset

ds = load_dataset("NMikka/Common-Voice-Geo-Cleaned")

# Access a sample
sample = ds["train"][0]
print(sample["text"])       # Georgian text
print(sample["audio"])      # {'array': [...], 'sampling_rate': 24000}
print(sample["speaker_id"]) # Speaker ID (0-11)
```

## Citation

If you use this dataset, please cite Mozilla Common Voice:

```bibtex
@inproceedings{ardila2020common,
  title={Common Voice: A Massively-Multilingual Speech Corpus},
  author={Ardila, Rosana and others},
  booktitle={LREC},
  year={2020}
}
```

## Part of the Georgian TTS Benchmark

This dataset was created as part of the first Georgian TTS benchmark — a comparative study of 4 open-source TTS architectures (F5-TTS, Orpheus, Qwen3-TTS, CSM-1B) fine-tuned on the same data and evaluated with the same metrics.
