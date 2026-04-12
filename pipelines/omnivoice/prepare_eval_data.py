"""Prepare evaluation test lists from FLEURS Georgian and Common Voice Georgian.

Downloads the test sets, applies normalization, and writes JSONL test lists for the
evaluation pipeline. Run this once before running evaluate.py.

Usage:
    python pipelines/omnivoice/prepare_eval_data.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

EVAL_DIR = Path(__file__).parent / "eval_data"
EVAL_DIR.mkdir(exist_ok=True)
(EVAL_DIR / "test_lists").mkdir(exist_ok=True)
(EVAL_DIR / "cv_geo" / "audio").mkdir(parents=True, exist_ok=True)
(EVAL_DIR / "fleurs_ka" / "audio").mkdir(parents=True, exist_ok=True)
(EVAL_DIR / "fleurs_en" / "audio").mkdir(parents=True, exist_ok=True)


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from text_normalizer import normalize as norm_ka
    from text_normalizer_en import normalize as norm_en

    # =====================================================================
    # 1. CV-GEO test set (12 speakers, 10 clips each, voice cloning)
    # =====================================================================
    print("=== Downloading Common Voice Georgian Cleaned ===")
    cv = load_dataset("NMikka/Common-Voice-Geo-Cleaned", split="test")
    print(f"  {len(cv)} samples")

    by_speaker = defaultdict(list)
    for sample in cv:
        audio_path = EVAL_DIR / "cv_geo" / "audio" / f"{sample['id']}.wav"
        sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
        by_speaker[sample["speaker_id"]].append({
            "id": sample["id"],
            "text": sample["text"],
            "audio_path": str(audio_path.absolute()),
            "speaker_id": sample["speaker_id"],
        })

    # Build voice cloning pairs: clip[0] = ref, clips[1:] = targets
    vc_test = []
    for spk in sorted(by_speaker.keys()):
        clips = by_speaker[spk]
        ref = clips[0]
        for tgt in clips[1:]:
            vc_test.append({
                "id": f"cv_{spk}_{tgt['id']}",
                "text": tgt["text"],
                "ref_audio": ref["audio_path"],
                "ref_text": ref["text"],
                "gt_audio": tgt["audio_path"],
                "speaker_id": spk,
                "language_id": "ka",
                "language_name": "Georgian",
            })

    refs_by_speaker = {s: clips[0] for s, clips in by_speaker.items()}
    ref_list = [refs_by_speaker[s] for s in sorted(refs_by_speaker.keys())[:3]]

    with open(EVAL_DIR / "test_lists" / "cv_geo_voice_clone.jsonl", "w") as f:
        for e in vc_test:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote cv_geo_voice_clone.jsonl: {len(vc_test)} pairs")

    # =====================================================================
    # 2. FLEURS Georgian (979 samples, intelligibility test)
    # =====================================================================
    print("\n=== Downloading FLEURS Georgian (test) ===")
    fleurs_ka = load_dataset("google/fleurs", "ka_ge", split="test", trust_remote_code=True)
    print(f"  {len(fleurs_ka)} samples")

    fleurs_ka_test = []
    for i, sample in enumerate(fleurs_ka):
        audio_path = EVAL_DIR / "fleurs_ka" / "audio" / f"fleurs_ka_{i:04d}.wav"
        sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])

        raw = sample["transcription"]
        normalized = norm_ka(raw)

        ref = ref_list[i % 3]
        fleurs_ka_test.append({
            "id": f"fleurs_ka_{i:04d}",
            "text": normalized,
            "text_raw": raw,
            "ref_audio": ref["audio_path"],
            "ref_text": ref["text"],
            "gt_audio": str(audio_path.absolute()),
            "language_id": "ka",
            "language_name": "Georgian",
        })

    with open(EVAL_DIR / "test_lists" / "fleurs_ka.jsonl", "w") as f:
        for e in fleurs_ka_test:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote fleurs_ka.jsonl: {len(fleurs_ka_test)} samples")

    # =====================================================================
    # 3. FLEURS English (cross-lingual + regression)
    # =====================================================================
    print("\n=== Downloading FLEURS English (test) ===")
    fleurs_en = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    print(f"  {len(fleurs_en)} samples")

    # Cross-lingual: Georgian ref → English text (first 100)
    cross_test = []
    for i in range(min(100, len(fleurs_en))):
        sample = fleurs_en[i]
        audio_path = EVAL_DIR / "fleurs_en" / "audio" / f"fleurs_en_{i:04d}.wav"
        sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])
        ref = ref_list[i % 3]
        cross_test.append({
            "id": f"cross_en_{i:04d}",
            "text": norm_en(sample["transcription"]),
            "text_raw": sample["transcription"],
            "ref_audio": ref["audio_path"],
            "ref_text": ref["text"],
            "gt_audio": str(audio_path.absolute()),
            "language_id": "en",
            "language_name": "English",
        })
    with open(EVAL_DIR / "test_lists" / "fleurs_en_cross.jsonl", "w") as f:
        for e in cross_test:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote fleurs_en_cross.jsonl: {len(cross_test)} samples")

    # Regression: English ref → English text (samples 100-200)
    regress_test = []
    for i in range(100, min(200, len(fleurs_en))):
        sample = fleurs_en[i]
        audio_path = EVAL_DIR / "fleurs_en" / "audio" / f"fleurs_en_{i:04d}.wav"
        sf.write(audio_path, sample["audio"]["array"], sample["audio"]["sampling_rate"])

        ref_idx = (i + 50) % len(fleurs_en)
        ref_sample = fleurs_en[ref_idx]
        ref_path = EVAL_DIR / "fleurs_en" / "audio" / f"fleurs_en_{ref_idx:04d}.wav"
        if not ref_path.exists():
            sf.write(ref_path, ref_sample["audio"]["array"], ref_sample["audio"]["sampling_rate"])

        regress_test.append({
            "id": f"regress_en_{i:04d}",
            "text": norm_en(sample["transcription"]),
            "text_raw": sample["transcription"],
            "ref_audio": str(ref_path.absolute()),
            "ref_text": ref_sample["transcription"],
            "gt_audio": str(audio_path.absolute()),
            "language_id": "en",
            "language_name": "English",
        })
    with open(EVAL_DIR / "test_lists" / "fleurs_en_regress.jsonl", "w") as f:
        for e in regress_test:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote fleurs_en_regress.jsonl: {len(regress_test)} samples")

    print("\nDone! Test lists in", EVAL_DIR / "test_lists")


if __name__ == "__main__":
    main()
