"""
Compute CER (via Meta Omnilingual ASR subprocess) and MCD-DTW on generated audio.

ASR runs via /root/asr_env/bin/python3 subprocess (torch 2.8 + fairseq2).
MCD-DTW runs in-process via pymcd (proper mel cepstral coefficients).

Reads generated audio from eval_speaker_results/ (produced by eval_speakers.py).

Usage:
    python eval_cer_mcd.py
"""

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymcd.mcd import Calculate_MCD

EVAL_DIR = Path("eval_speaker_results")
RESULTS_FILE = EVAL_DIR / "results.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKER_REFS = REPO_ROOT / "data/clean/speaker_refs_manifest.json"
ASR_PYTHON = "/root/asr_env/bin/python3"

CHECKPOINTS = ["model_100000", "model_110000"]

TEST_SENTENCES = [
    "საქართველო მდებარეობს კავკასიის რეგიონში, ევროპისა და აზიის გასაყარზე.",
    "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია.",
    "ქართული ენა მსოფლიოში ერთ-ერთი უძველესი ენაა.",
    "კახეთი ცნობილია თავისი ღვინით და ულამაზესი ბუნებით.",
    "საქართველოს მდიდარი ისტორია და კულტურული მემკვიდრეობა აქვს.",
]


# ── ASR via subprocess ───────────────────────────────────────

def batch_transcribe(audio_paths: list[str], batch_size: int = 8) -> list[str]:
    """Transcribe audio files via Meta Omnilingual ASR in /root/asr_env subprocess, in batches."""
    all_transcriptions = []

    for batch_start in range(0, len(audio_paths), batch_size):
        batch = audio_paths[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(audio_paths) + batch_size - 1) // batch_size

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch, f)
            paths_file = f.name

        out_file = paths_file.replace(".json", "_out.json")

        asr_script = f'''
import json, sys
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

with open("{paths_file}") as f:
    audio_paths = json.load(f)

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
transcriptions = pipeline.transcribe(audio_paths, lang=["kat_Geor"] * len(audio_paths), batch_size=4)

with open("{out_file}", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False)
'''

        print(f"  ASR batch {batch_num}/{total_batches} ({len(batch)} files)...", flush=True)
        result = subprocess.run(
            [ASR_PYTHON, "-u", "-c", asr_script],
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            print(f"  ASR STDERR: {result.stderr[-300:]}", flush=True)
            # Fill with empty strings on failure
            all_transcriptions.extend([""] * len(batch))
            Path(paths_file).unlink(missing_ok=True)
            Path(out_file).unlink(missing_ok=True)
            continue

        with open(out_file) as f:
            transcriptions = json.load(f)

        all_transcriptions.extend(transcriptions)
        print(f"  ASR batch {batch_num}/{total_batches} done. Sample: [{transcriptions[0][:50]}...]", flush=True)

        Path(paths_file).unlink(missing_ok=True)
        Path(out_file).unlink(missing_ok=True)

    return all_transcriptions


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


# ── MCD with DTW ─────────────────────────────────────────────

_mcd_calculator = Calculate_MCD(MCD_mode="dtw")


def compute_mcd_dtw(ref_path, gen_path):
    """Mel Cepstral Distortion with DTW alignment via pymcd."""
    return _mcd_calculator.calculate_mcd(str(ref_path), str(gen_path))


# ── Speaker ref lookup ───────────────────────────────────────

def load_speaker_refs():
    with open(SPEAKER_REFS) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    by_speaker = defaultdict(list)
    for e in entries:
        by_speaker[e["speaker_id"]].append(e)
    best_refs = {}
    for sid, refs in by_speaker.items():
        best = max(refs, key=lambda x: x.get("nisqa_mos", 0))
        audio_path = Path(best["audio_path"])
        if not audio_path.is_absolute():
            audio_path = REPO_ROOT / audio_path
        best_refs[sid] = str(audio_path)
    return best_refs


# ── Main ─────────────────────────────────────────────────────

def main():
    speaker_refs = load_speaker_refs()
    speaker_ids = sorted(speaker_refs.keys())
    print(f"Speakers: {speaker_ids}")

    # Load existing similarity results
    sim_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for r in json.load(f):
                key = (r["checkpoint"], r["speaker_id"])
                sim_results[key] = r["avg_similarity"]

    # Collect all audio paths for batch ASR
    all_audio = []  # (ckpt, sid, sent_idx, path)
    for ckpt in CHECKPOINTS:
        for sid in speaker_ids:
            for i in range(len(TEST_SENTENCES)):
                gen_path = EVAL_DIR / ckpt / f"spk{sid}_sent{i}.wav"
                if gen_path.exists():
                    all_audio.append((ckpt, sid, i, str(gen_path)))

    # Batch transcribe all at once
    print(f"\nTranscribing {len(all_audio)} audio files via ASR...")
    audio_paths = [a[3] for a in all_audio]
    transcriptions = batch_transcribe(audio_paths)

    # Build transcription lookup
    trans_map = {}
    for (ckpt, sid, i, path), hyp in zip(all_audio, transcriptions):
        trans_map[(ckpt, sid, i)] = hyp

    # Compute CER + MCD per checkpoint × speaker
    all_results = []

    for ckpt in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt}")

        for sid in speaker_ids:
            cers = []
            mcds = []
            ref_audio = speaker_refs[sid]
            print(f"\n  Speaker {sid}:", flush=True)

            for i, sentence in enumerate(TEST_SENTENCES):
                gen_path = EVAL_DIR / ckpt / f"spk{sid}_sent{i}.wav"
                if not gen_path.exists():
                    print(f"    Sent {i}: MISSING")
                    continue

                # CER
                hyp = trans_map.get((ckpt, sid, i), "")
                cer = compute_cer(sentence, hyp)
                cers.append(cer)

                # MCD-DTW
                mcd = compute_mcd_dtw(ref_audio, str(gen_path))
                mcds.append(mcd)

                print(f"    Sent {i}: CER={cer:.3f}  MCD={mcd:.2f}  hyp=[{hyp[:60]}]")

            avg_cer = float(np.mean(cers)) if cers else -1
            avg_mcd = float(np.mean(mcds)) if mcds else -1
            sim = sim_results.get((f"{ckpt}.pt", sid), -1)

            all_results.append({
                "checkpoint": ckpt,
                "speaker_id": sid,
                "avg_cer": avg_cer,
                "avg_mcd": avg_mcd,
                "avg_similarity": sim,
                "per_sentence_cer": [float(c) for c in cers],
                "per_sentence_mcd": [float(m) for m in mcds],
            })
            print(f"    AVG: CER={avg_cer:.3f}  MCD={avg_mcd:.2f}  sim={sim:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("FULL SUMMARY (sorted by CER ascending)")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<18} {'Speaker':<10} {'CER':<8} {'MCD':<8} {'Sim':<8}")
    print("-" * 52)
    for r in sorted(all_results, key=lambda x: x["avg_cer"]):
        print(f"{r['checkpoint']:<18} {r['speaker_id']:<10} {r['avg_cer']:.3f}    {r['avg_mcd']:.2f}    {r['avg_similarity']:.4f}")

    # Best per checkpoint
    print(f"\n{'='*60}")
    print("BEST SPEAKER PER CHECKPOINT (by CER)")
    print(f"{'='*60}")
    for ckpt in CHECKPOINTS:
        ckpt_results = [r for r in all_results if r["checkpoint"] == ckpt]
        best = min(ckpt_results, key=lambda x: x["avg_cer"])
        print(f"  {ckpt}: Speaker {best['speaker_id']}  CER={best['avg_cer']:.3f}  MCD={best['avg_mcd']:.2f}  sim={best['avg_similarity']:.4f}")

    # Save
    out_file = EVAL_DIR / "full_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
