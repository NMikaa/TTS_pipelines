"""
Phase 2: Evaluate best CSM-1B checkpoint on FLEURS Georgian test set.
Checkpoint: checkpoint-2212, Speaker: 7
Metric: CER (round-trip via Meta Omnilingual ASR 7B)
"""

import gc
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

CHECKPOINT = "checkpoints/checkpoint-2212"
SPEAKER_ID = 7
BATCH_SIZE = 8
DEVICE = "cuda"
OUTPUT_DIR = Path("eval_fleurs")
RESULTS_FILE = OUTPUT_DIR / "results.json"


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def log(msg):
    print(msg, flush=True)


def normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def char_error_rate(ref, hyp):
    ref = list(normalize_text(ref))
    hyp = list(normalize_text(hyp))
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / len(ref)


def word_error_rate(ref, hyp):
    ref = normalize_text(ref).split()
    hyp = normalize_text(hyp).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / len(ref)


def load_fleurs():
    """Load FLEURS Georgian test set."""
    from datasets import load_dataset

    log("Loading FLEURS Georgian test set...")
    ds = load_dataset("google/fleurs", "ka_ge", split="test", trust_remote_code=True)
    log(f"  Loaded {len(ds)} samples")

    samples = []
    audio_dir = OUTPUT_DIR / "fleurs_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(ds):
        # Save audio for reference (FLEURS provides audio arrays)
        audio_path = audio_dir / f"fleurs_{i:04d}.wav"
        if not audio_path.exists():
            audio_array = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            sf.write(str(audio_path), audio_array, sr)

        samples.append({
            "id": f"fleurs_{i:04d}",
            "text": item["transcription"],
            "ref_audio": str(audio_path),
        })

    log(f"  Prepared {len(samples)} FLEURS samples")
    return samples


def generate_audio(samples):
    """Generate audio for all FLEURS samples using checkpoint-2212, speaker 7."""
    from transformers import AutoProcessor, CsmForConditionalGeneration

    gen_dir = OUTPUT_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Check resume — find which samples still need generation
    pending = []
    completed = []
    for s in samples:
        out_path = gen_dir / f"{s['id']}.wav"
        if out_path.exists():
            completed.append({**s, "gen_path": str(out_path)})
        else:
            pending.append((s, out_path))

    if not pending:
        log(f"  All {len(samples)} samples already generated, skipping")
        return completed

    log(f"  {len(completed)} cached, {len(pending)} to generate")

    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    model = CsmForConditionalGeneration.from_pretrained(CHECKPOINT, device_map=DEVICE)
    model.eval()

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        texts = [f"[{SPEAKER_ID}]{s['text']}" for s, _ in batch]
        inputs = processor(texts, add_special_tokens=True, padding=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audios = model.generate(**inputs, output_audio=True, max_new_tokens=125 * 15)

        for j, (s, out_path) in enumerate(batch):
            audio_np = audios[j].cpu().float().numpy()
            sf.write(str(out_path), audio_np, 24000)
            completed.append({**s, "gen_path": str(out_path)})

        done = batch_start + len(batch)
        log(f"    [{done}/{len(pending)}] batch done, last dur={len(audios[-1])/24000:.1f}s")
        del inputs, audios
        free_vram()

    del model, processor
    free_vram()
    return completed


def compute_cer_wer(gen_files):
    """Round-trip CER/WER via Omnilingual ASR subprocess."""
    audio_paths = [f["gen_path"] for f in gen_files]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(audio_paths, tmp)
        paths_file = tmp.name

    transcriptions_file = paths_file.replace(".json", "_results.json")

    asr_script = f'''
import json
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

with open("{paths_file}") as f:
    audio_paths = json.load(f)

print(f"Loading ASR model...", flush=True)
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
print(f"Transcribing {{len(audio_paths)}} files...", flush=True)
transcriptions = pipeline.transcribe(audio_paths, lang=["kat_Geor"] * len(audio_paths), batch_size=4)
print(f"Done transcribing.", flush=True)

with open("{transcriptions_file}", "w") as f:
    json.dump(transcriptions, f, ensure_ascii=False)
'''

    log("  Running ASR in /root/asr_env (torch 2.8 + fairseq2 0.6)...")
    result = subprocess.run(
        ["/root/asr_env/bin/python3", "-u", "-c", asr_script],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        log(f"  ASR FAILED: {result.stderr[-500:]}")
        raise RuntimeError("ASR subprocess failed")

    with open(transcriptions_file) as f:
        transcriptions = json.load(f)

    Path(paths_file).unlink(missing_ok=True)
    Path(transcriptions_file).unlink(missing_ok=True)

    results = []
    for f_info, hyp in zip(gen_files, transcriptions):
        cer = char_error_rate(f_info["text"], hyp)
        wer = word_error_rate(f_info["text"], hyp)
        results.append({
            "id": f_info["id"],
            "cer": cer,
            "wer": wer,
            "ref_text": f_info["text"],
            "hyp_text": hyp,
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load FLEURS
    log(f"\n{'='*60}")
    log("FLEURS Georgian Evaluation — CSM-1B checkpoint-2212 speaker 7")
    log(f"{'='*60}")

    samples = load_fleurs()

    # Step 2: Generate audio
    log(f"\n[1/2] Generating audio with checkpoint-2212, speaker {SPEAKER_ID}...")
    t0 = time.time()
    gen_files = generate_audio(samples)
    log(f"  Generation took {time.time()-t0:.0f}s")

    # Step 3: CER/WER
    log(f"\n[2/2] Computing CER/WER (round-trip ASR)...")
    t0 = time.time()
    cer_results = compute_cer_wer(gen_files)
    log(f"  CER/WER took {time.time()-t0:.0f}s")

    # Aggregate
    cers = [r["cer"] for r in cer_results]
    wers = [r["wer"] for r in cer_results]

    log(f"\n{'='*60}")
    log("FLEURS RESULTS")
    log(f"{'='*60}")
    log(f"  Samples:    {len(cers)}")
    log(f"  CER mean:   {np.mean(cers):.4f}")
    log(f"  CER median: {np.median(cers):.4f}")
    log(f"  CER std:    {np.std(cers):.4f}")
    log(f"  CER p90:    {np.percentile(cers, 90):.4f}")
    log(f"  WER mean:   {np.mean(wers):.4f}")
    log(f"  WER median: {np.median(wers):.4f}")

    # Distribution
    brackets = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.01)]
    log(f"\n  CER distribution:")
    for lo, hi in brackets:
        count = sum(1 for c in cers if lo <= c < hi)
        pct = count / len(cers) * 100
        label = f"  [{lo:.2f}, {hi:.2f})" if hi < 1.01 else f"  [{lo:.2f}, 1.00]"
        log(f"    {label}: {count:>4} ({pct:>5.1f}%)")

    # Save
    save_data = {
        "checkpoint": CHECKPOINT,
        "speaker_id": SPEAKER_ID,
        "dataset": "FLEURS Georgian (ka_ge) test",
        "num_samples": len(cers),
        "cer": {
            "mean": float(np.mean(cers)),
            "median": float(np.median(cers)),
            "std": float(np.std(cers)),
            "p90": float(np.percentile(cers, 90)),
            "min": float(np.min(cers)),
            "max": float(np.max(cers)),
        },
        "wer": {
            "mean": float(np.mean(wers)),
            "median": float(np.median(wers)),
            "std": float(np.std(wers)),
        },
        "per_sample": cer_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
