"""
Phase 1: Evaluate all 5 CSM-1B checkpoints on speaker_refs_manifest.json
Metrics: CER (round-trip ASR), MCD, ECAPA-TDNN speaker similarity
Goal: Select best checkpoint + best speaker

Phase 2: Best checkpoint + best speaker → CER on FLEURS Georgian test set
"""

import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio

# Fix SpeechBrain compatibility with torchaudio 2.10+
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def log(msg):
    print(msg, flush=True)


# ── Data loading ──────────────────────────────────────────────────────
def load_manifest(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


# ── Audio generation ──────────────────────────────────────────────────
def generate_for_checkpoint(ckpt_path, samples, output_dir, device="cuda"):
    from transformers import AutoProcessor, CsmForConditionalGeneration

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check resume
    all_exist = all(
        (output_dir / f"spk{s['speaker_id']}_{s['id']}.wav").exists()
        for s in samples
    )
    if all_exist:
        log(f"  All {len(samples)} samples already generated, skipping")
        return [
            {
                "speaker_id": str(s["speaker_id"]),
                "entry_id": s["id"],
                "text": s["text"],
                "gen_path": str(output_dir / f"spk{s['speaker_id']}_{s['id']}.wav"),
                "ref_path": str(Path("../../") / s["audio_path"]),
            }
            for s in samples
        ]

    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    model = CsmForConditionalGeneration.from_pretrained(ckpt_path, device_map=device)
    model.eval()

    # Split into batches, skip already-generated samples
    BATCH_SIZE = 8
    results = []
    pending = []
    for s in samples:
        sid = str(s["speaker_id"])
        out_path = output_dir / f"spk{sid}_{s['id']}.wav"
        info = {
            "speaker_id": sid, "entry_id": s["id"], "text": s["text"],
            "gen_path": str(out_path),
            "ref_path": str(Path("../../") / s["audio_path"]),
        }
        if out_path.exists():
            results.append(info)
        else:
            pending.append((s, info, out_path))

    if pending:
        log(f"  {len(results)} cached, {len(pending)} to generate (batch_size={BATCH_SIZE})")

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        texts = [f"[{s['speaker_id']}]{s['text']}" for s, _, _ in batch]
        inputs = processor(texts, add_special_tokens=True, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            audios = model.generate(**inputs, output_audio=True, max_new_tokens=125 * 10)

        for j, (s, info, out_path) in enumerate(batch):
            audio_np = audios[j].cpu().float().numpy()
            sf.write(str(out_path), audio_np, 24000)
            results.append(info)

        done = batch_start + len(batch)
        log(f"    [{done}/{len(pending)}] batch done, last dur={len(audios[-1])/24000:.1f}s")
        del inputs, audios
        free_vram()

    del model, processor
    free_vram()
    return results


# ── CER computation ──────────────────────────────────────────────────
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


def compute_cer_wer(gen_files, device="cuda"):
    """Round-trip CER/WER: generated audio → Omnilingual ASR → compare to original text.
    Uses /root/asr_env/bin/python3 subprocess (fairseq2 needs torch 2.8).
    """
    import subprocess, tempfile

    # Write audio paths to temp file for the subprocess
    audio_paths = [f["gen_path"] for f in gen_files]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(audio_paths, tmp)
        paths_file = tmp.name

    transcriptions_file = paths_file.replace(".json", "_results.json")

    # Shell out to ASR venv
    asr_script = f'''
import json, sys
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
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        log(f"  ASR FAILED: {result.stderr[-500:]}")
        raise RuntimeError("ASR subprocess failed")

    with open(transcriptions_file) as f:
        transcriptions = json.load(f)

    # Clean up temp files
    Path(paths_file).unlink(missing_ok=True)
    Path(transcriptions_file).unlink(missing_ok=True)

    results = []
    for f_info, hyp in zip(gen_files, transcriptions):
        cer = char_error_rate(f_info["text"], hyp)
        wer = word_error_rate(f_info["text"], hyp)
        results.append({
            "speaker_id": f_info["speaker_id"],
            "entry_id": f_info["entry_id"],
            "cer": cer,
            "wer": wer,
            "ref_text": f_info["text"],
            "hyp_text": hyp,
        })
        log(f"    spk={f_info['speaker_id']}: CER={cer:.3f} WER={wer:.3f}")

    return results


# ── MCD (Mel Cepstral Distortion) ────────────────────────────────────
def compute_mcd(gen_files):
    """Compute MCD between generated and reference audio using pymcd (proper MCCs + DTW)."""
    from pymcd.mcd import Calculate_MCD

    log("  Computing MCD (pymcd with DTW alignment)...")
    mcd_calc = Calculate_MCD(MCD_mode="dtw_sl")  # DTW alignment + silence removal

    results = []
    for i, f in enumerate(gen_files):
        try:
            mcd = mcd_calc.calculate_mcd(f["ref_path"], f["gen_path"])
        except Exception as e:
            log(f"    WARNING: MCD failed for {f['entry_id']}: {e}")
            mcd = float("nan")

        results.append({
            "speaker_id": f["speaker_id"],
            "entry_id": f["entry_id"],
            "mcd": float(mcd),
        })
        if (i + 1) % 20 == 0:
            log(f"    [{i+1}/{len(gen_files)}] last MCD={mcd:.2f} dB")

    return results


# ── ECAPA-TDNN speaker similarity ────────────────────────────────────
def compute_ecapa_similarity(gen_files, device="cuda"):
    """ECAPA-TDNN cosine similarity via SpeechBrain."""
    from speechbrain.inference.speaker import EncoderClassifier

    log("  Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    results = []
    for i, f in enumerate(gen_files):
        gen_wav, _ = librosa.load(f["gen_path"], sr=16000)
        ref_wav, _ = librosa.load(f["ref_path"], sr=16000)

        gen_tensor = torch.tensor(gen_wav).unsqueeze(0).to(device)
        ref_tensor = torch.tensor(ref_wav).unsqueeze(0).to(device)

        with torch.no_grad():
            gen_emb = classifier.encode_batch(gen_tensor)
            ref_emb = classifier.encode_batch(ref_tensor)

        sim = torch.nn.functional.cosine_similarity(
            gen_emb.squeeze(), ref_emb.squeeze(), dim=0
        ).item()

        results.append({
            "speaker_id": f["speaker_id"],
            "entry_id": f["entry_id"],
            "similarity": sim,
        })
        if (i + 1) % 20 == 0:
            log(f"    [{i+1}/{len(gen_files)}] last sim={sim:.4f}")

    del classifier
    free_vram()
    return results


# ── Aggregation ───────────────────────────────────────────────────────
def aggregate_by_speaker(results, metric_key):
    per_spk = {}
    for r in results:
        per_spk.setdefault(r["speaker_id"], []).append(r[metric_key])
    return {sid: {"mean": np.mean(vals), "std": np.std(vals), "n": len(vals)}
            for sid, vals in per_spk.items()}


def aggregate_overall(results, metric_key):
    vals = [r[metric_key] for r in results]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    manifest = "../../data/clean/speaker_refs_manifest.json"
    checkpoints = [
        "checkpoints/checkpoint-1975",
        "checkpoints/checkpoint-2054",
        "checkpoints/checkpoint-2133",
        "checkpoints/checkpoint-2212",
        "checkpoints/checkpoint-2291",
    ]
    device = "cuda"

    samples = load_manifest(manifest)
    speakers = sorted(set(s["speaker_id"] for s in samples))
    log(f"Loaded {len(samples)} samples, {len(speakers)} speakers: {speakers}")

    all_results = {}

    for ckpt in checkpoints:
        ckpt_name = Path(ckpt).name
        log(f"\n{'='*60}")
        log(f"CHECKPOINT: {ckpt_name}")
        log(f"{'='*60}")

        out_dir = f"eval_checkpoints/{ckpt_name}"

        # Step 1: Generate audio
        log(f"\n[1/4] Generating audio...")
        t0 = time.time()
        gen_files = generate_for_checkpoint(ckpt, samples, out_dir, device)
        log(f"  Generation took {time.time()-t0:.0f}s")

        # Step 2: CER/WER
        log(f"\n[2/4] Computing CER/WER (Omnilingual ASR)...")
        t0 = time.time()
        cer_results = compute_cer_wer(gen_files, device)
        log(f"  CER/WER took {time.time()-t0:.0f}s")

        # Step 3: MCD
        log(f"\n[3/4] Computing MCD...")
        t0 = time.time()
        mcd_results = compute_mcd(gen_files)
        log(f"  MCD took {time.time()-t0:.0f}s")

        # Step 4: ECAPA-TDNN
        log(f"\n[4/4] Computing ECAPA-TDNN speaker similarity...")
        t0 = time.time()
        ecapa_results = compute_ecapa_similarity(gen_files, device)
        log(f"  ECAPA took {time.time()-t0:.0f}s")

        all_results[ckpt_name] = {
            "cer": {"overall": aggregate_overall(cer_results, "cer"),
                    "per_speaker": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                                    for k, v in aggregate_by_speaker(cer_results, "cer").items()},
                    "per_sample": cer_results},
            "wer": {"overall": aggregate_overall(cer_results, "wer"),
                    "per_speaker": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                                    for k, v in aggregate_by_speaker(cer_results, "wer").items()}},
            "mcd": {"overall": aggregate_overall(mcd_results, "mcd"),
                    "per_speaker": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                                    for k, v in aggregate_by_speaker(mcd_results, "mcd").items()}},
            "ecapa": {"overall": aggregate_overall(ecapa_results, "similarity"),
                      "per_speaker": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                                      for k, v in aggregate_by_speaker(ecapa_results, "similarity").items()}},
        }

        # Print checkpoint summary
        log(f"\n  Summary for {ckpt_name}:")
        log(f"    CER:   {all_results[ckpt_name]['cer']['overall']['mean']:.4f}")
        log(f"    WER:   {all_results[ckpt_name]['wer']['overall']['mean']:.4f}")
        log(f"    MCD:   {all_results[ckpt_name]['mcd']['overall']['mean']:.2f}")
        log(f"    ECAPA: {all_results[ckpt_name]['ecapa']['overall']['mean']:.4f}")

    # ── Select best checkpoint ────────────────────────────────────────
    log(f"\n{'='*60}")
    log("CHECKPOINT COMPARISON")
    log(f"{'='*60}")
    log(f"{'Checkpoint':<22} {'CER':>8} {'WER':>8} {'MCD':>8} {'ECAPA':>8}")
    log("-" * 60)

    best_ckpt = None
    best_cer = float("inf")
    for ckpt_name, res in all_results.items():
        cer = res["cer"]["overall"]["mean"]
        wer = res["wer"]["overall"]["mean"]
        mcd = res["mcd"]["overall"]["mean"]
        ecapa = res["ecapa"]["overall"]["mean"]
        log(f"{ckpt_name:<22} {cer:>8.4f} {wer:>8.4f} {mcd:>8.2f} {ecapa:>8.4f}")
        if cer < best_cer:
            best_cer = cer
            best_ckpt = ckpt_name

    log(f"\nBest checkpoint (lowest CER): {best_ckpt} (CER={best_cer:.4f})")

    # ── Select best speaker from best checkpoint ──────────────────────
    best_res = all_results[best_ckpt]
    log(f"\nPer-speaker breakdown for {best_ckpt}:")
    log(f"{'Speaker':>8} {'CER':>8} {'WER':>8} {'MCD':>8} {'ECAPA':>8}")
    log("-" * 48)

    best_spk = None
    best_spk_cer = float("inf")
    for sid in sorted(best_res["cer"]["per_speaker"].keys(), key=lambda x: int(x)):
        cer = best_res["cer"]["per_speaker"][sid]["mean"]
        wer = best_res["wer"]["per_speaker"][sid]["mean"]
        mcd = best_res["mcd"]["per_speaker"][sid]["mean"]
        ecapa = best_res["ecapa"]["per_speaker"][sid]["mean"]
        log(f"{sid:>8} {cer:>8.4f} {wer:>8.4f} {mcd:>8.2f} {ecapa:>8.4f}")
        if cer < best_spk_cer:
            best_spk_cer = cer
            best_spk = sid

    log(f"\nBest speaker (lowest CER): speaker {best_spk} (CER={best_spk_cer:.4f})")

    # Save results
    out_path = "eval_checkpoints/results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def to_json(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(v) for v in obj]
        return obj

    save_data = {
        "all_results": to_json(all_results),
        "best_checkpoint": best_ckpt,
        "best_speaker": best_spk,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {out_path}")
    log(f"\n>>> NEXT: Run Phase 2 — evaluate best checkpoint ({best_ckpt}) + "
        f"best speaker ({best_spk}) on FLEURS Georgian test set")


if __name__ == "__main__":
    main()
