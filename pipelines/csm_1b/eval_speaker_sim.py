"""
Speaker similarity evaluation for CSM-1B checkpoints.
Generates 1 sample per speaker (12 total) per checkpoint,
then compares to reference audio using ECAPA-TDNN and WavLM embeddings.

Memory-safe: unloads each model before loading the next.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_gpu_free_mb():
    return torch.cuda.mem_get_info()[0] / 1024**2


def log(msg):
    print(msg, flush=True)


# ── Step 1: Pick 1 sample per speaker ──────────────────────────────────
def load_all_samples(manifest_path):
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Step 2: Generate audio from a checkpoint ───────────────────────────
def generate_for_checkpoint(checkpoint_path, samples, output_dir, device="cuda"):
    """Load model, generate audio for all samples, unload model."""
    from transformers import AutoProcessor, CsmForConditionalGeneration

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all samples already generated (resume support)
    generated_files = []
    all_exist = True
    for entry in samples:
        sid = str(entry["speaker_id"])
        out_path = output_dir / f"spk{sid}_{entry['id']}.wav"
        if not out_path.exists():
            all_exist = False
            break
        generated_files.append({
            "speaker_id": sid,
            "entry_id": entry["id"],
            "gen_path": str(out_path),
            "ref_path": str(Path("../../") / entry["audio_path"]),
            "text": entry["text"],
        })

    if all_exist:
        log(f"  All {len(samples)} samples already generated, skipping inference")
        return generated_files

    log(f"  GPU free before model load: {get_gpu_free_mb():.0f} MB")

    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    model = CsmForConditionalGeneration.from_pretrained(checkpoint_path, device_map=device)
    model.eval()
    log(f"  GPU free after model load: {get_gpu_free_mb():.0f} MB")

    generated_files = []
    for i, entry in enumerate(samples):
        sid = str(entry["speaker_id"])
        text = entry["text"]
        out_path = output_dir / f"spk{sid}_{entry['id']}.wav"

        inputs = processor(f"[{sid}]{text}", add_special_tokens=True).to(device)

        with torch.no_grad():
            audio = model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=125 * 5,
            )

        audio_np = audio[0].cpu().float().numpy()
        sf.write(str(out_path), audio_np, 24000)
        generated_files.append({
            "speaker_id": sid,
            "entry_id": entry["id"],
            "gen_path": str(out_path),
            "ref_path": str(Path("../../") / entry["audio_path"]),
            "text": text,
            "duration": len(audio_np) / 24000,
        })
        log(f"    [{i+1}/{len(samples)}] spk={sid} -> {len(audio_np)/24000:.1f}s")

        # Free KV cache between samples
        del inputs, audio
        free_vram()

    # Unload model completely
    del model, processor
    free_vram()
    log(f"  GPU free after model unload: {get_gpu_free_mb():.0f} MB")

    return generated_files


# ── Step 3: ECAPA-TDNN speaker similarity ──────────────────────────────
def ecapa_similarity(generated_files, device="cuda"):
    """Compute cosine similarity using ECAPA2 via HuggingFace (no SpeechBrain dependency)."""
    log("  Loading Wav2Vec2 speaker embedding model (microsoft/wavlm-base-sv)...")
    from transformers import AutoFeatureExtractor, AutoModel

    # Use UniSpeech-SAT as ECAPA alternative (HF-native, no speechbrain)
    model_name = "microsoft/unispeech-sat-base-plus-sv"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    log(f"  GPU free after model load: {get_gpu_free_mb():.0f} MB")

    def get_embedding(audio_path):
        wav, _ = librosa.load(audio_path, sr=16000)
        inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pool over time dimension
        emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        return emb

    results = []
    for i, info in enumerate(generated_files):
        gen_emb = get_embedding(info["gen_path"])
        ref_emb = get_embedding(info["ref_path"])
        sim = torch.nn.functional.cosine_similarity(
            gen_emb, ref_emb, dim=0
        ).item()
        results.append({"speaker_id": info["speaker_id"], "entry_id": info["entry_id"], "similarity": sim})
        log(f"    [{i+1}/{len(generated_files)}] spk={info['speaker_id']}: {sim:.4f}")

    del model, feature_extractor
    free_vram()
    return results


# ── Step 4: WavLM speaker similarity ──────────────────────────────────
def wavlm_similarity(generated_files, device="cuda"):
    """Compute cosine similarity using WavLM-base-plus-sv."""
    log("  Loading WavLM speaker verification model...")
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    )
    model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    ).to(device).eval()
    log(f"  GPU free after WavLM load: {get_gpu_free_mb():.0f} MB")

    results = []
    for i, info in enumerate(generated_files):
        # Load and resample to 16kHz using librosa
        gen_wav, _ = librosa.load(info["gen_path"], sr=16000)
        ref_wav, _ = librosa.load(info["ref_path"], sr=16000)

        with torch.no_grad():
            gen_inputs = feature_extractor(
                gen_wav, sampling_rate=16000, return_tensors="pt"
            ).to(device)
            gen_emb = model(**gen_inputs).embeddings

            ref_inputs = feature_extractor(
                ref_wav, sampling_rate=16000, return_tensors="pt"
            ).to(device)
            ref_emb = model(**ref_inputs).embeddings

        sim = torch.nn.functional.cosine_similarity(
            gen_emb.squeeze(), ref_emb.squeeze(), dim=0
        ).item()
        results.append({"speaker_id": info["speaker_id"], "entry_id": info["entry_id"], "similarity": sim})
        log(f"    [{i+1}/{len(generated_files)}] spk={info['speaker_id']}: {sim:.4f}")

        del gen_inputs, ref_inputs, gen_emb, ref_emb
        free_vram()

    del model, feature_extractor
    free_vram()
    return results


# ── Main ───────────────────────────────────────────────────────────────
def main():
    manifest = "../../data/clean/speaker_refs_manifest.json"
    checkpoints = [
        "checkpoints/checkpoint-1027",
        "checkpoints/checkpoint-1106",
    ]

    samples = load_all_samples(manifest)
    speakers = sorted(set(s["speaker_id"] for s in samples))
    log(f"Loaded {len(samples)} samples across {len(speakers)} speakers: {speakers}")

    all_results = {}

    for ckpt in checkpoints:
        ckpt_name = Path(ckpt).name
        log(f"\n{'='*60}")
        log(f"Checkpoint: {ckpt_name}")
        log(f"{'='*60}")

        # Generate audio
        out_dir = f"eval_speaker_sim/{ckpt_name}"
        log(f"\n[1/3] Generating audio...")
        gen_files = generate_for_checkpoint(ckpt, samples, out_dir)

        # ECAPA-TDNN
        log(f"\n[2/3] ECAPA-TDNN similarity...")
        ecapa = ecapa_similarity(gen_files)

        # WavLM
        log(f"\n[3/3] WavLM similarity...")
        wavlm = wavlm_similarity(gen_files)

        # Aggregate per-speaker and overall
        def aggregate(results_list):
            all_sims = [r["similarity"] for r in results_list]
            per_spk = {}
            for r in results_list:
                per_spk.setdefault(r["speaker_id"], []).append(r["similarity"])
            per_spk_mean = {sid: np.mean(sims) for sid, sims in per_spk.items()}
            return {
                "per_sample": results_list,
                "per_speaker_mean": per_spk_mean,
                "overall_mean": np.mean(all_sims),
                "overall_std": np.std(all_sims),
                "overall_min": np.min(all_sims),
                "overall_max": np.max(all_sims),
            }

        all_results[ckpt_name] = {
            "ecapa_tdnn": aggregate(ecapa),
            "wavlm_sv": aggregate(wavlm),
            "num_samples": len(gen_files),
        }

    # Save results
    out_path = "eval_speaker_sim/results.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2, ensure_ascii=False)

    # Print summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    for ckpt_name, res in all_results.items():
        log(f"\n{ckpt_name} ({res['num_samples']} samples):")
        e = res["ecapa_tdnn"]
        w = res["wavlm_sv"]
        log(f"  ECAPA-TDNN:  mean={e['overall_mean']:.4f} ± {e['overall_std']:.4f}  (min={e['overall_min']:.4f}, max={e['overall_max']:.4f})")
        log(f"  WavLM-SV:    mean={w['overall_mean']:.4f} ± {w['overall_std']:.4f}  (min={w['overall_min']:.4f}, max={w['overall_max']:.4f})")
        log(f"  Per-speaker ECAPA means: {', '.join(f'spk{k}={v:.3f}' for k, v in sorted(e['per_speaker_mean'].items()))}")
        log(f"  Per-speaker WavLM means: {', '.join(f'spk{k}={v:.3f}' for k, v in sorted(w['per_speaker_mean'].items()))}")

    log(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
