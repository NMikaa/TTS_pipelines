"""OmniVoice Georgian evaluation pipeline.

Generates audio from test lists using each checkpoint, then scores with ASR CER.

Usage:
    # Stage 1: Generate audio (venv_omnivoice, needs GPU)
    python evaluate.py --stage generate --checkpoint k2-fsa/OmniVoice --gpu 0

    # Stage 2: ASR transcribe (run via venv_meta_ctc)
    python evaluate.py --stage transcribe --gpu 0

    # Stage 3: Compute metrics
    python evaluate.py --stage metrics

    # Or run all for one checkpoint:
    python evaluate.py --stage generate --checkpoint 099/checkpoint-1000 --gpu 0
"""

import argparse
import json
import os
import sys
import time
import re
import subprocess
from pathlib import Path

# Monkeypatch torchaudio.load to use soundfile (torch 2.11 removed all backends except torchcodec)
import torchaudio
import soundfile as sf
import torch
import numpy as np

_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    try:
        return _original_torchaudio_load(filepath, *args, **kwargs)
    except (ImportError, RuntimeError, OSError):
        data, sr = sf.read(str(filepath), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        return waveform, sr

torchaudio.load = _patched_torchaudio_load

_original_torchaudio_save = torchaudio.save

def _patched_torchaudio_save(filepath, waveform, sample_rate, *args, **kwargs):
    try:
        return _original_torchaudio_save(filepath, waveform, sample_rate, *args, **kwargs)
    except (ImportError, RuntimeError, OSError):
        data = waveform.cpu().numpy()
        if data.ndim == 2:
            data = data.T  # (channels, samples) -> (samples, channels)
        sf.write(str(filepath), data, sample_rate)

torchaudio.save = _patched_torchaudio_save

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR / "eval_data"
RESULTS_DIR = SCRIPT_DIR / "eval_results"
EXP_DIR = SCRIPT_DIR / "exp"
OMNIVOICE_DIR = SCRIPT_DIR.parent.parent / "OmniVoice"

TEST_LISTS = {
    "cv_geo": EVAL_DIR / "test_lists" / "cv_geo_voice_clone.jsonl",
    "fleurs_ka": EVAL_DIR / "test_lists" / "fleurs_ka.jsonl",
    "fleurs_en_cross": EVAL_DIR / "test_lists" / "fleurs_en_cross.jsonl",
    "fleurs_en_regress": EVAL_DIR / "test_lists" / "fleurs_en_regress.jsonl",
}

CHECKPOINTS = {
    "pretrained": "k2-fsa/OmniVoice",
    "099_ckpt1000": str(EXP_DIR / "099" / "checkpoint-1000"),
    "097_ckpt556": str(EXP_DIR / "097" / "checkpoint-556"),
    "097_ckpt1112": str(EXP_DIR / "097" / "checkpoint-1112"),
    "099v2_ckpt240": str(EXP_DIR / "099_v2" / "checkpoint-240"),
    "099v2_ckpt480": str(EXP_DIR / "099_v2" / "checkpoint-480"),
    "099v2_ckpt720": str(EXP_DIR / "099_v2" / "checkpoint-720"),
    "099v2_ckpt960": str(EXP_DIR / "099_v2" / "checkpoint-960"),
    "099v2_ckpt1200": str(EXP_DIR / "099_v2" / "checkpoint-1200"),
    "099v2_ckpt1440": str(EXP_DIR / "099_v2" / "checkpoint-1440"),
    "099v2_ckpt1500": str(EXP_DIR / "099_v2" / "checkpoint-1500"),
}

SAMPLING_RATE = 24000


def generate_for_checkpoint(ckpt_tag, ckpt_path, test_lists, device="cuda:0"):
    """Generate audio for all test lists using one checkpoint."""
    sys.path.insert(0, str(OMNIVOICE_DIR))
    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
    import torchaudio

    print(f"\n{'='*60}")
    print(f"Generating: {ckpt_tag} ({ckpt_path})")
    print(f"{'='*60}")

    print("Loading model...")
    model = OmniVoice.from_pretrained(
        ckpt_path, device_map=device, dtype=torch.float16, load_asr=False,
    )

    config = OmniVoiceGenerationConfig(
        num_step=32, guidance_scale=2.0, denoise=True,
        preprocess_prompt=True, postprocess_output=True,
    )

    for test_name, test_list_path in test_lists.items():
        out_dir = RESULTS_DIR / ckpt_tag / test_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done
        manifest_path = out_dir / "generated.jsonl"
        if manifest_path.exists():
            existing = sum(1 for _ in open(manifest_path))
            expected = sum(1 for _ in open(test_list_path))
            if existing >= expected:
                print(f"  {test_name}: already done ({existing} samples)")
                continue

        print(f"\n  {test_name}: generating...")
        samples = []
        with open(test_list_path) as f:
            for line in f:
                samples.append(json.loads(line))

        results = []
        prompt_cache = {}  # Cache voice clone prompts by ref_audio

        for i, sample in enumerate(samples):
            try:
                ref_audio = sample["ref_audio"]
                ref_text = sample.get("ref_text")
                # text_input: what we feed to the model (normalized, no digits)
                # text: what we compare against for CER (benchmark reference, raw)
                text_input = sample.get("text_input", sample["text"])
                text_ref = sample["text"]
                lang = sample.get("language_name", "Georgian")

                # Cache voice clone prompts
                if ref_audio not in prompt_cache:
                    prompt_cache[ref_audio] = model.create_voice_clone_prompt(
                        ref_audio=ref_audio, ref_text=ref_text,
                    )

                audio = model.generate(
                    text=text_input, language=lang,
                    voice_clone_prompt=prompt_cache[ref_audio],
                    generation_config=config,
                )

                wav = audio[0].squeeze(0).cpu()
                wav_path = out_dir / f"{sample['id']}.wav"
                torchaudio.save(str(wav_path), wav.unsqueeze(0), SAMPLING_RATE)

                results.append({
                    "id": sample["id"],
                    "text": text_ref,
                    "generated_audio": str(wav_path),
                    "ref_audio": ref_audio,
                    "gt_audio": sample.get("gt_audio", ""),
                    "language_id": sample.get("language_id", "ka"),
                    "duration": wav.shape[0] / SAMPLING_RATE,
                })

            except Exception as e:
                print(f"    Failed {sample['id']}: {e}")
                results.append({
                    "id": sample["id"], "text": sample.get("text", ""),
                    "generated_audio": "", "error": str(e),
                    "language_id": sample.get("language_id", "ka"),
                })

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(samples)}")

        with open(manifest_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    Done: {len(results)} samples → {manifest_path}")

    del model
    import torch
    torch.cuda.empty_cache()


def transcribe_all(device="cuda:0"):
    """Transcribe all generated audio using omniASR-CTC-1B via subprocess."""
    print(f"\n{'='*60}")
    print("ASR Transcription (omniASR-CTC-1B)")
    print(f"{'='*60}")

    # Set CTC_PYTHON env var to point at your venv with omnilingual_asr installed
    CTC_PYTHON = os.environ.get("CTC_PYTHON", "venv_meta_ctc/bin/python")
    gpu_idx = device.split(":")[-1] if ":" in device else "0"

    for ckpt_dir in sorted(RESULTS_DIR.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        for test_dir in sorted(ckpt_dir.iterdir()):
            if not test_dir.is_dir():
                continue

            manifest_path = test_dir / "generated.jsonl"
            transcript_path = test_dir / "transcripts.jsonl"

            if not manifest_path.exists():
                continue
            if transcript_path.exists():
                print(f"  {ckpt_dir.name}/{test_dir.name}: already transcribed")
                continue

            # Collect audio paths
            entries = []
            with open(manifest_path) as f:
                for line in f:
                    e = json.loads(line)
                    if e.get("generated_audio") and os.path.exists(e["generated_audio"]):
                        entries.append(e)

            if not entries:
                continue

            lang = entries[0].get("language_id", "ka")
            print(f"  {ckpt_dir.name}/{test_dir.name}: {len(entries)} files, lang={lang}")

            # Write audio paths to temp file for CTC worker
            audio_paths = [e["generated_audio"] for e in entries]

            # Call CTC via subprocess (batch)
            script = f"""
import sys, json, os
sys.stdout = open(os.devnull, 'w')
import fairseq2n
fairseq2n._check_torch_version = lambda: None
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
sys.stdout = sys.__stdout__

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_1B", device="cuda:0")
paths = json.loads(sys.argv[1])
langs = ["{lang}"] * len(paths)
BATCH = 16
all_texts = []
for i in range(0, len(paths), BATCH):
    batch = paths[i:i+BATCH]
    try:
        texts = pipeline.transcribe(batch, lang=langs[:len(batch)], batch_size=BATCH)
        all_texts.extend([t if isinstance(t, str) else str(t) for t in texts])
    except Exception as e:
        print(f"Batch error: {{e}}", file=sys.stderr)
        all_texts.extend([""] * len(batch))
print(json.dumps(all_texts, ensure_ascii=False))
"""
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

            result = subprocess.run(
                [CTC_PYTHON, "-c", script, json.dumps(audio_paths)],
                capture_output=True, text=True, env=env, timeout=600,
            )

            if result.returncode != 0:
                print(f"    ASR failed: {result.stderr[:200]}")
                continue

            try:
                transcripts = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                print(f"    Failed to parse ASR output")
                continue

            # Save transcripts
            with open(transcript_path, "w") as f:
                for e, t in zip(entries, transcripts):
                    e["asr_text"] = t
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

            print(f"    Saved {len(transcripts)} transcripts")


def compute_metrics():
    """Compute CER, WER, and language leakage from transcripts."""
    print(f"\n{'='*60}")
    print("Computing Metrics")
    print(f"{'='*60}")

    from Levenshtein import distance as lev_distance

    def cer(ref, hyp):
        ref = re.sub(r'[.,!?;:\-\\\"\'()«»„""\[\]{}#\n]', '', ref).lower().strip()
        hyp = re.sub(r'[.,!?;:\-\\\"\'()«»„""\[\]{}#\n]', '', hyp).lower().strip()
        if not ref:
            return 0.0
        return lev_distance(ref, hyp) / len(ref)

    def wer(ref, hyp):
        ref_words = re.sub(r'[.,!?;:\-\\\"\'()«»„""\[\]{}#\n]', '', ref).lower().split()
        hyp_words = re.sub(r'[.,!?;:\-\\\"\'()«»„""\[\]{}#\n]', '', hyp).lower().split()
        if not ref_words:
            return 0.0
        return lev_distance(" ".join(ref_words), " ".join(hyp_words)) / len(" ".join(ref_words))

    def has_georgian(text):
        return bool(re.search(r'[\u10D0-\u10FF]', text))

    all_results = {}

    for ckpt_dir in sorted(RESULTS_DIR.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        ckpt_tag = ckpt_dir.name
        all_results[ckpt_tag] = {}

        for test_dir in sorted(ckpt_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            test_name = test_dir.name

            transcript_path = test_dir / "transcripts.jsonl"
            if not transcript_path.exists():
                continue

            entries = []
            with open(transcript_path) as f:
                for line in f:
                    entries.append(json.loads(line))

            if not entries:
                continue

            cers = []
            wers = []
            leakage_count = 0

            for e in entries:
                ref_text = e["text"]
                asr_text = e.get("asr_text", "")
                if not asr_text:
                    continue

                c = cer(ref_text, asr_text)
                w = wer(ref_text, asr_text)
                cers.append(c)
                wers.append(w)

                # Language leakage for English tests
                if e.get("language_id") == "en" and has_georgian(asr_text):
                    leakage_count += 1

            if not cers:
                continue

            metrics = {
                "test_set": test_name,
                "num_samples": len(cers),
                "cer_mean": round(sum(cers) / len(cers) * 100, 2),
                "wer_mean": round(sum(wers) / len(wers) * 100, 2),
                "cer_median": round(sorted(cers)[len(cers)//2] * 100, 2),
            }

            if "en" in test_name:
                metrics["leakage_pct"] = round(leakage_count / len(cers) * 100, 1)

            all_results[ckpt_tag][test_name] = metrics

            # Save per-test metrics
            with open(test_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    # Header
    header = f"{'Checkpoint':<20s} | {'KA-CER%':>8s} | {'KA-WER%':>8s} | {'FL-CER%':>8s} | {'EN-Cross%':>9s} | {'EN-Regr%':>8s} | {'Leakage%':>8s}"
    print(header)
    print("-" * len(header))

    for ckpt_tag in sorted(all_results.keys()):
        r = all_results[ckpt_tag]
        ka_cer = r.get("cv_geo", {}).get("cer_mean", "-")
        ka_wer = r.get("cv_geo", {}).get("wer_mean", "-")
        fl_cer = r.get("fleurs_ka", {}).get("cer_mean", "-")
        en_cross = r.get("fleurs_en_cross", {}).get("cer_mean", "-")
        en_regr = r.get("fleurs_en_regress", {}).get("cer_mean", "-")
        leakage = r.get("fleurs_en_cross", {}).get("leakage_pct", "-")

        print(f"{ckpt_tag:<20s} | {str(ka_cer):>8s} | {str(ka_wer):>8s} | {str(fl_cer):>8s} | {str(en_cross):>9s} | {str(en_regr):>8s} | {str(leakage):>8s}")

    # Save full results
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_DIR / 'comparison.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "transcribe", "metrics", "all"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint tag (e.g. 099_ckpt1000) or 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test-set", type=str, default=None,
                        help="Specific test set (cv_geo, fleurs_ka, fleurs_en_cross, fleurs_en_regress)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_lists = TEST_LISTS
    if args.test_set:
        test_lists = {args.test_set: TEST_LISTS[args.test_set]}

    if args.stage in ("generate", "all"):
        if args.checkpoint and args.checkpoint != "all":
            ckpts = {args.checkpoint: CHECKPOINTS[args.checkpoint]}
        else:
            ckpts = CHECKPOINTS

        for tag, path in ckpts.items():
            generate_for_checkpoint(tag, path, test_lists, device)

    if args.stage in ("transcribe", "all"):
        transcribe_all(device)

    if args.stage in ("metrics", "all"):
        compute_metrics()


if __name__ == "__main__":
    main()
