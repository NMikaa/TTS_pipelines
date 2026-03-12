"""
Evaluate which checkpoint × speaker produces the best results.

Step 1 (this script, F5-TTS env): Generate audio + compute speaker similarity
Step 2 (separate env with fairseq2): Run CER via Meta Omnilingual ASR on generated audio

Uses F5TTS high-level API for inference (same as user's notebook).

Usage:
    python eval_speakers.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = Path("ckpts/georgian_tts")
SPEAKER_REFS = REPO_ROOT / "data/clean/speaker_refs_manifest.json"
VOCAB_FILE = str(Path(__file__).resolve().parent / "data/extended_vocab.txt")
OUTPUT_DIR = Path("eval_speaker_results")

CHECKPOINTS = ["model_100000.pt", "model_110000.pt"]

# Georgian test sentences — varied lengths and complexity
TEST_SENTENCES = [
    "საქართველო მდებარეობს კავკასიის რეგიონში, ევროპისა და აზიის გასაყარზე.",
    "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია.",
    "ქართული ენა მსოფლიოში ერთ-ერთი უძველესი ენაა.",
    "კახეთი ცნობილია თავისი ღვინით და ულამაზესი ბუნებით.",
    "საქართველოს მდიდარი ისტორია და კულტურული მემკვიდრეობა აქვს.",
]


def load_speaker_refs():
    """Group speaker refs by speaker_id, pick best one per speaker (highest MOS)."""
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
        best_refs[sid] = {"audio_path": str(audio_path), "text": best["text"], "mos": best.get("nisqa_mos", 0)}

    return best_refs


def load_f5tts(checkpoint_path):
    """Load F5TTS using high-level API."""
    from f5_tts.api import F5TTS

    model = F5TTS(
        ckpt_file=checkpoint_path,
        vocab_file=VOCAB_FILE,
        device="cuda",
        use_ema=False,
    )
    return model


def generate(model, ref_audio, ref_text, gen_text, output_path):
    """Generate audio using F5TTS.infer()."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wav, sr, _ = model.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
    )
    sf.write(output_path, wav, sr)
    return output_path, sr


# ── Speaker Similarity ───────────────────────────────────────

def get_classifier():
    """Load ECAPA-TDNN once, cached."""
    if not hasattr(get_classifier, "_model"):
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        from speechbrain.inference.speaker import EncoderClassifier
        get_classifier._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
        )
    return get_classifier._model


def compute_speaker_similarity(ref_path, gen_path):
    """ECAPA-TDNN cosine similarity."""
    classifier = get_classifier()
    ref_emb = classifier.encode_batch(classifier.load_audio(ref_path).unsqueeze(0).cuda())
    gen_emb = classifier.encode_batch(classifier.load_audio(gen_path).unsqueeze(0).cuda())
    sim = torch.nn.functional.cosine_similarity(ref_emb.squeeze(), gen_emb.squeeze(), dim=0)
    return sim.item()


# ── Main ─────────────────────────────────────────────────────

def main():
    speaker_refs = load_speaker_refs()
    print(f"Found {len(speaker_refs)} speakers in refs")
    for sid, ref in sorted(speaker_refs.items(), key=lambda x: x[0]):
        print(f"  Speaker {sid}: MOS={ref['mos']:.2f}, {Path(ref['audio_path']).name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for ckpt_name in CHECKPOINTS:
        ckpt_path = str(CKPT_DIR / ckpt_name)
        print(f"\n{'='*60}")
        print(f"Loading {ckpt_name}...")
        model = load_f5tts(ckpt_path)

        for sid, ref in sorted(speaker_refs.items(), key=lambda x: x[0]):
            sims = []
            print(f"\n  Speaker {sid} x {ckpt_name}:", flush=True)

            for i, sentence in enumerate(TEST_SENTENCES):
                out_path = OUTPUT_DIR / ckpt_name.replace(".pt", "") / f"spk{sid}_sent{i}.wav"
                gen_path, _ = generate(model, ref["audio_path"], ref["text"], sentence, str(out_path))

                # Speaker similarity
                sim = compute_speaker_similarity(ref["audio_path"], gen_path)
                sims.append(sim)

                print(f"    Sent {i}: sim={sim:.4f}  [{sentence[:50]}...]")

            avg_sim = np.mean(sims)
            results.append({
                "checkpoint": ckpt_name,
                "speaker_id": sid,
                "avg_similarity": float(avg_sim),
                "per_sentence_sim": [float(s) for s in sims],
                "sentences": TEST_SENTENCES,
            })
            print(f"    AVG sim={avg_sim:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Checkpoint × Speaker (sorted by similarity desc)")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Speaker':<10} {'Avg Sim':<10}")
    print("-" * 40)
    for r in sorted(results, key=lambda x: -x["avg_similarity"]):
        print(f"{r['checkpoint']:<20} {r['speaker_id']:<10} {r['avg_similarity']:.4f}")

    # Best per checkpoint
    print(f"\n{'='*60}")
    print("BEST SPEAKER PER CHECKPOINT")
    print(f"{'='*60}")
    for ckpt in CHECKPOINTS:
        ckpt_results = [r for r in results if r["checkpoint"] == ckpt]
        best = max(ckpt_results, key=lambda x: x["avg_similarity"])
        print(f"  {ckpt}: Speaker {best['speaker_id']}  sim={best['avg_similarity']:.4f}")

    print(f"\nGenerated audio saved in {OUTPUT_DIR}/")
    print("Run CER evaluation separately with Meta Omnilingual ASR (needs fairseq2/PyTorch 2.8 env)")

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
