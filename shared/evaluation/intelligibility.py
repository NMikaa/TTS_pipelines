"""
Intelligibility evaluation via round-trip CER.

Round-trip: TTS-generated audio → ASR transcription → compare to original text.
This measures whether the TTS output is understandable, without needing
matched reference audio from the same speaker.

CER is preferred over WER for Georgian because:
- Georgian is agglutinative with long compound words, making WER harsh
- CER better captures partial recognition
- Georgian script is unicameral (no case normalization needed)

ASR backend:
- Meta Omnilingual ASR 7B (1.9% CER on Georgian — state of the art)

WARNING: Do NOT use Whisper for Georgian. It scores 78-88% WER on Georgian,
making CER measurements meaningless.

Usage:
    from shared.evaluation.intelligibility import compute_cer

    results = compute_cer(
        generated_dir="outputs/",
        references={"sample_001.wav": "გამარჯობა მსოფლიო", ...},
    )
"""

import re
from pathlib import Path


def _normalize_text(text: str) -> str:
    """Normalize text for CER computation. Georgian is unicameral so no lowering needed."""
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _levenshtein(ref: list, hyp: list) -> int:
    """Compute Levenshtein distance between two sequences."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref)][len(hyp)]


def _compute_cer_pair(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    ref_chars = list(_normalize_text(reference))
    hyp_chars = list(_normalize_text(hypothesis))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def _compute_wer_pair(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def transcribe_omnilingual(audio_paths: list[str]) -> dict[str, str]:
    """
    Transcribe audio files using Meta Omnilingual ASR 7B.

    Requires: pip install omnilingual-asr
    Model: facebook/omniASR-LLM-7B (~30GB download, ~17GB VRAM)
    Georgian language code: kat_Geor
    """
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
    languages = ["kat_Geor"] * len(audio_paths)
    results = pipeline.transcribe(audio_paths, lang=languages, batch_size=4)

    transcriptions = {}
    for path, text in zip(audio_paths, results):
        transcriptions[Path(path).name] = text

    return transcriptions


def compute_cer(
    generated_dir: str,
    references: dict[str, str],
) -> dict:
    """
    Compute CER for all generated audio files using round-trip evaluation.

    Round-trip: generated audio → Meta Omnilingual ASR → CER vs original text.
    No reference audio needed — this is a text-to-text comparison.

    Args:
        generated_dir: Directory containing generated .wav files
        references: Dict mapping filename -> ground truth text

    Returns:
        Dict with per-sample CER and aggregate statistics
    """
    gen_path = Path(generated_dir)
    audio_files = sorted(gen_path.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {generated_dir}")

    # Filter to files we have references for
    paths = [str(f) for f in audio_files if f.name in references]
    if not paths:
        raise ValueError("No generated files match the reference texts")

    # Transcribe with Omnilingual ASR
    print(f"  Transcribing {len(paths)} files with Meta Omnilingual ASR...")
    transcriptions = transcribe_omnilingual(paths)

    # Compute CER and WER per sample
    per_sample = {}
    for filename, hypothesis in transcriptions.items():
        if filename in references:
            cer = _compute_cer_pair(references[filename], hypothesis)
            wer = _compute_wer_pair(references[filename], hypothesis)
            per_sample[filename] = {
                "cer": cer,
                "wer": wer,
                "reference": references[filename],
                "hypothesis": hypothesis,
            }

    cers = [v["cer"] for v in per_sample.values()]
    wers = [v["wer"] for v in per_sample.values()]
    return {
        "metric": "cer_wer",
        "asr_model": "meta_omnilingual_asr_7b",
        "mean_cer": sum(cers) / len(cers) if cers else 0.0,
        "median_cer": sorted(cers)[len(cers) // 2] if cers else 0.0,
        "mean_wer": sum(wers) / len(wers) if wers else 0.0,
        "median_wer": sorted(wers)[len(wers) // 2] if wers else 0.0,
        "num_samples": len(cers),
        "per_sample": per_sample,
    }
