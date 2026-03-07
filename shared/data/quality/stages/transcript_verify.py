"""Stage: Transcript verification via round-trip ASR.

Uses Meta Omnilingual ASR 7B to transcribe audio, then compares to original
text via Character Error Rate. Drops clips with CER > threshold.
This is our addition beyond the Catalan paper — we have ground truth transcripts.
"""

import numpy as np
from tqdm import tqdm

from ..config import PipelineContext, CER_THRESHOLD

NAME = "transcript_verify"
DESCRIPTION = f"Round-trip ASR transcript verification, CER <= {CER_THRESHOLD}"


def _compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    ref = " ".join(reference.strip().split())
    hyp = " ".join(hypothesis.strip().split())

    if not ref:
        return 1.0 if hyp else 0.0

    r, h = list(ref), list(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(r)][len(h)] / len(r)


def run(entries, ctx: PipelineContext):
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError:
        ctx.logger.error("omnilingual-asr not installed. pip install omnilingual-asr")
        raise

    ctx.logger.info("Loading Meta Omnilingual ASR 7B model...")
    asr = ASRInferencePipeline(model_card="omniASR_LLM_7B", device=ctx.device)

    BATCH_SIZE = 64
    kept = []
    dropped = 0
    cers = []

    for i in tqdm(range(0, len(entries), BATCH_SIZE), desc="Transcript verify"):
        batch = entries[i : i + BATCH_SIZE]
        audio_paths = [e["audio_path"] for e in batch]
        langs = ["kat_Geor"] * len(batch)

        try:
            results = asr.transcribe(audio_paths, lang=langs, batch_size=len(batch))
            if not isinstance(results, list):
                results = [str(results)]

            for entry, hypothesis in zip(batch, results):
                cer = _compute_cer(hypothesis, entry["text"])
                cers.append(cer)

                if cer <= CER_THRESHOLD:
                    entry_copy = entry.copy()
                    entry_copy["asr_cer"] = cer
                    entry_copy["asr_text"] = hypothesis
                    kept.append(entry_copy)
                else:
                    dropped += 1
        except Exception as e:
            ctx.logger.debug(f"ASR batch error at index {i}: {e}")
            for entry in batch:
                try:
                    res = asr.transcribe([entry["audio_path"]], lang=["kat_Geor"], batch_size=1)
                    hypothesis = res[0] if isinstance(res, list) else str(res)
                    cer = _compute_cer(hypothesis, entry["text"])
                    cers.append(cer)
                    if cer <= CER_THRESHOLD:
                        entry_copy = entry.copy()
                        entry_copy["asr_cer"] = cer
                        entry_copy["asr_text"] = hypothesis
                        kept.append(entry_copy)
                    else:
                        dropped += 1
                except Exception as e2:
                    ctx.logger.debug(f"ASR error for {entry['id']}: {e2}")
                    dropped += 1

    if cers:
        ctx.logger.info(
            f"Transcript verify: {len(kept)}/{len(entries)} kept (dropped {dropped}). "
            f"CER stats: mean={np.mean(cers):.3f}, median={np.median(cers):.3f}, "
            f"p90={np.percentile(cers, 90):.3f}"
        )
    return kept
