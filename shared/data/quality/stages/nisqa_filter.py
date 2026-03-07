"""Stage: NISQA quality filtering.

Uses NISQA (Non-Intrusive Speech Quality Assessment) to filter low-quality audio.
Returns 5 dimensions: MOS, noisiness, discontinuity, coloration, loudness.
Saves ALL scores to nisqa_scores.json for later re-filtering at any threshold.
"""

import json
import numpy as np
import torch
from tqdm import tqdm

from ..audio_io import load
from ..config import PipelineContext, NISQA_THRESHOLD

NAME = "nisqa_filter"
DESCRIPTION = f"Filter by NISQA MOS score >= {NISQA_THRESHOLD}"


def run(entries, ctx: PipelineContext):
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

    nisqa = NonIntrusiveSpeechQualityAssessment(fs=24000)

    kept = []
    dropped = 0
    all_scores = {}  # id -> all 5 NISQA dimensions

    for entry in tqdm(entries, desc="NISQA filter"):
        try:
            waveform, sr = load(entry["audio_path"])
            audio = waveform.squeeze(0)

            if sr != 24000:
                import torchaudio
                audio = torchaudio.transforms.Resample(sr, 24000)(audio.unsqueeze(0)).squeeze(0)

            with torch.no_grad():
                result = nisqa(audio)

            mos = result[0].item()
            score_dict = {
                "mos": mos,
                "noisiness": result[1].item(),
                "discontinuity": result[2].item(),
                "coloration": result[3].item(),
                "loudness": result[4].item(),
            }
            all_scores[entry["id"]] = score_dict

            if mos >= NISQA_THRESHOLD:
                entry_copy = entry.copy()
                entry_copy["nisqa_mos"] = mos
                entry_copy["nisqa_noisiness"] = score_dict["noisiness"]
                entry_copy["nisqa_discontinuity"] = score_dict["discontinuity"]
                entry_copy["nisqa_coloration"] = score_dict["coloration"]
                entry_copy["nisqa_loudness"] = score_dict["loudness"]
                kept.append(entry_copy)
            else:
                dropped += 1

        except Exception as e:
            ctx.logger.debug(f"NISQA error for {entry['id']}: {e}")
            dropped += 1

    # Save ALL scores to JSON for later re-filtering
    scores_path = ctx.output_dir / "nisqa_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    ctx.logger.info(f"Saved NISQA scores for {len(all_scores)} entries to {scores_path}")

    mos_values = [s["mos"] for s in all_scores.values()]
    if mos_values:
        ctx.logger.info(
            f"NISQA filter: {len(kept)}/{len(entries)} kept (dropped {dropped}). "
            f"MOS stats: mean={np.mean(mos_values):.3f}, median={np.median(mos_values):.3f}, "
            f"min={np.min(mos_values):.3f}, max={np.max(mos_values):.3f}"
        )
    return kept
