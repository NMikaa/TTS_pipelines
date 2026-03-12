"""Pipeline runner — orchestrates stages with checkpoint/resume and manifests."""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*implementation will be changed.*")
warnings.filterwarnings("ignore", message=".*clipped samples.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")

from .config import PipelineContext, DEFAULT_STAGES
from .stages import get_stage, available_stages

logger = logging.getLogger("quality_pipeline")


def _setup_logging(output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(output_dir / "quality_pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
        force=True,
    )
    return log_file


def _load_checkpoint(path: Path) -> Tuple[int, List[Dict]]:
    if not path.exists():
        return -1, []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("last_stage", -1), data.get("entries", [])


def _save_checkpoint(path: Path, stage_idx: int, entries: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"last_stage": stage_idx, "entries": entries}, f)
    logger.info(f"Checkpoint saved: stage {stage_idx}, {len(entries)} entries")


def _save_stage_manifests(
    output_dir: Path, stage_idx: int, stage_name: str,
    entries_before: List[Dict], entries_after: List[Dict],
):
    stages_dir = output_dir / "stage_manifests"
    stages_dir.mkdir(parents=True, exist_ok=True)

    kept_ids = {e["id"] for e in entries_after}
    dropped = [e for e in entries_before if e["id"] not in kept_ids]

    kept_path = stages_dir / f"stage{stage_idx}_{stage_name}_kept.json"
    dropped_path = stages_dir / f"stage{stage_idx}_{stage_name}_dropped.json"

    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(entries_after, f, ensure_ascii=False, indent=2)
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump(dropped, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Manifests: {len(entries_after)} kept -> {kept_path.name}, "
        f"{len(dropped)} dropped -> {dropped_path.name}"
    )


def run_pipeline(
    data_dir: str,
    output_dir: str,
    stages: Optional[List[str]] = None,
    resume: bool = True,
    device: str = "cuda",
) -> List[Dict]:
    """Run the quality pipeline.

    Args:
        data_dir: Path to data directory with manifest and audio/.
        output_dir: Path to output directory for clean data.
        stages: List of stage names to run (default: all from config).
        resume: Whether to resume from checkpoint.
        device: CUDA device string.

    Returns:
        List of cleaned entries.
    """
    output_path = Path(output_dir)
    log_file = _setup_logging(output_path)
    logger.info(f"Log file: {log_file}")

    # Validate requested stages
    if stages is None:
        stages = DEFAULT_STAGES

    for s in stages:
        if s not in available_stages():
            raise ValueError(f"Unknown stage '{s}'. Available: {available_stages()}")

    logger.info(f"Pipeline stages: {stages}")

    # Build context
    ctx = PipelineContext(
        data_dir=Path(data_dir),
        output_dir=output_path,
        cache_dir=output_path / ".cache",
        audio_dir=output_path / "audio_clean",
        device=device,
        logger=logger,
    )
    ctx.cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / ".pipeline_checkpoint.json"

    # Load checkpoint
    last_stage = -1
    entries = []
    if resume:
        last_stage, entries = _load_checkpoint(checkpoint_path)
        if last_stage >= 0:
            logger.info(
                f"Resuming from after stage {last_stage} "
                f"({stages[last_stage] if last_stage < len(stages) else '?'}), "
                f"{len(entries)} entries"
            )

    # Initial load
    if last_stage < 0:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from shared.data import get_splits
        logger.info("Loading dataset...")
        train, val, test = get_splits(data_dir)
        entries = train + val + test
        logger.info(f"Loaded {len(entries)} entries")

    total_start = len(entries)

    # Run stages
    for idx, stage_name in enumerate(stages):
        if idx <= last_stage:
            logger.info(f"Skipping stage {idx} ({stage_name}) — already completed")
            continue

        stage_module = get_stage(stage_name)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Stage {idx}: {stage_name} — {stage_module.DESCRIPTION} ({len(entries)} entries)")
        logger.info(f"{'=' * 60}")

        entries_before = entries
        t0 = time.time()
        entries = stage_module.run(entries_before, ctx)
        elapsed = time.time() - t0

        logger.info(f"Stage {idx} ({stage_name}) done in {elapsed:.1f}s — {len(entries)} entries remaining")

        _save_stage_manifests(output_path, idx, stage_name, entries_before, entries)
        _save_checkpoint(checkpoint_path, idx, entries)

    # Save final manifest
    manifest_path = output_path / "clean_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete: {len(entries)}/{total_start} entries kept ({100 * len(entries) / total_start:.1f}%)")
    logger.info(f"Clean manifest: {manifest_path}")
    logger.info(f"{'=' * 60}")

    return entries


def main():
    available = available_stages()
    parser = argparse.ArgumentParser(description="Data quality pipeline for Georgian TTS")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./data/clean")
    parser.add_argument(
        "--stages", type=str, nargs="+", default=None,
        help=f"Stages to run (default: all). Available: {available}",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stages=args.stages,
        resume=not args.no_resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
