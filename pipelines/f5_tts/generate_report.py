"""
Generate a markdown report from F5-TTS evaluation results.

Usage:
    python generate_report.py --results-dir results/ --output report.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def generate_report(results_dir: str, output: str):
    results_path = Path(results_dir) / "evaluation.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No evaluation results found at {results_path}. Run evaluate.py first.")

    with open(results_path) as f:
        results = json.load(f)

    lines = [
        "# F5-TTS — Georgian TTS Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Model",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Architecture | Non-autoregressive flow matching (DiT + ConvNeXt V2) |",
        "| Parameters | ~335M |",
        "| Base model | SWivid/F5-TTS |",
        "| Fine-tuning | Full fine-tune |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if "intelligibility" in results:
        r = results["intelligibility"]
        if "mean_cer" in r:
            lines.append(f"| CER ({r.get('asr_model', 'whisper')}) | {r['mean_cer']:.4f} |")
        elif "error" in r:
            lines.append(f"| CER | Error: {r['error']} |")

    if "naturalness" in results:
        r = results["naturalness"]
        if "mean_score" in r:
            lines.append(f"| UTMOS | {r['mean_score']:.4f} |")

    if "speaker_similarity" in results:
        r = results["speaker_similarity"]
        if "mean_similarity" in r:
            lines.append(f"| Speaker similarity (ECAPA-TDNN) | {r['mean_similarity']:.4f} |")

    if "fad" in results:
        r = results["fad"]
        if "fad_score" in r:
            lines.append(f"| FAD (VGGish) | {r['fad_score']:.4f} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- CER is computed using ASR transcription of generated audio vs ground truth text",
        "- UTMOS is trained on English data; scores are useful for relative comparison, not absolute quality",
        "- Speaker similarity measures voice identity preservation (cosine similarity of ECAPA-TDNN embeddings)",
        "- FAD measures distributional similarity to real Georgian speech",
    ])

    report = "\n".join(lines)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate F5-TTS evaluation report")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output", type=str, default="report.md")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
