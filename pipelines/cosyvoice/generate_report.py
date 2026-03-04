"""
Generate a markdown report from CosyVoice 3 evaluation results.

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
        raise FileNotFoundError(f"No results at {results_path}. Run evaluate.py first.")

    with open(results_path) as f:
        results = json.load(f)

    lines = [
        "# CosyVoice 3 — Georgian TTS Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Model",
        "",
        "| Property | Value |",
        "|----------|-------|",
        "| Architecture | LLM + conditional flow matching (DiT) |",
        "| Parameters | ~0.5B |",
        "| Base model | FunAudioLLM/CosyVoice (v3) |",
        "| Fine-tuning | SFT |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for key, label, val_key in [
        ("intelligibility", "CER", "mean_cer"), ("naturalness", "UTMOS", "mean_score"),
        ("speaker_similarity", "Speaker sim", "mean_similarity"), ("fad", "FAD", "fad_score"),
    ]:
        if key in results and val_key in results.get(key, {}):
            lines.append(f"| {label} | {results[key][val_key]:.4f} |")

    report = "\n".join(lines)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output", type=str, default="report.md")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
