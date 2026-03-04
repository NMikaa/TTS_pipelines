"""
F5-TTS inference for Georgian.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --text "გამარჯობა" --output output.wav
    python infer.py --checkpoint checkpoints/best.pt --test-set ./data --output-dir outputs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def generate_single(checkpoint: str, text: str, output: str, reference_audio: str | None = None):
    """Generate a single utterance."""
    # TODO: Load fine-tuned F5-TTS and generate
    raise NotImplementedError("F5-TTS inference not yet implemented.")


def generate_test_set(checkpoint: str, data_dir: str, output_dir: str):
    """Generate audio for all test set samples (for evaluation)."""
    from shared.data import prepare_dataset, get_splits

    entries = prepare_dataset(data_dir)
    _, _, test_ids = get_splits(data_dir)
    test_entries = [e for e in entries if e["id"] in set(test_ids)]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(test_entries)} test samples...")

    for entry in test_entries:
        output_path = Path(output_dir) / f"{entry['id']}.wav"
        # TODO: Generate audio for entry["text"] and save to output_path
        pass

    raise NotImplementedError("F5-TTS inference not yet implemented.")


def main():
    parser = argparse.ArgumentParser(description="F5-TTS inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Single text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path for single generation")
    parser.add_argument("--reference-audio", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--test-set", type=str, default=None, help="Data dir to generate full test set")
    parser.add_argument("--output-dir", type=str, default="outputs/", help="Output dir for test set generation")
    args = parser.parse_args()

    if args.text:
        generate_single(args.checkpoint, args.text, args.output, args.reference_audio)
    elif args.test_set:
        generate_test_set(args.checkpoint, args.test_set, args.output_dir)
    else:
        parser.error("Provide either --text or --test-set")


if __name__ == "__main__":
    main()
