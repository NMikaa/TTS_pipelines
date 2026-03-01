"""
Process CTM alignments into JSON format for CSM training

This script reads the CTM files organized in batch directories
and creates alignments.json file needed for training.
"""

import json
import argparse
from pathlib import Path
from data_loader import MyDataLoader


def main():
    parser = argparse.ArgumentParser(description="Process CTM alignments into JSON format")
    parser.add_argument(
        '--voice_actor_manifest',
        type=str,
        default='C:/Users/nikam/Projects/TTS/alignment/voice_actor_manifest.json',
        help='Path to voice actor manifest'
    )
    parser.add_argument(
        '--alignment_base_path',
        type=str,
        default='C:/Users/nikam/Projects/TTS/alignment',
        help='Base path containing outputs/ directory with batch folders'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='C:/Users/nikam/Projects/TTS/alignments.json',
        help='Output path for alignments JSON'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=0.08,
        help='Minimum word duration in seconds (default: 0.08 = 80ms)'
    )
    parser.add_argument(
        '--list_of_speakers',
        nargs='*',
        default=[],
        help='Optional list of speaker IDs to filter'
    )

    args = parser.parse_args()

    print("="*70)
    print("PROCESSING CTM ALIGNMENTS")
    print("="*70)

    # Validate paths
    if not Path(args.voice_actor_manifest).exists():
        print(f"❌ Voice actor manifest not found: {args.voice_actor_manifest}")
        print("Run generate_alignments_complete.py first!")
        return 1

    alignment_outputs = Path(args.alignment_base_path) / "outputs"
    if not alignment_outputs.exists():
        print(f"❌ Alignment outputs directory not found: {alignment_outputs}")
        print("Run generate_alignments_complete.py first!")
        return 1

    # Count batch directories
    batch_dirs = list(alignment_outputs.glob("batch_*"))
    if len(batch_dirs) == 0:
        print(f"❌ No batch directories found in: {alignment_outputs}")
        return 1

    print(f"\nFound {len(batch_dirs)} batch directories")

    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = MyDataLoader(
        shards_path=None,
        voice_actor_path=args.voice_actor_manifest,
        list_of_speakers=args.list_of_speakers
    )

    print(f"Loaded dataset with {len(data_loader.ds)} samples")

    # Process alignments
    print(f"\nProcessing alignments from: {args.alignment_base_path}")
    print(f"Minimum word duration: {args.min_duration*1000:.0f}ms")

    try:
        alignment_dict = data_loader.process_alignments(
            alignment_base_path=args.alignment_base_path,
            min_duration=args.min_duration
        )
    except Exception as e:
        print(f"\n❌ Error processing alignments: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if len(alignment_dict) == 0:
        print("\n❌ No alignments were processed successfully!")
        print("Check that:")
        print("  1. CTM files exist in outputs/batch_*/")
        print("  2. Audio paths match between manifest and alignment files")
        return 1

    # Save alignments
    print(f"\nSaving alignments to: {args.output_json}")
    data_loader.save_alignments(args.output_json)

    # Statistics
    total_files = len(data_loader.ds)
    aligned_files = len(alignment_dict)
    success_rate = (aligned_files / total_files * 100) if total_files > 0 else 0

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total files in dataset: {total_files}")
    print(f"Successfully aligned: {aligned_files}")
    print(f"Success rate: {success_rate:.2f}%")

    if success_rate < 90:
        print("\n⚠️  Warning: Success rate below 90%!")
        print("This may indicate path mismatch issues.")
        print("\nDebugging tips:")
        print("  1. Check that audio_filepath in manifest matches exactly")
        print("  2. Verify CTM files were generated correctly")
        print("  3. Check for any error messages in the output above")
    else:
        print("\n✅ Alignments processed successfully!")
        print(f"\nReady for training! Run:")
        print(f"  python runner.py --voice_actor_path {args.voice_actor_manifest} --alignment_json_path {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
