"""
OpenAudio S1-mini (Fish Speech) fine-tuning for Georgian.

This is a wrapper around the Fish Speech 3-step training pipeline:
  1. Extract VQ tokens from audio
  2. Build protobuf dataset
  3. LoRA fine-tune the LLAMA component

Prerequisites:
    git clone https://github.com/fishaudio/fish-speech.git
    cd fish-speech && pip install -e .[cu129]
    huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

Usage:
    python train.py --data-dir ./data --run-name georgian_fish_v1
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import FishSpeechConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import prepare_dataset, get_splits


def prepare_lab_files(data_dir: str, output_dir: str):
    """Convert our unified manifest to Fish Speech .lab format.

    Fish Speech expects:
        output_dir/SPK_ID/clip_id.wav
        output_dir/SPK_ID/clip_id.lab  (plain text transcription)
    """
    entries = prepare_dataset(data_dir)
    train_ids, val_ids, _ = get_splits(data_dir)
    train_set = set(train_ids)

    out = Path(output_dir)
    count = 0
    for entry in entries:
        if entry["id"] not in train_set:
            continue
        spk = entry.get("speaker_id", "SPK0")
        spk_dir = out / spk
        spk_dir.mkdir(parents=True, exist_ok=True)

        # Symlink or copy audio
        src_audio = Path(entry["audio_path"])
        dst_audio = spk_dir / f"{entry['id']}.wav"
        if not dst_audio.exists() and src_audio.exists():
            dst_audio.symlink_to(src_audio.resolve())

        # Write .lab file (plain text transcription)
        lab_path = spk_dir / f"{entry['id']}.lab"
        if not lab_path.exists():
            lab_path.write_text(entry["text"], encoding="utf-8")
        count += 1

    print(f"Prepared {count} samples in Fish Speech format at {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAudio S1-mini on Georgian")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--run-name", type=str, default="georgian_fish_v1")
    parser.add_argument("--fish-speech-dir", type=str, default=None,
                        help="Path to cloned fish-speech repo (if not in PATH)")
    parser.add_argument("--skip-vq", action="store_true", help="Skip VQ extraction (already done)")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset build (already done)")
    args = parser.parse_args()

    config = FishSpeechConfig(data_dir=args.data_dir, run_name=args.run_name)

    # Step 0: Prepare data in Fish Speech format (speaker dirs + .lab files)
    fish_data_dir = str(Path(config.output_dir) / "fish_data")
    print("Step 0: Converting to Fish Speech format (.wav + .lab)...")
    prepare_lab_files(config.data_dir, fish_data_dir)

    # The remaining steps use Fish Speech's own CLI tools.
    # They must be run from within the fish-speech repo directory.
    fish_dir = args.fish_speech_dir
    if fish_dir is None:
        print("\n" + "=" * 60)
        print("DATA PREPARED. Now run the following commands from the fish-speech repo:")
        print("=" * 60)
        print(f"""
# Step 1: Extract VQ tokens
python tools/vqgan/extract_vq.py {Path(fish_data_dir).resolve()} \\
    --num-workers {config.num_workers} --batch-size {config.vq_batch_size} \\
    --config-name "modded_dac_vq" \\
    --checkpoint-path "{Path(config.checkpoint_dir).resolve()}/codec.pth"

# Step 2: Build protobuf dataset
python tools/llama/build_dataset.py \\
    --input "{Path(fish_data_dir).resolve()}" \\
    --output "{Path(fish_data_dir).resolve()}/protos" \\
    --text-extension .lab \\
    --num-workers 16

# Step 3: LoRA fine-tune
python fish_speech/train.py --config-name text2semantic_finetune \\
    project={config.run_name} \\
    +lora@model.model.lora_config={config.lora_config}

# Step 4: Merge LoRA weights
python tools/llama/merge_lora.py \\
    --lora-config {config.lora_config} \\
    --base-weight {Path(config.checkpoint_dir).resolve()} \\
    --lora-weight results/{config.run_name}/checkpoints/<best_step>.ckpt \\
    --output checkpoints/openaudio-s1-mini-georgian/
""")
        print("TIP: Earlier checkpoints often perform better. Try multiple.")
        return

    # If fish-speech-dir is provided, run the steps automatically
    # TODO: Implement automatic pipeline execution
    raise NotImplementedError(
        "Automatic Fish Speech pipeline execution not yet implemented. "
        "Use the printed commands above, or pass --fish-speech-dir to automate."
    )


if __name__ == "__main__":
    main()
