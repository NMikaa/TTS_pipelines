"""
CSM-1B dataset utilities.

Loads JSONL manifests and preprocesses entries into the CSM chat format
expected by processor.apply_chat_template().
"""

import json
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path):
    """Load a JSONL manifest file. Each line is a JSON object with keys:
    id, audio_path, text, speaker_id, duration_sec."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def preprocess_example(example, processor, speaker_key="speaker_id",
                       max_text_tokens=512, max_audio_samples=240000):
    """Convert a manifest entry to CSM training tensors.

    Returns a dict with keys: input_ids, attention_mask, labels,
    input_values, input_values_cutoffs, text, speaker_id.
    Returns None on processing failure.
    """
    audio_path = example["audio_path"]
    if not Path(audio_path).is_absolute():
        audio_path = str(REPO_ROOT / audio_path)

    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if len(audio_np) > max_audio_samples:
        audio_np = audio_np[:max_audio_samples]

    conversation = [
        {
            "role": str(example[speaker_key]),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": audio_np},
            ],
        }
    ]

    try:
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length",
                "max_length": max_text_tokens,
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": 24000,
                "max_length": max_audio_samples + 1,
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )
    except Exception as e:
        print(f"Error processing '{example['text'][:50]}...': {e}")
        return None

    result = {}
    for key in ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]:
        if key not in model_inputs:
            return None
        result[key] = model_inputs[key][0]

    result["text"] = example["text"]
    result["speaker_id"] = str(example[speaker_key])
    return result
