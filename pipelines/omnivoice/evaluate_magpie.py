"""Evaluate MagPIE TTS Georgian on the same normalized FLEURS-KA test set as OmniVoice.

Runs in the main venv (/venv/main) since MagPIE is a NeMo model.

Usage:
    /venv/main/bin/python pipelines/omnivoice/evaluate_magpie.py
"""

import os
import re
import json
import sys
import torch
import torchaudio
from pathlib import Path
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR / "eval_data"
RESULTS_DIR = SCRIPT_DIR / "eval_results" / "magpie_tts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_NAME = "text_ce_tokenizer"
MAX_TOKENS_PER_CHUNK = 400


def split_georgian_text(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for sentence in sentences:
        est_tokens = len(sentence.encode('utf-8'))
        if est_tokens <= MAX_TOKENS_PER_CHUNK:
            chunks.append(sentence)
            continue
        clauses = re.split(r'(?<=[,;:—])\s+', sentence)
        current = ""
        for clause in clauses:
            combined = f"{current} {clause}".strip() if current else clause
            if len(combined.encode('utf-8')) <= MAX_TOKENS_PER_CHUNK:
                current = combined
            else:
                if current:
                    chunks.append(current)
                if len(clause.encode('utf-8')) > MAX_TOKENS_PER_CHUNK:
                    words = clause.split()
                    current = ""
                    for word in words:
                        combined = f"{current} {word}".strip() if current else word
                        if len(combined.encode('utf-8')) <= MAX_TOKENS_PER_CHUNK:
                            current = combined
                        else:
                            if current:
                                chunks.append(current)
                            current = word
                else:
                    current = clause
        if current:
            chunks.append(current)
    return [c for c in chunks if c.strip()]


def tokenize_chunks(chunks, tokenizer, eos_id):
    out_tokens, out_lens = [], []
    for chunk in chunks:
        tokens = tokenizer.encode(text=chunk, tokenizer_name=TOKENIZER_NAME)
        tokens = tokens + [eos_id]
        out_tokens.append(torch.tensor(tokens, dtype=torch.int32))
        out_lens.append(len(tokens))
    return out_tokens, out_lens


def synthesize(model, text, speaker=1):
    text = text.strip()
    if text[-1] not in ".!?,:;":
        text += "."
    chunks = split_georgian_text(text)
    chunked_tokens, chunked_tokens_len = tokenize_chunks(chunks, model.tokenizer, model.eos_id)
    chunk_state = model.create_chunk_state(batch_size=1)
    all_codes = []
    for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
        batch = {
            "text": toks.unsqueeze(0).cuda(),
            "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
            "speaker_indices": speaker,
        }
        with torch.no_grad():
            output = model.generate_speech(
                batch,
                chunk_state=chunk_state,
                end_of_text=[i == len(chunked_tokens) - 1],
                beginning_of_text=(i == 0),
                use_cfg=True,
                use_local_transformer_for_inference=True,
            )
        if output.predicted_codes_lens[0] > 0:
            all_codes.append(output.predicted_codes[0, :, :output.predicted_codes_lens[0]])
    if not all_codes:
        return None
    codes = torch.cat(all_codes, dim=1).unsqueeze(0)
    codes_lens = torch.tensor([codes.shape[2]], device="cuda", dtype=torch.long)
    audio, audio_lens, _ = model._codec_helper.codes_to_audio(codes, codes_lens)
    return audio[0, :audio_lens[0]].cpu().float()


def main():
    print("Loading MagPIE TTS Georgian...")
    # Optional: add local NeMo clone to path if installed there
    nemo_path = os.environ.get("NEMO_PATH", str(SCRIPT_DIR.parent.parent / "NeMo"))
    if os.path.exists(nemo_path):
        sys.path.insert(0, nemo_path)
    from nemo.collections.tts.models.magpietts import MagpieTTSModel

    nemo_path = hf_hub_download(repo_id="NMikka/Magpie-TTS-Geo-357m", filename="magpie_tts_georgian.nemo")
    model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    model = model.eval().cuda()
    print("Model loaded")

    # Run on FLEURS-KA only (MagPIE doesn't support voice cloning, no English)
    test_list_path = EVAL_DIR / "test_lists" / "fleurs_ka.jsonl"
    out_dir = RESULTS_DIR / "fleurs_ka"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(l) for l in open(test_list_path)]
    print(f"Total samples: {len(samples)}")

    results = []
    for i, sample in enumerate(samples):
        try:
            text = sample["text"]  # already normalized
            wav = synthesize(model, text, speaker=1)
            if wav is None:
                raise RuntimeError("No audio generated")
            wav_path = out_dir / f"{sample['id']}.wav"
            torchaudio.save(str(wav_path), wav.unsqueeze(0), 22050)
            results.append({
                "id": sample["id"],
                "text": text,
                "generated_audio": str(wav_path),
                "language_id": "ka",
                "duration": wav.shape[0] / 22050,
            })
        except Exception as e:
            print(f"  Failed {sample['id']}: {e}")
            results.append({
                "id": sample["id"],
                "text": sample["text"],
                "generated_audio": "",
                "error": str(e),
                "language_id": "ka",
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}")

    manifest_path = out_dir / "generated.jsonl"
    with open(manifest_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    success = sum(1 for r in results if r.get("generated_audio"))
    print(f"\nDone: {success}/{len(results)} successful → {manifest_path}")


if __name__ == "__main__":
    main()
