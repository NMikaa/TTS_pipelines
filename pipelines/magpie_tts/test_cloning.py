"""Quick CPU test of zero-shot voice cloning from a training checkpoint."""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"  # some parallelism for CPU inference

import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NeMo"))

SAMPLE_RATE = 22050
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cloning_model(ckpt_path):
    """Load MagPIE TTS with cloning checkpoint on CPU."""
    from nemo.collections.tts.models import MagpieTTSModel
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

    print("Loading pretrained base model...")
    model = MagpieTTSModel.from_pretrained(
        'nvidia/magpie_tts_multilingual_357m', map_location='cpu'
    )

    # Strip baked embeddings BEFORE loading cloning checkpoint,
    # otherwise MagPIE's custom load_state_dict tries to load baked_context_embedding
    # as a regular module and fails (cloning checkpoint doesn't have those keys).
    model.baked_context_embedding = None
    model._baked_embedding_T = None
    model._baked_embedding_D = None
    model.baked_context_embedding_len = None

    print(f"Overlaying cloning checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Use training config's tokenizer
    training_cfg = ckpt['hyper_parameters']['cfg']
    model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)

    model = model.eval().to(DEVICE)
    print(f"has_baked_context_embedding = {model.has_baked_context_embedding}")
    return model


def generate_cloning(model, text, ref_audio_path, use_cfg=False):
    """Generate speech cloning a reference voice. No CFG for speed on CPU."""
    from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

    # Load reference audio
    ref_waveform, sr = torchaudio.load(ref_audio_path)
    if ref_waveform.shape[0] > 1:
        ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        ref_waveform = torchaudio.functional.resample(ref_waveform, sr, SAMPLE_RATE)
    ref_waveform = ref_waveform.squeeze(0).to(DEVICE)  # (time,)
    ref_len = torch.tensor([ref_waveform.shape[-1]], dtype=torch.long, device=DEVICE)
    print(f"Reference audio: {ref_waveform.shape[-1]/SAMPLE_RATE:.2f}s")

    # Chunk text
    chunked_tokens, chunked_tokens_len, _ = chunk_text_for_inference(
        text=text,
        language="ka",
        tokenizer_name="text_ce_tokenizer",
        text_tokenizer=model.tokenizer,
        eos_token_id=model.eos_id,
    )
    num_chunks = len(chunked_tokens)
    print(f"Text chunks: {num_chunks}")

    t0 = time.time()
    with torch.no_grad():
        chunk_state = model.create_chunk_state(batch_size=1)
        all_codes = []

        for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
            t_chunk = time.time()
            batch = {
                'text': toks.unsqueeze(0).to(DEVICE),
                'text_lens': torch.tensor([toks_len], dtype=torch.long, device=DEVICE),
                'context_audio': ref_waveform.unsqueeze(0),  # (1, time)
                'context_audio_lens': ref_len,
                'context_sample_rate': SAMPLE_RATE,
                'context_text_tokens': torch.zeros(1, 1, dtype=torch.long, device=DEVICE),
                'context_text_tokens_lens': torch.zeros(1, dtype=torch.long, device=DEVICE),
                'has_text_context': torch.tensor([False], device=DEVICE),
            }

            output = model.generate_speech(
                batch,
                chunk_state=chunk_state,
                end_of_text=[i == num_chunks - 1],
                beginning_of_text=(i == 0),
                use_cfg=use_cfg,
                use_local_transformer_for_inference=True,
            )

            if output.predicted_codes_lens[0] > 0:
                codes = output.predicted_codes[0, :, :output.predicted_codes_lens[0]]
                all_codes.append(codes)
                print(f"  Chunk {i+1}/{num_chunks}: {codes.shape[1]} frames ({time.time()-t_chunk:.1f}s)")

    if not all_codes:
        print("No audio generated!")
        return None

    # Decode all codes at once
    print("Decoding codes to audio...")
    concat_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
    codes_lens = torch.tensor([concat_codes.shape[2]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        audio, audio_lens, _ = model.codes_to_audio(concat_codes, codes_lens)

    audio_out = audio[0, :audio_lens[0]].cpu().float().unsqueeze(0)
    elapsed = time.time() - t0
    dur = audio_out.shape[-1] / SAMPLE_RATE
    print(f"Generated {dur:.2f}s audio in {elapsed:.1f}s (RTF={elapsed/dur:.1f}x)")
    return audio_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ref-audio", type=str, required=True)
    parser.add_argument("--text", type=str, default="გამარჯობა, მე ვარ ხელოვნური ინტელექტი.")
    parser.add_argument("--output", type=str, default="cloned_test.wav")
    parser.add_argument("--cfg", action="store_true", help="Enable CFG (slower but better quality)")
    args = parser.parse_args()

    model = load_cloning_model(args.checkpoint)
    audio = generate_cloning(model, args.text, args.ref_audio, use_cfg=args.cfg)

    if audio is not None:
        torchaudio.save(args.output, audio, SAMPLE_RATE)
        print(f"Saved: {args.output}")
