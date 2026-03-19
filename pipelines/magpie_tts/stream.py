"""
Voice-consistent streaming MagPIE TTS.

Problem: when long text is chunked, per-chunk NanoCodec decoding can introduce
voice discontinuities at boundaries.

Fix:
  - consistent (default): generate all codec codes, decode in one pass — no
    boundary artefacts, guaranteed voice continuity
  - stream: per-chunk decode with cross-fade for lower latency to first audio

Both modes use the baked speaker embeddings (the model's voice-cloning
context_encoder was not trained during Georgian fine-tuning, so external
reference audio produces near-empty output).

Usage:
    python stream.py --text "გრძელი ქართული ტექსტი..." --speaker 1
    python stream.py --text "ტექსტი..." --speaker 1 --stream
    python stream.py --text "ტექსტი..." --no-cfg              # ~2x faster
    python stream.py --text "ტექსტი..." --temperature 0.4     # more deterministic
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SAMPLE_RATE = 22050


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(source: str = "NMikka/Magpie-TTS-Geo-357m"):
    """Load MagPIE TTS from HuggingFace repo, .nemo file, or .ckpt."""
    from nemo.collections.tts.models import MagpieTTSModel

    source = str(source)
    if source.endswith(".nemo"):
        model = MagpieTTSModel.restore_from(source, map_location="cpu")
    elif source.endswith(".ckpt"):
        from generate import load_model as _load_ckpt
        return _load_ckpt(source)
    else:
        from huggingface_hub import hf_hub_download
        nemo_path = hf_hub_download(repo_id=source, filename="magpie_tts_georgian.nemo")
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")

    return model.eval().cuda()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(model, text: str):
    from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference
    return chunk_text_for_inference(
        text=text,
        language="ka",
        tokenizer_name="text_ce_tokenizer",
        text_tokenizer=model.tokenizer,
        eos_token_id=model.eos_id,
    )


def _apply_inference_params(model, temperature=None, topk=None, cfg_scale=None):
    """Override inference parameters, return dict of originals for restore."""
    ip = model.inference_parameters
    orig = {}
    for key, val in [("temperature", temperature), ("topk", topk), ("cfg_scale", cfg_scale)]:
        if val is not None:
            orig[key] = getattr(ip, key)
            setattr(ip, key, val)
    return orig


def _restore_inference_params(model, orig):
    ip = model.inference_parameters
    for key, val in orig.items():
        setattr(ip, key, val)


def _generate_all_codes(model, text, speaker_idx=1, use_cfg=True):
    """Run chunked generate_speech with baked speaker, return per-chunk code tensors."""
    chunked_tokens, chunked_tokens_len, _ = _chunk_text(model, text)
    num_chunks = len(chunked_tokens)

    chunk_state = model.create_chunk_state(batch_size=1)
    all_codes = []

    for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
        batch = {
            "text": toks.unsqueeze(0).cuda(),
            "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
            "speaker_indices": speaker_idx,
        }

        with torch.no_grad():
            output = model.generate_speech(
                batch,
                chunk_state=chunk_state,
                end_of_text=[i == num_chunks - 1],
                beginning_of_text=(i == 0),
                use_cfg=use_cfg,
                use_local_transformer_for_inference=True,
            )

        if output.predicted_codes_lens[0] > 0:
            all_codes.append(
                output.predicted_codes[0, :, : output.predicted_codes_lens[0]]
            )

    return all_codes


def _decode_codes(model, codes_list):
    """Concatenate code tensors and decode to waveform (1, samples)."""
    if not codes_list:
        return None
    codes = torch.cat(codes_list, dim=1).unsqueeze(0)
    codes_lens = torch.tensor([codes.shape[2]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        audio, audio_lens, _ = model.codes_to_audio(codes, codes_lens)
    return audio[0, : audio_lens[0]].cpu().float().unsqueeze(0)


def _decode_single_codes(model, codes_tensor):
    """Decode a single chunk's codes to waveform (1D)."""
    codes = codes_tensor.unsqueeze(0)
    codes_lens = torch.tensor([codes.shape[2]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        audio, audio_lens, _ = model.codes_to_audio(codes, codes_lens)
    return audio[0, : audio_lens[0]].cpu().float()


def _crossfade(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Cross-fade 1D tensors *a* and *b* over *n* samples."""
    if n <= 0 or len(a) < n or len(b) < n:
        return torch.cat([a, b])
    fade_out = torch.linspace(1.0, 0.0, n)
    fade_in = torch.linspace(0.0, 1.0, n)
    mixed = a[-n:] * fade_out + b[:n] * fade_in
    return torch.cat([a[:-n], mixed, b[n:]])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_consistent(
    model,
    text: str,
    speaker_idx: int = 1,
    use_cfg: bool = True,
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    cfg_scale: Optional[float] = None,
) -> torch.Tensor:
    """Generate speech with consistent voice — all codes decoded in one pass.

    Returns:
        (1, samples) float32 tensor at 22050 Hz, or None.
    """
    orig = _apply_inference_params(model, temperature, topk, cfg_scale)
    try:
        t0 = time.time()
        codes = _generate_all_codes(model, text, speaker_idx=speaker_idx, use_cfg=use_cfg)
        audio = _decode_codes(model, codes)
        elapsed = time.time() - t0
        if audio is not None:
            dur = audio.shape[-1] / SAMPLE_RATE
            print(f"Generated {dur:.2f}s in {elapsed:.2f}s (RTF={elapsed / dur:.2f})")
    finally:
        _restore_inference_params(model, orig)
    return audio


def generate_stream(
    model,
    text: str,
    speaker_idx: int = 1,
    use_cfg: bool = True,
    crossfade_ms: float = 50,
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    cfg_scale: Optional[float] = None,
) -> Generator[torch.Tensor, None, None]:
    """Stream audio per-chunk with baked speaker and cross-fade at boundaries.

    Yields (1, N) float32 tensors at 22050 Hz.
    """
    chunked_tokens, chunked_tokens_len, _ = _chunk_text(model, text)
    num_chunks = len(chunked_tokens)
    print(f"Streaming {num_chunks} chunk(s)...")

    orig = _apply_inference_params(model, temperature, topk, cfg_scale)
    crossfade_n = int(SAMPLE_RATE * crossfade_ms / 1000)
    prev_tail: Optional[torch.Tensor] = None

    try:
        chunk_state = model.create_chunk_state(batch_size=1)

        for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
            t0 = time.time()

            batch = {
                "text": toks.unsqueeze(0).cuda(),
                "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
                "speaker_indices": speaker_idx,
            }

            with torch.no_grad():
                output = model.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=[i == num_chunks - 1],
                    beginning_of_text=(i == 0),
                    use_cfg=use_cfg,
                    use_local_transformer_for_inference=True,
                )

            if output.predicted_codes_lens[0] <= 0:
                continue

            chunk_audio = _decode_single_codes(
                model,
                output.predicted_codes[0, :, : output.predicted_codes_lens[0]],
            )

            elapsed = time.time() - t0
            dur = len(chunk_audio) / SAMPLE_RATE
            print(
                f"  Chunk {i + 1}/{num_chunks}: "
                f"{dur:.2f}s audio in {elapsed:.2f}s (RTF={elapsed / dur:.2f})"
            )

            is_last = i == num_chunks - 1

            if prev_tail is not None:
                joined = _crossfade(prev_tail, chunk_audio, crossfade_n)
                if not is_last and len(joined) > crossfade_n:
                    yield joined[:-crossfade_n].unsqueeze(0)
                    prev_tail = joined[-crossfade_n:]
                else:
                    yield joined.unsqueeze(0)
                    prev_tail = None
            else:
                if not is_last and len(chunk_audio) > crossfade_n:
                    yield chunk_audio[:-crossfade_n].unsqueeze(0)
                    prev_tail = chunk_audio[-crossfade_n:]
                else:
                    yield chunk_audio.unsqueeze(0)
                    prev_tail = None

        if prev_tail is not None and len(prev_tail) > 0:
            yield prev_tail.unsqueeze(0)

    finally:
        _restore_inference_params(model, orig)


# ---------------------------------------------------------------------------
# Real-time playback
# ---------------------------------------------------------------------------

class AudioPlayer:
    """Stream audio chunks to speakers via sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "Real-time playback requires sounddevice: pip install sounddevice"
            )
        self._sd = sd
        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=0,
            )
            self._stream.start()
        except sd.PortAudioError:
            raise RuntimeError(
                "No audio output device found (headless server?). "
                "Drop --play and use --stream to save to file instead."
            )

    def play(self, chunk: torch.Tensor):
        """Play a (1, N) or (N,) tensor immediately."""
        pcm = chunk.squeeze().cpu().numpy().astype(np.float32)
        self._stream.write(pcm[:, np.newaxis])

    def close(self):
        self._stream.stop()
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voice-consistent MagPIE TTS generation"
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=int, default=1, help="Baked speaker (0-4)")
    parser.add_argument(
        "--model", type=str, default="NMikka/Magpie-TTS-Geo-357m",
        help="HuggingFace repo, .nemo path, or .ckpt path",
    )
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--no-cfg", action="store_true", help="Disable CFG (~2x faster)")
    parser.add_argument("--stream", action="store_true", help="Per-chunk streaming decode with cross-fade")
    parser.add_argument("--play", action="store_true", help="Play audio through speakers in real-time")
    parser.add_argument("--crossfade-ms", type=float, default=50, help="Cross-fade duration (ms)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--topk", type=int, default=None, help="Top-k sampling (default: 80)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale (default: 2.5)")
    args = parser.parse_args()

    if args.play:
        args.stream = True

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    use_cfg = not args.no_cfg

    print(f"Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    mode_str = "stream+play" if args.play else ("stream" if args.stream else "consistent")
    print(f"Speaker: {args.speaker}  CFG: {use_cfg}  Mode: {mode_str}")
    overrides = {k: v for k, v in [("temperature", args.temperature), ("topk", args.topk), ("cfg_scale", args.cfg_scale)] if v is not None}
    if overrides:
        print(f"Overrides: {overrides}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    player = AudioPlayer() if args.play else None
    t_start = time.time()

    try:
        if args.stream:
            chunks = []
            for audio_chunk in generate_stream(
                model, args.text,
                speaker_idx=args.speaker,
                use_cfg=use_cfg,
                crossfade_ms=args.crossfade_ms,
                temperature=args.temperature,
                topk=args.topk,
                cfg_scale=args.cfg_scale,
            ):
                if player:
                    player.play(audio_chunk)
                chunks.append(audio_chunk.squeeze(0))

            if chunks:
                full_audio = torch.cat(chunks).unsqueeze(0)
            else:
                print("No audio generated.")
                return
        else:
            full_audio = generate_consistent(
                model, args.text,
                speaker_idx=args.speaker,
                use_cfg=use_cfg,
                temperature=args.temperature,
                topk=args.topk,
                cfg_scale=args.cfg_scale,
            )
            if full_audio is None:
                print("No audio generated.")
                return

    finally:
        if player:
            player.close()

    torchaudio.save(str(out_path), full_audio, SAMPLE_RATE)
    total = time.time() - t_start
    dur = full_audio.shape[-1] / SAMPLE_RATE
    print(f"\nSaved: {out_path} ({dur:.2f}s, total time {total:.2f}s)")


if __name__ == "__main__":
    main()
