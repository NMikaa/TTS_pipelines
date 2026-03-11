"""Generate Georgian speech with fine-tuned MagPIE TTS.

Usage:
    python3 generate.py --text "გამარჯობა"
    python3 generate.py --text "გამარჯობა" --speakers 0 2 4
    python3 generate.py --text "გამარჯობა" --temperature 0.5 --cfg-scale 3.0
    python3 generate.py --text "გამარჯობა" --output-dir ./my_outputs
    python3 generate.py --text "გამარჯობა" --no-cfg          # ~2x faster, slight quality loss
    python3 generate.py --text "გამარჯობა" --fast             # fast mode: no CFG + compile
    python3 generate.py --text "გამარჯობა" --ref-audio ref.wav  # voice cloning from reference audio
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NeMo"))

SAMPLE_RATE = 22050


def get_latest_checkpoint():
    ckpt_dir = Path("/root/TTS_pipelines/NeMo/exp/magpie_tts_georgian/checkpoints/")
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def load_model(checkpoint_path=None):
    from nemo.collections.tts.models import MagpieTTSModel
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

    ckpt_path = checkpoint_path or get_latest_checkpoint()
    print(f"Checkpoint: {Path(ckpt_path).name}")

    model = MagpieTTSModel.from_pretrained('nvidia/magpie_tts_multilingual_357m', map_location='cpu')

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Use training config's tokenizer (correct offsets)
    training_cfg = ckpt['hyper_parameters']['cfg']
    model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)

    model = model.eval().cuda()
    return model


def compile_model(model):
    """torch.compile the decoder for faster autoregressive loop."""
    print("Compiling model (first run will be slower)...")
    model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
    return model


def load_reference_audio(ref_path):
    """Load and prepare reference audio for voice cloning."""
    waveform, sr = torchaudio.load(ref_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def generate(model, text, speaker_idx=0, ref_audio_path=None, ref_text=None, **kwargs):
    """Generate speech using generate_speech (chunked inference, no repetition bug).

    Args:
        model: Loaded MagpieTTSModel
        text: Georgian text string
        speaker_idx: Baked speaker index (0-4), ignored if ref_audio_path is set
        ref_audio_path: Path to reference audio for voice cloning (bypasses baked speakers)
        **kwargs: Override inference parameters:
            - temperature (float): Sampling temperature. Lower = more deterministic. Default 0.6
            - topk (int): Top-k sampling. Lower = more focused. Default 80
            - cfg_scale (float): Classifier-free guidance scale. Higher = stronger conditioning. Default 2.5
            - max_decoder_steps (int): Max generation length in frames. Default 500
    """
    from nemo.collections.tts.parts.utils.tts_dataset_utils import chunk_text_for_inference

    # Apply any inference parameter overrides
    ip = model.inference_parameters
    orig = {}
    for key, val in kwargs.items():
        if hasattr(ip, key) and val is not None:
            orig[key] = getattr(ip, key)
            setattr(ip, key, val)

    # If using reference audio, temporarily disable baked embedding
    baked_disabled = False
    if ref_audio_path:
        ref_waveform = load_reference_audio(ref_audio_path).cuda().squeeze(0)  # (time,) — no channel dim
        ref_len = torch.tensor([ref_waveform.shape[-1]], device='cuda', dtype=torch.long)
        # Disable baked embedding so model takes the context audio path
        model._orig_baked_embedding = model.baked_context_embedding
        model.baked_context_embedding = None
        baked_disabled = True

    chunked_tokens, chunked_tokens_len, _ = chunk_text_for_inference(
        text=text,
        language="ka",
        tokenizer_name="text_ce_tokenizer",
        text_tokenizer=model.tokenizer,
        eos_token_id=model.eos_id,
    )

    start = time.time()
    with torch.no_grad():
        chunk_state = model.create_chunk_state(batch_size=1)
        all_codes = []
        num_chunks = len(chunked_tokens)

        for chunk_idx, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
            batch = {
                'text': toks.unsqueeze(0).cuda(),
                'text_lens': torch.tensor([toks_len], device='cuda', dtype=torch.long),
            }
            if ref_audio_path:
                batch['context_audio'] = ref_waveform.unsqueeze(0)  # (1, time)
                batch['context_audio_lens'] = ref_len
                batch['context_sample_rate'] = SAMPLE_RATE
                # Text conditioning for context
                if ref_text:
                    ctx_ids = model.tokenizer.encode(ref_text, tokenizer_name="text_ce_tokenizer")
                    ctx_tokens = torch.tensor([ctx_ids], dtype=torch.long, device='cuda')
                    batch['context_text_tokens'] = ctx_tokens
                    batch['context_text_tokens_lens'] = torch.tensor([len(ctx_ids)], device='cuda', dtype=torch.long)
                    batch['has_text_context'] = torch.tensor([True], device='cuda')
                else:
                    batch['context_text_tokens'] = torch.zeros(1, 1, dtype=torch.long, device='cuda')
                    batch['context_text_tokens_lens'] = torch.zeros(1, dtype=torch.long, device='cuda')
                    batch['has_text_context'] = torch.tensor([False], device='cuda')
            else:
                batch['speaker_indices'] = speaker_idx

            use_cfg = kwargs.get('use_cfg', True)
            output = model.generate_speech(
                batch,
                chunk_state=chunk_state,
                end_of_text=[chunk_idx == num_chunks - 1],
                beginning_of_text=(chunk_idx == 0),
                use_cfg=use_cfg,
                use_local_transformer_for_inference=True,
            )
            if output.predicted_codes_lens[0] > 0:
                all_codes.append(output.predicted_codes[0, :, :output.predicted_codes_lens[0]])

    # Restore baked embedding and parameters
    if baked_disabled:
        model.baked_context_embedding = model._orig_baked_embedding
        del model._orig_baked_embedding
    for key, val in orig.items():
        setattr(ip, key, val)

    if all_codes:
        concat_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
        codes_lens = torch.tensor([concat_codes.shape[2]], device='cuda', dtype=torch.long)
        audio, audio_lens, _ = model.codes_to_audio(concat_codes, codes_lens)
        audio_out = audio[0, :audio_lens[0]].cpu().float().unsqueeze(0)
        elapsed = time.time() - start
        duration = audio_out.shape[-1] / SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0
        return audio_out, duration, rtf

    return None, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Generate Georgian speech with MagPIE TTS")
    parser.add_argument("--text", type=str, required=True, help="Georgian text to synthesize")
    parser.add_argument("--speakers", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Speaker indices (0-4)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt file (default: latest)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--topk", type=int, default=None, help="Top-k sampling (default: 80)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale (default: 2.5)")
    parser.add_argument("--max-decoder-steps", type=int, default=None, help="Max decoder steps (default: 500)")
    parser.add_argument("--no-cfg", action="store_true", help="Disable CFG (~2x faster, slight quality loss)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: no CFG + torch.compile")
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio for voice cloning (bypasses baked speakers)")
    parser.add_argument("--ref-text", type=str, default=None, help="Transcript of reference audio (improves voice cloning)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    if args.fast:
        model = compile_model(model)
    print(f"\nText: {args.text}")

    overrides = {}
    if args.temperature is not None: overrides['temperature'] = args.temperature
    if args.topk is not None: overrides['topk'] = args.topk
    if args.cfg_scale is not None: overrides['cfg_scale'] = args.cfg_scale
    if args.max_decoder_steps is not None: overrides['max_decoder_steps'] = args.max_decoder_steps
    if args.no_cfg or args.fast: overrides['use_cfg'] = False
    if overrides:
        print(f"Overrides: {overrides}")

    if args.ref_audio:
        print(f"Voice cloning from: {args.ref_audio}")
        audio, duration, rtf = generate(model, args.text, ref_audio_path=args.ref_audio, ref_text=args.ref_text, **overrides)
        if audio is not None:
            path = out_dir / "cloned.wav"
            torchaudio.save(str(path), audio, SAMPLE_RATE)
            print(f"  cloned: {path} ({duration:.2f}s, RTF={rtf:.2f})")
        else:
            print("  no audio generated")
    else:
        print(f"Speakers: {args.speakers}")
        for spk in args.speakers:
            audio, duration, rtf = generate(model, args.text, speaker_idx=spk, **overrides)
            if audio is not None:
                path = out_dir / f"spk{spk}.wav"
                torchaudio.save(str(path), audio, SAMPLE_RATE)
                print(f"  spk{spk}: {path} ({duration:.2f}s, RTF={rtf:.2f})")
            else:
                print(f"  spk{spk}: no audio generated")

    print("\nDone!")


if __name__ == "__main__":
    main()
