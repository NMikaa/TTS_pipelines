"""
CSM-1B Streaming Inference.

Word-by-word generation with KV-cache for real-time-factor < 1.
Matches the 2-word look-ahead scheme used during training.

Usage:
    python -m csm_training.inference \
        --text "your text here" \
        --context-audio context.wav \
        --context-text "transcript of context audio" \
        --output output.wav

    # Or from a checkpoint:
    python -m csm_training.inference \
        --model-path checkpoints/csm_georgian/... \
        --text "your text here" \
        --output output.wav
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CsmForConditionalGeneration, StaticCache


SAMPLE_RATE = 24_000
MAX_CACHE_LEN = 4096
DEPTH_CACHE_LEN = 32


def sample_top_p(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Nucleus (top-p) sampling from logits [B, V]."""
    if temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Zero out tokens beyond the top-p threshold
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    token = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, token).squeeze(-1)


class CSMStreamingInference:
    """Streaming word-by-word inference for CSM-1B.

    Uses backbone KV-cache so each new audio frame is a single-step
    forward pass. Text injections (new words) are also cached
    incrementally.
    """

    def __init__(
        self,
        model_id: str = "sesame/csm-1b",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        if model_path:
            self.model = CsmForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            self.model = CsmForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id)

        self.model.eval()

    def _split_text_for_streaming(self, text: str) -> List[str]:
        """Split text into streaming chunks matching the 2-word look-ahead training scheme.

        Returns a list where:
        - First element: first 3 words joined
        - Subsequent elements: " " + next word (one at a time)
        - Two empty strings at the end (flush signal)
        """
        words = text.split()
        if not words:
            return [""]

        chunks = []
        chunks.append(" ".join(words[:3]))
        for w in words[3:]:
            chunks.append(" " + w)
        chunks.extend(["", ""])
        return chunks

    def _prepare_inputs(
        self,
        text_chunks: List[str],
        context_audio: np.ndarray,
        context_text: str,
        speaker_id: int = 0,
    ) -> List[dict]:
        """Build tokenized inputs for each streaming step."""
        all_inputs = []

        # First input: context audio + first text chunk
        conversation = [
            {
                "role": str(speaker_id),
                "content": [
                    {"type": "text", "text": context_text},
                    {"type": "audio", "path": context_audio},
                ],
            },
            {
                "role": str(speaker_id),
                "content": [
                    {"type": "text", "text": text_chunks[0]},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True,
        ).to(self.device)
        inputs["input_values"] = inputs["input_values"].to(self.dtype)
        all_inputs.append(inputs)

        # Subsequent inputs: just text chunks
        for chunk in text_chunks[1:]:
            conversation = [
                {"role": str(speaker_id), "content": [{"type": "text", "text": chunk}]},
            ]
            inputs = self.processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True,
            ).to(self.device)
            all_inputs.append(inputs)

        return all_inputs

    def _decode_audio_codes(self, frames: List[torch.Tensor]) -> np.ndarray:
        """Decode stacked audio code frames to waveform."""
        if not frames:
            return np.zeros(0, dtype=np.float32)

        codes = torch.cat(frames, dim=0)  # [T, 32]
        codes = codes.unsqueeze(0)  # [1, T, 32]

        # Trim at EOS if present
        eos_id = self.model.config.codebook_eos_token_id
        eos_mask = (codes == eos_id).all(dim=-1)
        eos_idxs = eos_mask.nonzero()
        if eos_idxs.numel() > 0:
            cutoff = eos_idxs.min().item()
            codes = codes[:, :cutoff]

        # Decode: codec expects [B, num_codebooks, T]
        codes_transposed = codes.squeeze(0).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            decoded = self.model.codec_model.decode(codes_transposed)

        audio = decoded.audio_values[0, 0].float().cpu().numpy()
        return audio

    @torch.no_grad()
    def generate(
        self,
        text: str,
        context_audio: np.ndarray,
        context_text: str,
        speaker_id: int = 0,
        max_frames_per_word: int = 62,
        temperature: float = 0.1,
        top_p: float = 0.999,
        decoder_temperature: float = 0.1,
        decoder_top_p: float = 0.999,
    ) -> np.ndarray:
        """Generate audio for the given text using streaming word-by-word decoding.

        Args:
            text: Text to synthesize.
            context_audio: Reference audio as float32 numpy array at 24kHz.
            context_text: Transcript of the context audio.
            speaker_id: Speaker ID for multi-speaker.
            max_frames_per_word: Max audio frames per word chunk before forcing text injection.
            temperature: Backbone sampling temperature.
            top_p: Backbone nucleus sampling threshold.
            decoder_temperature: Depth decoder sampling temperature.
            decoder_top_p: Depth decoder nucleus sampling threshold.

        Returns:
            Generated audio as float32 numpy array at 24kHz.
        """
        text_chunks = self._split_text_for_streaming(text)
        all_inputs = self._prepare_inputs(
            text_chunks, context_audio, context_text, speaker_id
        )

        model = self.model
        num_codebooks = model.config.num_codebooks
        eos_id = model.config.codebook_eos_token_id

        # --- Process first input (context + first text) ---
        inputs = all_inputs[0]
        text_ids = inputs["input_ids"]
        model_inputs = model.prepare_inputs_for_generation(**inputs)
        text_embeds = model_inputs["inputs_embeds"]

        attn_mask = inputs["attention_mask"].clone()
        seq_len = text_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Initialize KV caches
        bb_cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=MAX_CACHE_LEN,
            device=self.device,
            dtype=self.dtype,
        )
        dd_cache = StaticCache(
            config=model.depth_decoder.config,
            max_batch_size=1,
            max_cache_len=DEPTH_CACHE_LEN,
            device=self.device,
            dtype=self.dtype,
        )

        # Backbone forward on full context
        bb_out = model.backbone_model(
            inputs_embeds=text_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=bb_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        bb_cache = bb_out.past_key_values
        h_last = bb_out.hidden_states[-1][:, -1, :]  # [B, D]

        # First depth decode to prime the cache
        logits0 = model.lm_head(h_last.unsqueeze(1))[:, -1, :]
        c0 = sample_top_p(logits0, temperature, top_p)
        depth_prompt = F.pad(c0, (1, 0), value=0)

        dd_out = model.depth_decoder.generate(
            input_ids=depth_prompt,
            backbone_last_hidden_state=h_last.clone(),
            position_ids=pos_ids,
            temperature=decoder_temperature,
            top_p=decoder_top_p,
            max_new_tokens=num_codebooks - 1,
            logits_to_keep=1,
            use_cache=True,
            past_key_values=dd_cache,
            return_dict_in_generate=True,
        )
        dd_cache = dd_out.past_key_values

        # --- Autoregressive generation loop ---
        frames = []
        chunk_idx = 1
        cache_pos = torch.arange(0, 2, device=self.device)
        eos_token = torch.tensor([eos_id], device=self.device, dtype=c0.dtype)

        t_start = time.perf_counter()

        while chunk_idx < len(all_inputs):
            cur_step = 0
            while cur_step < max_frames_per_word:
                # Sample c0 from backbone
                logits0 = model.lm_head(h_last.unsqueeze(1))[:, -1, :]
                c0 = sample_top_p(logits0, temperature, top_p)

                # Check for EOS -> inject next text chunk
                if (c0 == eos_token).any() or cur_step == max_frames_per_word - 1:
                    chunk_idx += 1
                    if chunk_idx >= len(all_inputs):
                        break

                    # Inject new text
                    inject = all_inputs[chunk_idx]
                    inject_embeds = model.embed_text_tokens(inject["input_ids"])
                    h_last, bb_cache, seq_len, attn_mask = self._step_backbone(
                        inject_embeds, attn_mask, bb_cache, seq_len
                    )
                    break

                # Depth decode for codebooks 1..N
                depth_prompt = F.pad(c0, (1, 0), value=0)
                dd_out = model.depth_decoder.generate(
                    input_ids=depth_prompt,
                    backbone_last_hidden_state=h_last.clone(),
                    max_new_tokens=num_codebooks - 1,
                    temperature=decoder_temperature,
                    top_p=decoder_top_p,
                    cache_position=cache_pos,
                    logits_to_keep=1,
                    use_cache=True,
                    past_key_values=dd_cache,
                    return_dict_in_generate=False,
                )
                frame = dd_out[:, 1:]  # [B, 32]
                frames.append(frame)

                # Feed audio frame back into backbone
                frame_3d = frame.unsqueeze(1)  # [B, 1, 32]
                audio_embeds = model.backbone_model.embed_tokens(frame_3d)
                h_last, bb_cache, seq_len, attn_mask = self._step_backbone(
                    audio_embeds, attn_mask, bb_cache, seq_len
                )
                cur_step += 1

        gen_time = time.perf_counter() - t_start

        # Decode all frames to audio
        audio = self._decode_audio_codes(frames)
        audio_dur = len(audio) / SAMPLE_RATE

        print(f"Generated {audio_dur:.2f}s audio in {gen_time:.2f}s "
              f"(RTF: {gen_time / max(audio_dur, 0.01):.3f})")

        return audio

    def _step_backbone(
        self,
        new_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: StaticCache,
        seq_len: int,
    ):
        """Single cached backbone step for new token(s)."""
        bsz, n_new, _ = new_embeds.shape
        new_pos = torch.arange(
            seq_len, seq_len + n_new, device=self.device
        ).unsqueeze(0).expand(bsz, n_new)

        new_mask = torch.ones(
            bsz, n_new, device=self.device, dtype=attn_mask.dtype
        )
        attn_mask = torch.cat([attn_mask, new_mask], dim=1)

        out = self.model.backbone_model(
            inputs_embeds=new_embeds,
            attention_mask=attn_mask,
            position_ids=new_pos,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        h_last = out.hidden_states[-1][:, -1, :]
        return h_last, out.past_key_values, seq_len + n_new, attn_mask


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CSM-1B streaming inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--context-audio", required=True, help="Path to context audio (.wav)")
    parser.add_argument("--context-text", default=None,
                        help="Transcript of context audio (or path to .json mapping)")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--model-id", default="sesame/csm-1b")
    parser.add_argument("--model-path", default=None, help="Fine-tuned model path")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--max-frames-per-word", type=int, default=62)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.999)

    args = parser.parse_args()

    # Load context audio
    context_audio, sr = sf.read(args.context_audio, dtype="float32")
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torch.from_numpy(context_audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        context_audio = waveform.squeeze(0).numpy()

    # Load context text
    if args.context_text and Path(args.context_text).suffix == ".json":
        with open(args.context_text) as f:
            text_map = json.load(f)
        context_text = text_map.get(args.context_audio, "")
    else:
        context_text = args.context_text or ""

    # Generate
    engine = CSMStreamingInference(
        model_id=args.model_id,
        model_path=args.model_path,
    )
    audio = engine.generate(
        text=args.text,
        context_audio=context_audio,
        context_text=context_text,
        speaker_id=args.speaker_id,
        max_frames_per_word=args.max_frames_per_word,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
