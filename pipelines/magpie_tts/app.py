"""Gradio app for MagPIE TTS Georgian inference."""

import sys
import glob
import re
import time
from pathlib import Path

import torch
import torchaudio
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NeMo"))

SAMPLE_RATE = 22050
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "exp" / "magpie_georgian" / "checkpoints"

# Speaker map from training
SPEAKER_MAP = {
    "გიორგი ზანგური (5) - 108h": 5,
    "თეკო ჩუბინიძე (6) - 66h": 6,
    "მიხეილ რაზმაძე (7) - 42h": 7,
    "სოსო ხვედელიძე (8) - 61h": 8,
    "ლალი ვეზირიშვილი (9) - 28h": 9,
    "ნიკოლოზ წივილაძე (10) - 17h": 10,
    "ნუგზარ ყურაშვილი (11) - 22h": 11,
    "ნინო მითაიშვილი (12) - 17h": 12,
    "მერაბ მეტრეველი (13) - 14h": 13,
    "ანა მატუაშვილი (14) - 10h": 14,
    "გაგა შიშინაშვილი (15) - 5.9h": 15,
    "ანა ზამბახიძე (16) - 5.6h": 16,
    "მაია კოკოჩაშვილი (17) - 5.5h": 17,
    "ნინო წეროძე (18) - 9.3h": 18,
    "სანდრო ლელუაშვილი (19) - 3.4h": 19,
    "ზვიად დოლიძე (20) - 3.3h": 20,
    "გიორგი მეგრელიშვილი (21) - 1.6h": 21,
    "კახაბერ ღირსიაშვილი (22) - 1.7h": 22,
    "მარიამ შიშნიაშვილი (23) - 2.8h": 23,
    "ნინო თარხან-მოურავი (24) - 1.3h": 24,
    "ვარლამ კორშია (25) - 1.3h": 25,
    "ზურა გორგაძე (26) - 0.8h": 26,
    "იოსებ მოლოდინაშვილი (27) - 0.1h": 27,
    "ეროსი მანჯგალაძე (28) - 0.1h": 28,
    "Pretrained Speaker 0": 0,
    "Pretrained Speaker 1": 1,
    "Pretrained Speaker 2": 2,
    "Pretrained Speaker 3": 3,
    "Pretrained Speaker 4": 4,
}

MODEL = None
CURRENT_CKPT = None


def get_checkpoints():
    """List available checkpoints sorted by val_loss."""
    ckpts = sorted(CHECKPOINT_DIR.glob("epoch=*.ckpt"))
    ckpts = [c for c in ckpts if "-last" not in c.name]
    return [c.name for c in ckpts]


def load_model(ckpt_name):
    """Load or reload model from checkpoint."""
    global MODEL, CURRENT_CKPT

    if MODEL is not None and CURRENT_CKPT == ckpt_name:
        return f"Model already loaded: {ckpt_name}"

    from nemo.collections.tts.models import MagpieTTSModel
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

    ckpt_path = CHECKPOINT_DIR / ckpt_name

    # Load pretrained base
    model = MagpieTTSModel.from_pretrained(
        "nvidia/magpie_tts_multilingual_357m", map_location="cpu"
    )

    # Expand baked embedding to 29 speakers before loading fine-tuned weights
    from omegaconf import open_dict
    with open_dict(model.cfg):
        model.cfg.num_speakers = 29

    # Load fine-tuned weights (triggers embedding expansion in load_state_dict)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model.load_state_dict(state_dict, strict=False)

    # Fix tokenizer from training config
    training_cfg = ckpt["hyper_parameters"]["cfg"]
    model.tokenizer = setup_tokenizers(training_cfg.text_tokenizers)

    model = model.eval().cuda()
    del ckpt

    MODEL = model
    CURRENT_CKPT = ckpt_name
    return f"Loaded: {ckpt_name}"


def generate_speech(text, speaker_name, temperature, cfg_scale, use_cfg):
    """Generate speech and return audio."""
    global MODEL

    if MODEL is None:
        return None, "Load a model first!"

    pass  # placeholder

    if not text.strip():
        return None, "Please enter some text."

    speaker_idx = SPEAKER_MAP.get(speaker_name, 0)

    # Set inference parameters
    ip = MODEL.inference_parameters
    orig_temp = ip.temperature
    orig_cfg = ip.cfg_scale
    ip.temperature = temperature
    ip.cfg_scale = cfg_scale

    try:
        # Split by sentences for consistent voice across chunks
        # Each sentence is a separate chunk — short enough for the decoder to handle
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]

        chunked_tokens = []
        chunked_tokens_len = []
        for sent in sentences:
            tokens = MODEL.tokenizer.encode(text=sent, tokenizer_name="text_ce_tokenizer")
            tokens = tokens + [MODEL.eos_id]
            tokens_tensor = torch.tensor(tokens, dtype=torch.int32)
            chunked_tokens.append(tokens_tensor)
            chunked_tokens_len.append(tokens_tensor.shape[0])

        start = time.time()
        with torch.no_grad():
            chunk_state = MODEL.create_chunk_state(batch_size=1)
            all_codes = []

            for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
                batch = {
                    "text": toks.unsqueeze(0).cuda(),
                    "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
                    "speaker_indices": speaker_idx,
                }
                output = MODEL.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=[i == len(chunked_tokens) - 1],
                    beginning_of_text=(i == 0),
                    use_cfg=use_cfg,
                    use_local_transformer_for_inference=True,
                )
                if output.predicted_codes_lens[0] > 0:
                    all_codes.append(
                        output.predicted_codes[0, :, : output.predicted_codes_lens[0]]
                    )

        if not all_codes:
            return None, "No audio generated."

        # Concatenate all codes and decode once (NVIDIA's intended approach)
        concat_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
        codes_lens = torch.tensor([concat_codes.shape[2]], device="cuda", dtype=torch.long)
        decode_out = MODEL._codec_model.decode(tokens=concat_codes, tokens_len=codes_lens)
        audio = decode_out[0] if isinstance(decode_out, tuple) else decode_out
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        waveform = audio[0].cpu().float().numpy()

        elapsed = time.time() - start
        duration = len(waveform) / SAMPLE_RATE
        info = f"Duration: {duration:.1f}s | Generation time: {elapsed:.1f}s | RTF: {elapsed/duration:.2f}"

        return (SAMPLE_RATE, waveform), info

    except Exception as e:
        return None, f"Error: {e}"
    finally:
        ip.temperature = orig_temp
        ip.cfg_scale = orig_cfg


def generate_cloned_speech(text, ref_audio, temperature, cfg_scale, use_cfg):
    """Generate speech cloning the voice from reference audio."""
    global MODEL

    if MODEL is None:
        return None, "Load a model first!"

    if ref_audio is None:
        return None, "Please upload reference audio."

    if not text.strip():
        return None, "Please enter some text."

    # Set inference parameters
    ip = MODEL.inference_parameters
    orig_temp = ip.temperature
    orig_cfg = ip.cfg_scale
    ip.temperature = temperature
    ip.cfg_scale = cfg_scale

    try:
        # Load reference audio
        sr, ref_wav = ref_audio
        ref_waveform = torch.tensor(ref_wav, dtype=torch.float32)
        if ref_waveform.dim() == 1:
            ref_waveform = ref_waveform.unsqueeze(0)
        if ref_waveform.shape[0] > 1:
            ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
        # Normalize to [-1, 1]
        if ref_waveform.abs().max() > 1.0:
            ref_waveform = ref_waveform / ref_waveform.abs().max()
        # Resample if needed
        if sr != SAMPLE_RATE:
            ref_waveform = torchaudio.functional.resample(ref_waveform, sr, SAMPLE_RATE)
        ref_waveform = ref_waveform.squeeze(0).cuda()  # (time,)
        ref_len = torch.tensor([ref_waveform.shape[0]], device="cuda", dtype=torch.long)

        # Temporarily disable baked embedding for voice cloning
        orig_baked = MODEL.baked_context_embedding
        MODEL.baked_context_embedding = None

        # Tokenize text (single chunk for consistency)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]

        chunked_tokens = []
        chunked_tokens_len = []
        for sent in sentences:
            tokens = MODEL.tokenizer.encode(text=sent, tokenizer_name="text_ce_tokenizer")
            tokens = tokens + [MODEL.eos_id]
            tokens_tensor = torch.tensor(tokens, dtype=torch.int32)
            chunked_tokens.append(tokens_tensor)
            chunked_tokens_len.append(tokens_tensor.shape[0])

        start = time.time()
        with torch.no_grad():
            chunk_state = MODEL.create_chunk_state(batch_size=1)
            all_codes = []

            for i, (toks, toks_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
                batch = {
                    "text": toks.unsqueeze(0).cuda(),
                    "text_lens": torch.tensor([toks_len], device="cuda", dtype=torch.long),
                    "context_audio": ref_waveform.unsqueeze(0),
                    "context_audio_lens": ref_len,
                    "context_sample_rate": SAMPLE_RATE,
                    "context_text_tokens": torch.zeros(1, 1, dtype=torch.long, device="cuda"),
                    "context_text_tokens_lens": torch.zeros(1, dtype=torch.long, device="cuda"),
                    "has_text_context": torch.tensor([False], device="cuda"),
                }
                output = MODEL.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=[i == len(chunked_tokens) - 1],
                    beginning_of_text=(i == 0),
                    use_cfg=use_cfg,
                    use_local_transformer_for_inference=True,
                )
                if output.predicted_codes_lens[0] > 0:
                    all_codes.append(
                        output.predicted_codes[0, :, : output.predicted_codes_lens[0]]
                    )

        # Restore baked embedding
        MODEL.baked_context_embedding = orig_baked

        if not all_codes:
            return None, "No audio generated."

        concat_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
        codes_lens = torch.tensor([concat_codes.shape[2]], device="cuda", dtype=torch.long)
        decode_out = MODEL._codec_model.decode(tokens=concat_codes, tokens_len=codes_lens)
        audio = decode_out[0] if isinstance(decode_out, tuple) else decode_out
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        waveform = audio[0].cpu().float().numpy()

        elapsed = time.time() - start
        duration = len(waveform) / SAMPLE_RATE
        info = f"Duration: {duration:.1f}s | Generation time: {elapsed:.1f}s | RTF: {elapsed/duration:.2f}"

        return (SAMPLE_RATE, waveform), info

    except Exception as e:
        # Restore baked embedding on error
        MODEL.baked_context_embedding = orig_baked
        return None, f"Error: {e}"
    finally:
        ip.temperature = orig_temp
        ip.cfg_scale = orig_cfg


# Build Gradio UI
with gr.Blocks(title="MagPIE TTS Georgian") as demo:
    gr.Markdown("# MagPIE TTS - Georgian Speech Synthesis")
    gr.Markdown("Fine-tuned on Georgian speech data")

    with gr.Row():
        with gr.Column(scale=1):
            ckpt_dropdown = gr.Dropdown(
                choices=get_checkpoints(),
                label="Checkpoint",
                value=get_checkpoints()[-1] if get_checkpoints() else None,
            )
            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            refresh_btn = gr.Button("Refresh Checkpoints")

    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("Baked Speakers"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Georgian Text",
                        placeholder="გამარჯობა, მე მქვია მაგპაი და ქართულად ვლაპარაკობ.",
                        lines=3,
                        value="გამარჯობა, მე მქვია მაგპაი და ქართულად ვლაპარაკობ.",
                    )
                    speaker_dropdown = gr.Dropdown(
                        choices=list(SPEAKER_MAP.keys()),
                        label="Speaker",
                        value="გიორგი ზანგური (5) - 108h",
                    )
                    with gr.Row():
                        temperature = gr.Slider(0.1, 1.5, value=0.6, step=0.05, label="Temperature")
                        cfg_scale = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="CFG Scale")
                    use_cfg = gr.Checkbox(value=True, label="Use Classifier-Free Guidance")
                    generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

                with gr.Column(scale=2):
                    audio_output = gr.Audio(label="Generated Speech", type="numpy")
                    info_output = gr.Textbox(label="Info", interactive=False)

        with gr.TabItem("Voice Cloning"):
            with gr.Row():
                with gr.Column(scale=2):
                    clone_text = gr.Textbox(
                        label="Georgian Text",
                        placeholder="გამარჯობა, მე მქვია მაგპაი და ქართულად ვლაპარაკობ.",
                        lines=3,
                        value="გამარჯობა, მე მქვია მაგპაი და ქართულად ვლაპარაკობ.",
                    )
                    ref_audio = gr.Audio(label="Reference Audio (3-10s recommended)", type="numpy")
                    with gr.Row():
                        clone_temp = gr.Slider(0.1, 1.5, value=0.6, step=0.05, label="Temperature")
                        clone_cfg = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="CFG Scale")
                    clone_use_cfg = gr.Checkbox(value=True, label="Use Classifier-Free Guidance")
                    clone_btn = gr.Button("Clone & Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    clone_audio_output = gr.Audio(label="Cloned Speech", type="numpy")
                    clone_info = gr.Textbox(label="Info", interactive=False)

    # Wire up events
    load_btn.click(load_model, inputs=[ckpt_dropdown], outputs=[load_status])
    refresh_btn.click(lambda: gr.update(choices=get_checkpoints()), outputs=[ckpt_dropdown])
    generate_btn.click(
        generate_speech,
        inputs=[text_input, speaker_dropdown, temperature, cfg_scale, use_cfg],
        outputs=[audio_output, info_output],
    )
    clone_btn.click(
        generate_cloned_speech,
        inputs=[clone_text, ref_audio, clone_temp, clone_cfg, clone_use_cfg],
        outputs=[clone_audio_output, clone_info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
