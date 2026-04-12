"""OmniVoice Georgian — Gradio inference UI.

Supports voice cloning and voice design from fine-tuned checkpoints.

Usage:
    source venv_omnivoice/bin/activate
    python pipelines/omnivoice/app.py --port 7860
"""

import argparse
import glob
import os
import sys
import torch
import gradio as gr

# Patch torchaudio.load/save to use soundfile (torch 2.11 removed all backends except torchcodec)
import torchaudio
import soundfile as sf
import numpy as np

_orig_load = torchaudio.load
def _patched_load(filepath, *args, **kwargs):
    try:
        return _orig_load(filepath, *args, **kwargs)
    except (ImportError, RuntimeError, OSError):
        data, sr = sf.read(str(filepath), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        return waveform, sr
torchaudio.load = _patched_load

_orig_save = torchaudio.save
def _patched_save(filepath, waveform, sample_rate, *args, **kwargs):
    try:
        return _orig_save(filepath, waveform, sample_rate, *args, **kwargs)
    except (ImportError, RuntimeError, OSError):
        data = waveform.cpu().numpy()
        if data.ndim == 2:
            data = data.T
        sf.write(str(filepath), data, sample_rate)
torchaudio.save = _patched_save

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "OmniVoice"))

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

SAMPLING_RATE = 24000
EXP_DIR = os.path.join(os.path.dirname(__file__), "exp")

# Global model reference
current_model = {"model": None, "path": None}


def find_checkpoints():
    """Find all available checkpoints."""
    ckpts = []
    # Pretrained
    ckpts.append(("k2-fsa/OmniVoice (pretrained)", "k2-fsa/OmniVoice"))
    # Fine-tuned
    for subset_dir in sorted(glob.glob(os.path.join(EXP_DIR, "*"))):
        if not os.path.isdir(subset_dir):
            continue
        tag = os.path.basename(subset_dir)
        for ckpt_dir in sorted(glob.glob(os.path.join(subset_dir, "checkpoint-*"))):
            step = os.path.basename(ckpt_dir).replace("checkpoint-", "")
            if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
                label = f"{tag}/checkpoint-{step}"
                ckpts.append((label, ckpt_dir))
    return ckpts


def load_model(checkpoint_path, device="cuda:0"):
    """Load or switch model checkpoint."""
    if current_model["path"] == checkpoint_path and current_model["model"] is not None:
        return current_model["model"]

    print(f"Loading model from {checkpoint_path}...")
    model = OmniVoice.from_pretrained(
        checkpoint_path,
        device_map=device,
        dtype=torch.float16,
        load_asr=True,
    )
    current_model["model"] = model
    current_model["path"] = checkpoint_path
    print(f"Model loaded: {checkpoint_path}")
    return model


def generate_voice_clone(
    text, ref_audio, ref_text, instruct, checkpoint_choice,
    speed, num_steps, guidance_scale, denoise,
):
    """Voice cloning: generate speech matching reference speaker, with optional style instruct."""
    if not text.strip():
        return None, "Please enter text."
    if ref_audio is None:
        return None, "Please upload reference audio."

    ckpts = find_checkpoints()
    ckpt_path = dict(ckpts).get(checkpoint_choice, "k2-fsa/OmniVoice")

    try:
        model = load_model(ckpt_path)
    except Exception as e:
        return None, f"Failed to load model: {e}"

    config = OmniVoiceGenerationConfig(
        num_step=int(num_steps),
        guidance_scale=guidance_scale,
        denoise=denoise,
    )

    ref_text_val = ref_text.strip() if ref_text and ref_text.strip() else None
    instruct_val = instruct.strip() if instruct and instruct.strip() else None

    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text_val,
        )
        kw = dict(
            text=text,
            language="Georgian",
            voice_clone_prompt=prompt,
            speed=speed,
            generation_config=config,
        )
        if instruct_val:
            kw["instruct"] = instruct_val

        result = model.generate(**kw)
        wav = result[0].squeeze().cpu().numpy()
        desc = f"Generated {len(wav)/SAMPLING_RATE:.1f}s audio"
        if instruct_val:
            desc += f" (instruct: {instruct_val})"
        return (SAMPLING_RATE, wav), desc
    except Exception as e:
        return None, f"Generation failed: {e}"


def generate_voice_design(
    text, checkpoint_choice, gender, age, pitch,
    speed, num_steps, guidance_scale, denoise,
):
    """Voice design: generate speech from attribute description."""
    if not text.strip():
        return None, "Please enter text."

    ckpts = find_checkpoints()
    ckpt_path = dict(ckpts).get(checkpoint_choice, "k2-fsa/OmniVoice")

    try:
        model = load_model(ckpt_path)
    except Exception as e:
        return None, f"Failed to load model: {e}"

    config = OmniVoiceGenerationConfig(
        num_step=int(num_steps),
        guidance_scale=guidance_scale,
        denoise=denoise,
    )

    # Build instruction from attributes
    parts = []
    if gender and gender != "None":
        parts.append(gender)
    if age and age != "None":
        parts.append(age)
    if pitch and pitch != "None":
        parts.append(f"{pitch} pitch")
    instruct = ", ".join(parts) if parts else None

    try:
        result = model.generate(
            text=text,
            language="Georgian",
            instruct=instruct,
            speed=speed,
            generation_config=config,
        )
        wav = result[0].squeeze().cpu().numpy()
        desc = instruct or "auto voice"
        return (SAMPLING_RATE, wav), f"Generated {len(wav)/SAMPLING_RATE:.1f}s audio ({desc})"
    except Exception as e:
        return None, f"Generation failed: {e}"


def build_ui():
    ckpts = find_checkpoints()
    ckpt_labels = [c[0] for c in ckpts]
    default_ckpt = ckpt_labels[-1] if ckpt_labels else "k2-fsa/OmniVoice (pretrained)"

    with gr.Blocks(title="OmniVoice Georgian TTS") as demo:
        gr.Markdown("# OmniVoice Georgian TTS\nFine-tuned on Georgian speech corpus")

        checkpoint_dropdown = gr.Dropdown(
            choices=ckpt_labels,
            value=default_ckpt,
            label="Checkpoint",
        )

        with gr.Tab("Voice Clone"):
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(
                        label="Text (Georgian)",
                        placeholder="შეიყვანეთ ქართული ტექსტი...",
                        lines=3,
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio (3-10 sec)",
                        type="filepath",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text (optional, auto-transcribed if empty)",
                        placeholder="რეფერენს აუდიოს ტექსტი...",
                        lines=2,
                    )
                    with gr.Accordion("Instruct (optional)", open=False):
                        vc_instruct = gr.Textbox(
                            label="Style Instruction",
                            placeholder="e.g. happy, whisper, slow, excited, sad...",
                            lines=2,
                        )

                with gr.Column():
                    vc_output = gr.Audio(label="Generated Audio")
                    vc_status = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("Generation Settings", open=False):
                vc_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                vc_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                vc_cfg = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                vc_denoise = gr.Checkbox(value=True, label="Denoise")

            vc_btn = gr.Button("Generate", variant="primary")
            vc_btn.click(
                generate_voice_clone,
                inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_instruct, checkpoint_dropdown,
                        vc_speed, vc_steps, vc_cfg, vc_denoise],
                outputs=[vc_output, vc_status],
            )

        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(
                        label="Text (Georgian)",
                        placeholder="შეიყვანეთ ქართული ტექსტი...",
                        lines=3,
                    )
                    vd_gender = gr.Dropdown(
                        choices=["None", "male", "female"],
                        value="None", label="Gender",
                    )
                    vd_age = gr.Dropdown(
                        choices=["None", "child", "teenager", "young adult", "middle-aged", "elderly"],
                        value="None", label="Age",
                    )
                    vd_pitch = gr.Dropdown(
                        choices=["None", "very low", "low", "moderate", "high", "very high"],
                        value="None", label="Pitch",
                    )

                with gr.Column():
                    vd_output = gr.Audio(label="Generated Audio")
                    vd_status = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("Generation Settings", open=False):
                vd_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                vd_steps = gr.Slider(4, 64, value=32, step=4, label="Inference Steps")
                vd_cfg = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale")
                vd_denoise = gr.Checkbox(value=True, label="Denoise")

            vd_btn = gr.Button("Generate", variant="primary")
            vd_btn.click(
                generate_voice_design,
                inputs=[vd_text, checkpoint_dropdown, vd_gender, vd_age, vd_pitch,
                        vd_speed, vd_steps, vd_cfg, vd_denoise],
                outputs=[vd_output, vd_status],
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    os.environ["OMNIVOICE_DEVICE"] = args.device

    # Pre-load latest checkpoint
    ckpts = find_checkpoints()
    if ckpts:
        load_model(ckpts[-1][1], device=args.device)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
