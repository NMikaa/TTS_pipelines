"""W&B callbacks for CSM-1B training."""

import random

import numpy as np
import torch
import wandb
from transformers import TrainerCallback


class WandbValSampleLogger(TrainerCallback):
    """Log reference vs generated audio samples to W&B on each evaluation.

    Picks n_examples random samples from the eval dataset. On every
    on_evaluate event, generates audio for each sample and logs a W&B
    table with columns: text, speaker_id, ref_audio, gen_audio.
    """

    def __init__(self, processor, eval_dataset, sample_rate=24000, n_examples=8,
                 text_key="text", speaker_key="speaker_id", gen_kwargs=None):
        self.processor = processor
        self.eval_ds = eval_dataset
        self.sr = int(sample_rate)
        self.text_key = text_key
        self.spk_key = speaker_key
        self.n = int(n_examples)
        self.gen_kwargs = gen_kwargs or dict(
            max_new_tokens=125 * 3, depth_decoder_do_sample=False,
            do_sample=False, output_audio=True,
        )
        rng = random.Random(42)
        self.indices = rng.sample(range(len(self.eval_ds)), min(self.n, len(self.eval_ds)))

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not wandb.run:
            return
        model.eval()
        device = next(model.parameters()).device
        rows = []
        for idx in self.indices:
            try:
                item = self.eval_ds[idx]
                ref_wav = np.asarray(item["input_values"])
                text = item[self.text_key]
                spk = item[self.spk_key]

                conversation = [
                    {"role": str(spk),
                     "content": [{"type": "text", "text": text}]},
                ]
                inputs = self.processor.apply_chat_template(
                    conversation, tokenize=True, return_dict=True,
                    return_tensors="pt",
                ).to(device)

                out = model.generate(**inputs, **self.gen_kwargs)
                gen_audio = out[0].to(torch.float32).detach().cpu().numpy()

                ref_audio_wb = wandb.Audio(ref_wav[0] if ref_wav.ndim > 1 else ref_wav,
                                           sample_rate=self.sr, caption="reference")
                gen_audio_wb = wandb.Audio(gen_audio, sample_rate=self.sr, caption="generated")
                rows.append([text, str(spk), ref_audio_wb, gen_audio_wb])
            except Exception as e:
                print(f"[WandbValSampleLogger] Failed idx={idx}: {e}")

        if rows:
            table = wandb.Table(columns=["text", "speaker_id", "ref_audio", "gen_audio"], rows=rows)
            wandb.log({"eval/samples": table, "global_step": state.global_step})
