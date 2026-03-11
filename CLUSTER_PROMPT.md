# Cluster Claude Code — Task Prompt

You are working on the TTS_pipelines repo — the first Georgian TTS benchmark. Read CLAUDE.md first for full context.

## Your immediate task: Implement and run the data quality pipeline

We have ~71k Common Voice Georgian WAV files on S3 (24kHz). The data is noisy. Before training any models, we need to clean it.

### Step 1: Download data from S3
```bash
cd TTS_pipelines
pip install -r requirements.txt
python -m shared.data.download --output-dir ./data
```
This downloads `audio.rar` (12.1GB) from S3 bucket `ttsopensource` (eu-central-1), extracts to `data/audio/`. Also downloads `alignment/voice_actor_manifest.json` (JSONL manifest with audio_filepath, text, source/speaker_id).

You need `aws` CLI configured and `unrar` installed.

### Step 2: Implement `shared/data/quality.py`

Create the 6-stage data quality filtering pipeline. Each stage takes a list of entries (from `prepare_dataset()`) and returns a filtered list. The pipeline should:

1. **Standardize** — Resample to 24kHz mono, normalize loudness to -23 LUFS (use `torchaudio` for resampling, `pyloudnorm` for LUFS normalization). Write cleaned audio to `data/audio_clean/`. Store at 24kHz — all current pipelines use 24kHz.

2. **DNSMOS filter** — Score each clip with Microsoft DNSMOS P.835 (ONNX model). Drop clips scoring < 3.0. The DNSMOS model can be downloaded from the Microsoft DNS Challenge repo. Use `onnxruntime` for inference. Process in batches.

3. **VAD trim** — Use Silero VAD (`torch.hub.load('snakers4/silero-vad', 'silero_vad')`) to detect speech boundaries. Trim leading/trailing silence. Drop clips where speech is < 50% of total duration (mostly silence).

4. **SNR filter** — Estimate signal-to-noise ratio from the waveform. Drop clips with SNR < 15 dB. Simple approach: segment into voiced (signal) and unvoiced (noise) frames using energy thresholding.

5. **Transcript verification** — Use Meta Omnilingual ASR 7B (`pip install omnilingual-asr`, model `omniASR_LLM_7B`, language `kat_Geor`) to transcribe each clip. Compare to original text via CER (character error rate). Drop clips with CER > 0.20. This catches wrong transcripts, code-switching, and garbage audio. WARNING: Do NOT use Whisper — it's catastrophically bad for Georgian (78-88% WER).

6. **Speaker selection** — Use ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) to extract speaker embeddings. Cluster speakers. Keep top speakers by amount of data. This ensures speaker consistency in training data.

The pipeline should:
- Be runnable as `python -m shared.data.quality --data-dir ./data --output-dir ./data/clean`
- Save a filtered manifest (`clean_manifest.json`) and cleaned audio files
- Print stats at each stage (how many clips kept/dropped, reasons)
- Be resumable — if stage 3 was already completed, skip to stage 4
- Log everything so we can review what was dropped and why

Expected output: ~71k clips -> ~50-55k clean clips (keep ~75%).

### Step 3: Run the pipeline
```bash
python -m shared.data.quality --data-dir ./data --output-dir ./data/clean
```

### After data cleaning is done
Once we have clean data, the next steps are implementing train.py and infer.py for each of the 4 models, starting with CSM-1B (confirmed working with Unsloth LoRA). But data first.

### Important notes
- Read CLAUDE.md for ALL project context including model details, evaluation metrics, and design decisions
- The shared data code already exists: `shared/data/download.py`, `shared/data/prepare.py`, `shared/data/splits.py`
- The evaluation code already exists in `shared/evaluation/`
- All 4 pipeline directories exist with scaffolds but train.py/infer.py raise NotImplementedError
- GPU: You need at least one GPU for DNSMOS (ONNX), ASR (7B model), and speaker embeddings
- GPU: 48GB VRAM available — plenty for all stages including the ASR 7B model
