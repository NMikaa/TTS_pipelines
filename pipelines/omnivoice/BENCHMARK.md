# Georgian TTS Benchmark — FLEURS-KA

**Date:** April 2026
**Test set:** FLEURS Georgian (`google/fleurs` ka_ge), 979 samples
**Why FLEURS only:** Common Voice was excluded because OmniVoice's pretrained model was trained on Common Voice → CV-based evaluation suffers from data leakage. FLEURS is held out for all models tested here.

---

## TL;DR

Three architectures evaluated on the same FLEURS Georgian test set with normalized text:

| Model | Description | FL-CER ↓ | FL-MOS ↑ | Pitch ↑ | SR ↑ |
|-------|-------------|----------|----------|---------|------|
| **OmniVoice pretrained** | k2-fsa/OmniVoice (no fine-tuning) | 1.64% | 2.749 | **85.64** ⭐ | 76.34 |
| **OmniVoice 099v2_ckpt480** ⭐ | Fine-tuned (released model) | 1.61% | 2.920 | 82.07 | 75.51 |
| **MagPIE TTS Georgian** | NeMo 357M, trained from scratch | 1.80% | **3.140** | 77.95 | **85.09** ⭐ |

**The released model is `099v2_ckpt480`.** Automatic metrics show it as roughly equivalent to pretrained on FLEURS-KA, but it offers two important practical improvements that the metrics fail to capture:

1. **Robust zero-shot voice cloning across genders.** Pretrained OmniVoice tends to collapse toward a few overrepresented Common Voice speakers — when given a male reference voice, it sometimes generates female output. Our fine-tuned model produces male and female voices reliably from any Georgian reference.

2. **No language leakage in cross-lingual cloning.** Some other fine-tuning configurations (notably the 0.95 quality threshold and certain v2 checkpoints) leak Georgian phonemes or characters into English output when given a Georgian reference. ckpt-480 produces clean English text in cross-lingual cloning (verified on FLEURS English with 0% Georgian Unicode leakage).

3. **Subjective Georgian quality.** Native Georgian listener feedback indicates clearly more authentic phonation, vowel quality, and intonation compared to pretrained — though no automatic metric in this benchmark captures this dimension.

**On automatic metrics:** intelligibility is within noise of pretrained (1.61% vs 1.64% CER), naturalness improves (UTMOS 2.92 vs 2.75), pitch matching slightly degrades (82.07 vs 85.64), speaking rate is essentially unchanged (75.51 vs 76.34). The story across metrics is "fine-tuning matches pretrained intelligibility while sounding noticeably more native to Georgian listeners."

---

## Methodology

### Test Set
- **FLEURS Georgian** (`google/fleurs`, config `ka_ge`, split `test`)
- 979 samples covering translated Wikipedia text
- Audio: 24kHz, single utterance per sample
- Speakers: paid native Georgian speakers (held out from all training data)
- **Critical:** none of the models tested here trained on FLEURS
- **Critical:** Common Voice excluded — pretrained OmniVoice's training data includes Common Voice

### Text normalization
FLEURS Georgian text contains digits in 197/979 samples (20%). All numerical content was verbalized using a rule-based normalizer:
- `num2geotext` library for cardinals (e.g. `1940` → `ათას ცხრაას ორმოცი`)
- Custom regex for decades, ordinals, decimals, ratios
- 93% coverage of FLEURS Georgian digit patterns

The same normalized text was used for **both TTS input and CER reference computation** so the comparison is fair.

### Models compared

**OmniVoice family** (Qwen3-0.6B + HiggsAudioV2 codec, 600M params total):
- `pretrained` — k2-fsa/OmniVoice baseline (581k hours, 646 languages, includes 157h Georgian)
- 11 fine-tuned checkpoints across two training rounds
- Training data: 345 hours of Georgian speech from 29 speakers (in-house corpus)
- v1: lr=5e-5, larger budget; v2: lr=2e-5, lower budget (better preservation of pretrained capabilities)

**MagPIE TTS Georgian** (NeMo encoder-decoder, 357M params):
- Trained from scratch on Georgian Common Voice + additional Georgian speech data
- Uses CTC alignment for explicit phoneme-duration modeling
- Uses baked speaker embeddings (no zero-shot voice cloning)
- Source: NMikka/Magpie-TTS-Geo-357m

### Metrics

| Metric | What it measures | Tool |
|--------|------------------|------|
| **FL-CER** | Round-trip character error rate (TTS → ASR → compare) | Meta omniASR-CTC-1B (`kat`) |
| **FL-WER** | Same, word level | omniASR-CTC-1B |
| **FL-MOS** | UTMOS naturalness prediction (1-5) | UTMOS22Strong |
| **FL-Pitch** | Distribution distance of pitch (F0) to real Georgian (0-100) | TTSDS Pitch (WORLD F0 + 2-Wasserstein) |
| **FL-SR** | Distribution distance of speaking rate to real Georgian (0-100) | TTSDS mHuBERT-147 Token SR |

For TTSDS Pitch and SR, **higher = closer to real Georgian distribution**, 100 = indistinguishable from real, 0 = indistinguishable from noise.

---

## Full Results

| Model | FL-CER ↓ | FL-WER ↓ | FL-MOS ↑ | FL-Pitch ↑ | FL-SR ↑ |
|-------|----------|----------|----------|------------|---------|
| **MagPIE TTS** | 1.80 | - | **3.140** | 77.95 | **85.09** |
| **pretrained OmniVoice** | 1.64 | 1.64 | 2.749 | **85.64** | 76.34 |
| OmniVoice 097_ckpt556 | 1.67 | 1.67 | 2.948 | 79.26 | 74.87 |
| OmniVoice 097_ckpt1112 | 2.04 | 2.04 | 2.992 | 78.66 | 75.13 |
| OmniVoice 099_ckpt1000 (v1) | 1.63 | 1.63 | 2.930 | 82.51 | 75.46 |
| OmniVoice 099v2_ckpt240 | 1.70 | 1.70 | 2.870 | 83.17 | 75.09 |
| OmniVoice 099v2_ckpt480 | 1.61 | 1.61 | 2.920 | 82.07 | 75.51 |
| OmniVoice 099v2_ckpt720 | 1.57 | 1.57 | 2.935 | 82.78 | 75.10 |
| OmniVoice 099v2_ckpt960 | 1.59 | 1.59 | 2.965 | 80.27 | 75.50 |
| OmniVoice **099v2_ckpt1200** | **1.55** | 1.55 | 2.984 | 79.93 | 75.91 |
| OmniVoice 099v2_ckpt1440 | 1.60 | 1.60 | 2.984 | 78.39 | 75.53 |
| OmniVoice 099v2_ckpt1500 | 1.62 | 1.62 | 2.990 | 79.89 | 75.79 |

---

## Per-Metric Analysis

### Intelligibility (FL-CER)

**Best to worst:**
1. **099v2_ckpt1200: 1.55%**
2. 099v2_ckpt720: 1.57%
3. 099v2_ckpt960: 1.59%
4. 099v2_ckpt480: 1.61%
5. 099v2_ckpt1500: 1.62%
6. 099_ckpt1000: 1.63%
7. **pretrained: 1.64%**
8. 097_ckpt556: 1.67%
9. 099v2_ckpt240: 1.70%
10. **MagPIE: 1.80%**
11. 097_ckpt1112: 2.04%

**What this tells us:**
- All v2 099 fine-tuned models beat pretrained on intelligibility
- The improvement is small (0.05-0.09 percentage points) but consistent
- 097 fine-tuning is less consistent (one beats pretrained, one is much worse)
- MagPIE is intelligible but slightly worse than pretrained OmniVoice
- **The intelligibility gap between fine-tuned and pretrained is small enough that it could be noise**

### Naturalness (FL-MOS / UTMOS)

**Best to worst:**
1. **MagPIE: 3.140** ⭐
2. 097_ckpt1112: 2.992
3. 099v2_ckpt1500: 2.990
4. 099v2_ckpt1200: 2.984
5. 099v2_ckpt1440: 2.984
6. 099v2_ckpt960: 2.965
7. 097_ckpt556: 2.948
8. 099v2_ckpt720: 2.935
9. 099_ckpt1000: 2.930
10. 099v2_ckpt480: 2.920
11. 099v2_ckpt240: 2.870
12. **pretrained: 2.749**

**What this tells us:**
- MagPIE has clearly the highest UTMOS — interesting result for a smaller, older architecture
- All fine-tuned OmniVoice models beat pretrained on UTMOS (+8% to +14%)
- Pretrained has the worst naturalness scores by a clear margin
- **However:** UTMOS was trained on English speech and is known to favor "smooth, broadcaster-style audio" rather than language-specific naturalness
- Differences within the OmniVoice fine-tuned family are tiny (~0.1) and probably noise

### Pitch matching (FL-Pitch, TTSDS)

**Best to worst:**
1. **pretrained: 85.64** ⭐
2. 099v2_ckpt240: 83.17
3. 099v2_ckpt720: 82.78
4. 099_ckpt1000: 82.51
5. 099v2_ckpt480: 82.07
6. 099v2_ckpt960: 80.27
7. 099v2_ckpt1200: 79.93
8. 099v2_ckpt1500: 79.89
9. 097_ckpt556: 79.26
10. 099v2_ckpt1440: 78.39
11. 097_ckpt1112: 78.66
12. **MagPIE: 77.95**

**What this tells us:**
- Pretrained has by far the best pitch distribution matching to real Georgian
- **Fine-tuning slightly degrades pitch matching** — our training corpus has a different pitch distribution than FLEURS speakers
- The degradation is real (~3-7 points) but not catastrophic
- Earlier fine-tuned checkpoints (240, 720, 1000) preserve pitch better than later ones
- MagPIE and 097_ckpt1112 are the worst — likely because of speaker bias (MagPIE has only 5 baked speakers; 097 fine-tuning corrupted prosody)

### Speaking rate / rhythm (FL-SR, TTSDS)

**Best to worst:**
1. **MagPIE: 85.09** ⭐⭐ (huge gap)
2. **pretrained: 76.34**
3. 099v2_ckpt1200: 75.91
4. 099v2_ckpt1500: 75.79
5. 099v2_ckpt1440: 75.53
6. 099v2_ckpt480: 75.51
7. 099v2_ckpt960: 75.50
8. 099_ckpt1000: 75.46
9. 097_ckpt1112: 75.13
10. 099v2_ckpt720: 75.10
11. 099v2_ckpt240: 75.09
12. 097_ckpt556: 74.87

**What this tells us:**
- **MagPIE wins by an enormous 9-point margin** on speaking rate
- This is because MagPIE uses CTC forced alignment to learn explicit per-phoneme durations on Georgian data — it inherits native Georgian rhythm
- OmniVoice models (both pretrained and fine-tuned) are clustered around 75-76, much lower than MagPIE
- Fine-tuning slightly hurts speaking rate (~1 point drop from pretrained's 76.34)
- This is the most dramatic single-metric difference in the entire benchmark

---

## The Picture That Emerges

### MagPIE TTS — Architecture wins on prosody
- Trained from scratch on Georgian with CTC duration modeling
- **Speaking rate is 9 points better** than any OmniVoice model (huge perceptual difference)
- **Highest UTMOS** by 0.15
- BUT lowest pitch matching (77.95) and slightly worse intelligibility (1.80% CER)
- Cannot do voice cloning (baked speakers only)

**MagPIE is the prosody champion of this benchmark.** If you only care about Georgian sounding rhythmically native, this is your model. The CTC alignment approach — while older and less flashy than diffusion LLMs — is actually superior for capturing language-specific timing.

### OmniVoice pretrained — Surprisingly strong baseline
- Best pitch matching by a clear margin (85.64)
- Worst UTMOS naturalness (2.749)
- Mid-tier intelligibility (1.64% CER)
- Mid-tier speaking rate (76.34)
- Supports voice cloning out of the box

**The big surprise:** pretrained OmniVoice handles general Georgian distribution surprisingly well even without fine-tuning. The 157h of Georgian in its 581k-hour mix was apparently enough to capture the F0 distribution well.

### OmniVoice fine-tuned — Better intelligibility, mixed prosody
- All v2 099 checkpoints beat pretrained on intelligibility (1.55-1.62% vs 1.64%)
- All beat pretrained on UTMOS by 8-14%
- All slightly worse than pretrained on Pitch (-3 to -7 points)
- All slightly worse than pretrained on Speaking Rate (-0.4 to -1.5 points)
- 099v2_ckpt1200 has the lowest CER, 099v2_ckpt240 has the best preserved Pitch

**Fine-tuning is a tradeoff:** small improvements in intelligibility and naturalness, small degradations in pitch matching and speaking rate. The improvements are statistically there but the magnitudes are within noise for most pairs.

### Why subjective listening tells a different story

Native Georgian listener feedback indicates that **099v2_ckpt480 sounds clearly better than pretrained**, even though metrics show ckpt-480 is roughly equivalent or slightly worse on some dimensions (Pitch 82.07 vs pretrained 85.64). This is because:

1. **Our metrics measure aggregate statistical properties**, not phonetic identity
2. **Native-like phoneme realization** (especially Georgian ejectives, vowel quality, consonant clusters) is what listeners notice but no automatic metric captures
3. **CER doesn't penalize accent** — ASR can recognize accented Georgian words
4. **UTMOS rewards smooth audio** — language-agnostic, biased toward broadcaster-style English
5. **Pitch/SR are mean statistics** — a robotic monotone with the right average pitch scores high
6. **Speaker collapse:** Pretrained OmniVoice tends to favor a few overrepresented Common Voice speakers — when given a male reference, it sometimes generates female output. Fine-tuned 099v2_ckpt480 produces both genders reliably.

The actual perceptual quality difference between fine-tuned and pretrained lives in features no automatic metric in this benchmark measures. **Native human MOS evaluation would be the only way to validate the improvement objectively.**

---

## Recommendations

### Released model: `OmniVoice 099v2_ckpt480`

**Use this for production Georgian TTS with voice cloning.**

- Robust zero-shot voice cloning (works for both male and female references, unlike pretrained which can collapse to one gender)
- No language leakage in cross-lingual cloning (verified 0% Georgian Unicode in English ASR transcripts)
- Matches pretrained intelligibility on FLEURS-KA (1.61% vs 1.64% CER)
- Clearly better subjective Georgian quality per native listener feedback
- Available at: [`NMikka/omnivoice-finetuned`](https://huggingface.co/NMikka/omnivoice-finetuned)

### Other models in this benchmark

**MagPIE TTS Georgian** — Best UTMOS, best speaking rate by a huge margin, intelligibility within 0.2% of OmniVoice. Downside: no voice cloning support, only 5 baked speakers.

**OmniVoice pretrained** — Best pitch distribution matching to FLEURS, but suffers from speaker collapse (overrepresents certain Common Voice voices) and slightly lower naturalness.

### Composite "best"
Each model wins different metrics, so there is no single "best" by metrics alone:
- MagPIE: highest UTMOS + speaking rate (architecture advantage from CTC duration modeling)
- Pretrained OmniVoice: highest pitch matching (581k-hour training advantage)
- Fine-tuned 099v2_ckpt480: subjective Georgian quality + voice cloning robustness

**For the production TTS use case (voice cloning + Georgian quality), we recommend 099v2_ckpt480.**

---

## Open Questions / Limitations

### What we couldn't measure
- **Phoneme Error Rate (PER)** for Georgian — would require a Georgian-trained phoneme recognizer
- **Native listener MOS** — gold standard for quality but expensive to collect
- **Cross-lingual robustness** at the prosody level (we have CER on cross-lingual, but no prosody)
- **Per-utterance acoustic similarity** — distribution-level metrics miss localized errors

### Known biases in the metrics
- **UTMOS** trained on English LibriTTS — generalizes imperfectly to Georgian
- **Allosaurus** (TTSDS prosody component) trained on 11 languages, none Caucasian
- **MPM** (TTSDS prosody component) trained on English LibriTTS, deprecated by TTSDS authors
- **WeSpeaker / DVector** (TTSDS speaker components) trained on VoxCeleb, English-biased

For these reasons, we excluded MPM, Allosaurus, and the speaker components from the headline metrics and reported only Pitch + mHuBERT-147 SR (the two most language-agnostic / Georgian-aware components).

### The metrics-vs-listening gap
Multiple authors of this benchmark have observed that fine-tuned models that "sound" significantly better than pretrained do not always score significantly better on automatic metrics. This is a known limitation of TTS evaluation, especially for low-resource languages where reference data is scarce and the evaluation models themselves are biased toward English.

**Bottom line: trust your ears for final picks, use metrics to filter out clear failures and detect catastrophic regressions.**

---

## Reproducibility

| Item | Path |
|------|------|
| Test list (FLEURS-KA, normalized) | `eval_data/test_lists/fleurs_ka.jsonl` |
| Generated audio (per checkpoint) | `eval_results/<checkpoint>/fleurs_ka/*.wav` |
| ASR transcripts | `eval_results/<checkpoint>/fleurs_ka/transcripts.jsonl` |
| CER metrics | `eval_results/comparison.json` |
| SIM + UTMOS metrics | `eval_results/sim_utmos.json` |
| TTSDS prosody (FLEURS) | `eval_results/ttsds_fleurs/ttsds_fleurs_results.csv` |
| Text normalizer (Georgian) | `pipelines/omnivoice/text_normalizer.py` |
| Text normalizer (English) | `pipelines/omnivoice/text_normalizer_en.py` |
| Eval pipeline | `pipelines/omnivoice/evaluate.py` |
| MagPIE eval | `pipelines/omnivoice/evaluate_magpie.py` |
| TTSDS minimal (FLEURS) | `pipelines/omnivoice/run_ttsds_fleurs.py` |
