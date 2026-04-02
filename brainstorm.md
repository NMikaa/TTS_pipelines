# Brainstorm: Training MagPIE TTS on 1,524h Georgian

## The Data Quality Question

This is the right instinct — let's think about what each issue actually does to the model:

**Early cutoffs** are the most dangerous problem. MagPIE uses CTC alignment to enforce monotonic text→audio mapping. When audio is truncated mid-word, CTC has two choices: (1) compress the remaining text into silence (learns to skip words), or (2) the loss spikes and that sample acts as noise. Either way, the model learns that it's sometimes okay to stop before the text is finished. At inference time, this manifests as randomly truncated outputs. **This will absolutely compound.**

**Unspoken numbers** ("123" in text, silence or different words in audio) create a similar but subtler problem. CTC alignment will try to map those characters somewhere, corrupting the attention patterns around them. The model learns garbage mappings for digit characters. At inference, digits will produce unpredictable audio — mumbling, skipping, or hallucinated syllables.

**Missing punctuation** is actually the least dangerous, but affects prosody. Punctuation tells the model when to pause (comma), change intonation (question mark), or add emphasis. Without it, the model learns a flatter, more monotone prosody. Mixed training means the model can't reliably use punctuation as a prosody signal.

## Experiment Design: Is Data Quality a Problem?

I'd propose a quick **3-way comparison** on a small scale before committing to the full 1,524h run:

| Experiment | Data | Hours | Purpose |
|-----------|------|-------|---------|
| **A — Clean baseline** | Punctuated only, filtered aggressively | ~300h | Quality ceiling |
| **B — Quantity play** | All data, minimal filtering | ~1,500h | Does more data overcome noise? |
| **C — Filtered mix** | All data, aggressive filtering | ~800-1000h | Sweet spot? |

**Filtering criteria for experiment C:**
- Drop segments where `duration < 1.5s` (likely truncated)
- Drop segments where CPS (characters per second) is outside 5-20 range (misaligned)
- Drop segments containing digits/numbers (until we add a number normalizer)
- Drop `multi_speaker=True` segments (speaker identity contamination)
- Keep both punctuated and unpunctuated but add a **punctuation restoration model** as preprocessing

Train each for ~15-20 epochs on 1 GPU, evaluate CER on a held-out FLEURS set. This tells us within a few days whether data quality or quantity wins.

## Training Strategy: Curriculum Approach

Given the data split, I'd recommend a **3-phase curriculum**:

### Phase 1: Foundation (punctuated data only, ~450h)
- Start from `nvidia/magpie_tts_multilingual_357m` pretrained
- Train on clean punctuated data only
- LR: 2e-5, batch 48, ~50 epochs
- **Why**: Establish clean prosody and Georgian phonetics on the highest-quality data

### Phase 2: Scale (all filtered data, ~800-1000h)
- Initialize from Phase 1 best checkpoint
- Add unpunctuated data (after punctuation restoration)
- Lower LR: 5e-6, batch 48, ~30 epochs
- **Why**: The model already knows good prosody; now it absorbs speaker diversity and vocabulary breadth without unlearning prosodic cues

### Phase 3: Preference alignment (DPO/GRPO)
- Generate multiple samples per text, rank by ASR CER + speaker similarity
- GRPO with `num_generations_per_item=12`
- LR: 1e-7, disable dropout, disable CTC loss
- **Why**: NeMo already supports this (`+mode="onlinepo_train"`). The Koel-TTS paper shows GRPO reduces CER by ~40% and improves naturalness. This is the quality ceiling push.

## Voice Cloning: Three Approaches

### Approach A: Expanded Baked Embeddings (Simplest)
Increase `nn.Embedding(5, T*D)` -> `nn.Embedding(41, T*D)`. Each speaker gets a dedicated learned embedding. Fast, high quality for known speakers, zero generalization to new voices.

**Verdict**: Good for a production system serving these 41 speakers. Not useful for arbitrary voice cloning.

### Approach B: Context Encoder Training (Current Approach, Improved)
The existing `train_cloning.py` freezes everything except context encoder + cross-attention (~22.8M params). With 41 speakers and 1,524h, we can do much better:

- **Don't freeze the decoder** — with this much data, unfreezing more params won't overfit
- **Longer context windows** — current 3-8s, push to 5-15s for richer speaker representation
- **Multi-reference** — at inference, encode 3-5 reference clips and average embeddings for more robust speaker capture
- **Speaker-contrastive loss** — add auxiliary loss that pulls same-speaker embeddings together and pushes different-speaker embeddings apart

**Verdict**: Best balance of quality and zero-shot capability.

### Approach C: Joint Training (Most Ambitious, My Recommendation)

Don't train baked speakers first and then separately train cloning. Instead, do **joint training from the start**:

1. Every training sample comes with a context audio clip (different clip, same speaker)
2. The context encoder is always active — no baked embeddings at all
3. The model learns Georgian AND voice cloning simultaneously
4. With 41 speakers across 1,524h, there's massive diversity for the context encoder to learn from

**Training recipe:**
- Phase 1: Unfreeze context encoder + decoder cross-attention only (current freeze strategy), LR 5e-5, 30 epochs — lets the context encoder catch up to the pretrained decoder
- Phase 2: Unfreeze everything, LR 1e-5, 50 epochs — joint fine-tuning where all components adapt together
- Phase 3: GRPO preference alignment with speaker similarity as a reward signal

**Why this is better**: The current 2-step approach (baked -> cloning) means the decoder adapts to baked embeddings first, then has to re-learn to work with context encoder outputs. Joint training means the decoder and context encoder co-adapt from the start.

## Quality Improvement Ideas

### 1. Number Normalization (Critical)
Before any training, convert all numbers to Georgian words:
- "123" -> "ას ოცდასამი"
- "2026" -> "ორი ათას ოცდაექვსი"
- Build a Georgian `num2words` converter or use an LLM to batch-process

### 2. Punctuation Restoration
Run a Georgian punctuation model on the 1,000h unpunctuated data. Options:
- Fine-tune a multilingual model (mBERT, XLM-R) on the 450h punctuated subset
- Use an LLM (Gemma, Qwen) with few-shot prompting
- This unifies the data quality and gives the model consistent prosodic cues

### 3. Data Deduplication
Audiobook data often has repeated phrases (chapter headers, common expressions). Near-duplicate text segments will cause the model to memorize specific prosody patterns. Deduplicate on text similarity.

### 4. Speaker Balancing
Top speaker has 330h, smallest speakers have <5h. Options:
- Upsample small speakers 2-3x (with different context pairings each time)
- Downsample the top 3 speakers to ~100h each
- Use dynamic sampling weights per speaker during training

### 5. Codec Fine-tuning
NanoCodec was trained on 28.7k hours across 105 languages but may not have much Georgian. Fine-tuning the codec on Georgian audio could improve reconstruction quality at the bottleneck. This is high-effort but high-reward if the codec is the quality ceiling.

## My Recommended Plan

```
1. FILTER & PREPARE DATA
   |- Number normalization (all data)
   |- Punctuation restoration (1000h unpunctuated)
   |- Quality filtering (CPS, duration, Georgian char ratio)
   |- Drop multi_speaker=True segments
   |- Speaker balancing

2. QUICK EXPERIMENT (1 week)
   |- 3-way comparison: clean-only vs all vs filtered
   |- Determine if filtering matters

3. JOINT TRAINING (Approach C)
   |- Phase 1: Context encoder warmup (30 epochs, frozen decoder)
   |- Phase 2: Full fine-tune (50 epochs, everything unfrozen)
   |- Phase 3: GRPO preference alignment

4. EVALUATION
   |- CER/WER via round-trip ASR
   |- Speaker similarity (cosine on speaker embeddings)
   |- MOS (mean opinion score) if possible
```
