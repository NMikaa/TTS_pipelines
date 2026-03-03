#!/usr/bin/env python3
"""
Train a Georgian SentencePiece BPE tokenizer from the voice actor manifest.

Produces a .model file that gets added to the Pocket TTS vocabulary
(IDs 4000+) alongside the original English tokenizer.

Usage:
    python scripts/train_tokenizer.py \
        --manifest alignment/voice_actor_manifest.json \
        --output georgian_tokenizer.model \
        --vocab-size 500
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm


# Georgian Unicode range: U+10D0 (ა) to U+10FA (ჺ), main letters U+10D0-U+10F0
GEORGIAN_CHARS = [chr(c) for c in range(0x10D0, 0x10F1)]  # ა-ჰ (33 letters)


def extract_texts(manifest_path: str) -> list[str]:
    """Extract all text fields from JSONL manifest."""
    texts = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                text = entry.get("text", "").strip()
                if text:
                    texts.append(text)
    return texts


def train_spm(texts: list[str], output_path: str, vocab_size: int = 500):
    """Train SentencePiece BPE model on Georgian texts.

    Ensures all Georgian characters are included as user-defined symbols
    so they're never split into bytes.
    """
    # Write texts to temp file for SPM training
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for text in texts:
            f.write(text + "\n")
        temp_path = f.name

    try:
        # Force all Georgian characters as user_defined_symbols
        # This guarantees they get their own token IDs and are never
        # broken into bytes, even if they're rare in the corpus
        user_symbols = ",".join(GEORGIAN_CHARS)

        model_prefix = output_path.replace(".model", "")

        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,  # Cover ALL characters
            user_defined_symbols=user_symbols,
            pad_id=-1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            normalization_rule_name="identity",  # Don't normalize Georgian text
        )
        print(f"Trained SentencePiece model: {model_prefix}.model")
    finally:
        os.unlink(temp_path)


def verify_coverage(model_path: str):
    """Verify all Georgian characters are tokenized properly (not UNK/bytes)."""
    sp = spm.SentencePieceProcessor(model_file=model_path)

    print(f"\nVocabulary size: {sp.vocab_size()}")

    # Check each Georgian character
    missing = []
    for char in GEORGIAN_CHARS:
        pieces = sp.encode(char, out_type=str)
        token_ids = sp.encode(char, out_type=int)
        is_single = len(pieces) == 1 and not pieces[0].startswith("<0x")
        status = "OK" if is_single else "MULTI/BYTE"
        if not is_single:
            missing.append(char)
        print(f"  {char} (U+{ord(char):04X}) -> {pieces} [{status}]")

    if missing:
        print(f"\nWARNING: {len(missing)} chars not single tokens: {missing}")
        print("These will still work but tokenize less efficiently.")
    else:
        print(f"\nAll {len(GEORGIAN_CHARS)} Georgian characters have dedicated tokens.")

    # Test some Georgian words
    test_words = [
        "გამარჯობა",      # hello
        "როგორ ხარ",      # how are you
        "საქართველო",     # Georgia (country)
        "მადლობა",        # thank you
    ]
    print("\nTest tokenizations:")
    for word in test_words:
        ids = sp.encode(word, out_type=int)
        pieces = sp.encode(word, out_type=str)
        print(f"  \"{word}\" -> {len(ids)} tokens: {pieces}")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Train Georgian SentencePiece tokenizer")
    parser.add_argument("--manifest", required=True, help="Path to voice_actor_manifest.json")
    parser.add_argument("--output", default="georgian_tokenizer.model", help="Output model path")
    parser.add_argument("--vocab-size", type=int, default=500, help="Vocabulary size (default: 500)")
    args = parser.parse_args()

    if not Path(args.manifest).exists():
        print(f"Manifest not found: {args.manifest}")
        return 1

    print(f"Extracting texts from {args.manifest}...")
    texts = extract_texts(args.manifest)
    print(f"Found {len(texts):,} text samples")

    # Basic stats
    total_chars = sum(len(t) for t in texts)
    georgian_chars = sum(1 for t in texts for c in t if "\u10D0" <= c <= "\u10F0")
    print(f"Total characters: {total_chars:,}")
    print(f"Georgian characters: {georgian_chars:,} ({100*georgian_chars/total_chars:.1f}%)")

    print(f"\nTraining SentencePiece BPE (vocab_size={args.vocab_size})...")
    train_spm(texts, args.output, args.vocab_size)

    print("\nVerifying Georgian character coverage...")
    all_covered = verify_coverage(args.output)

    if all_covered:
        print(f"\nTokenizer ready: {args.output}")
    else:
        print(f"\nTokenizer saved but some chars need attention: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
