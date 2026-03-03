"""
Combined English + Georgian tokenizer for Pocket TTS fine-tuning.

Extends the pretrained 4000-token English SentencePiece vocabulary with
a Georgian SentencePiece model. Georgian tokens are offset to IDs 4000+
so they don't collide with the original vocabulary.

Architecture:
    IDs 0-3999:     Original English SentencePiece tokens
    IDs 4000-4499:  Georgian SentencePiece tokens (offset by GEORGIAN_OFFSET)
    ID 4500:        Padding (last ID)
"""

import logging
from pathlib import Path

import sentencepiece
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

GEORGIAN_OFFSET = 4000  # First Georgian token ID
GEORGIAN_RANGE = (0x10D0, 0x10FF)  # Unicode range for Georgian letters


def _has_georgian(text: str) -> bool:
    """Check if text contains Georgian characters."""
    for ch in text:
        if GEORGIAN_RANGE[0] <= ord(ch) <= GEORGIAN_RANGE[1]:
            return True
    return False


class GeorgianTokenizer:
    """Combined English + Georgian tokenizer.

    Routes text to the appropriate SentencePiece model based on script
    detection. Georgian tokens are offset by GEORGIAN_OFFSET (4000) so
    they occupy a separate ID range from the English tokens.
    """

    def __init__(self, original_sp: sentencepiece.SentencePieceProcessor,
                 georgian_spm_path: str):
        self.original_sp = original_sp
        self.georgian_sp = sentencepiece.SentencePieceProcessor(
            model_file=str(georgian_spm_path)
        )
        self._georgian_vocab_size = self.georgian_sp.vocab_size()
        logger.info(
            f"Georgian tokenizer loaded: {self._georgian_vocab_size} tokens "
            f"(IDs {GEORGIAN_OFFSET}-{GEORGIAN_OFFSET + self._georgian_vocab_size - 1})"
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (English + Georgian)."""
        return GEORGIAN_OFFSET + self._georgian_vocab_size

    def __call__(self, text: str):
        """Tokenize text, routing to English or Georgian SPM.

        Returns a TokenizedText-compatible result with .tokens attribute.
        """
        if _has_georgian(text):
            ids = self.georgian_sp.encode(text, out_type=int)
            # Offset Georgian IDs to avoid collision with English IDs
            ids = [token_id + GEORGIAN_OFFSET for token_id in ids]
        else:
            ids = self.original_sp.encode(text, out_type=int)

        # Match the SentencePieceTokenizer return format:
        # TokenizedText(tokens=[1, L])
        from pocket_tts.conditioners.base import TokenizedText
        return TokenizedText(torch.tensor(ids)[None, :])

    @property
    def sp(self):
        """Expose Georgian SP for compatibility (decode, etc.)."""
        return self.georgian_sp


def extend_embedding(
    old_embed: nn.Embedding,
    georgian_vocab_size: int,
) -> nn.Embedding:
    """Extend the embedding table to accommodate Georgian tokens.

    Copies all pretrained weights and initializes new Georgian token
    embeddings with the same scale as the pretrained ones.

    Args:
        old_embed: Pretrained embedding [4001, 1024] (4000 + 1 padding)
        georgian_vocab_size: Number of Georgian tokens to add

    Returns:
        New embedding [4000 + georgian_vocab_size + 1, 1024]
    """
    old_vocab = old_embed.num_embeddings  # 4001
    embed_dim = old_embed.embedding_dim   # 1024
    new_vocab = GEORGIAN_OFFSET + georgian_vocab_size + 1  # +1 for padding

    new_embed = nn.Embedding(new_vocab, embed_dim)

    with torch.no_grad():
        # Copy all pretrained weights (IDs 0-4000)
        new_embed.weight.data[:old_vocab] = old_embed.weight.data

        # Initialize Georgian tokens (IDs 4000 to 4000+georgian_vocab_size)
        # Use same scale as pretrained token embeddings (exclude padding token)
        pretrained_std = old_embed.weight.data[:GEORGIAN_OFFSET].std().item()
        nn.init.normal_(
            new_embed.weight.data[old_vocab:],
            mean=0.0,
            std=pretrained_std,
        )

    logger.info(
        f"Extended embedding: [{old_vocab}, {embed_dim}] -> [{new_vocab}, {embed_dim}] "
        f"(+{georgian_vocab_size} Georgian tokens, init std={pretrained_std:.4f})"
    )
    return new_embed
