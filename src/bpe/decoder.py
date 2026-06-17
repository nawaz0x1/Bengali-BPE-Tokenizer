"""BPE Decoder reconstructs text from a sequence of token IDs.

Decoding pipeline
~~~~~~~~~~~~~~~~~
1. Map each integer ID to its token string using the vocabulary.
2. Skip special tokens (``<pad>``, ``<bos>``, etc.).
3. Concatenate all token strings.
4. Replace the end-of-word suffix (``</w>``) with a space to recover
   word boundaries.

Round-trip accuracy
~~~~~~~~~~~~~~~~~~~
For text that was *encoded* with this Tokenizer, ``decode(encode(text))``
reproduces the original text up to Unicode normalisation and whitespace
collapsing (which are also applied during encoding).

For text with characters outside the training vocabulary, those characters
were replaced by ``<unk>`` during encoding and cannot be recovered.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class BPEDecoder:
    """Reconstructs text from a list of BPE token IDs.

    Args:
        vocabulary: The trained :class:`~bpe.vocabulary.Vocabulary`.
        end_of_word_suffix: End-of-word marker used during training
            (e.g. ``"</w>"``).  Set to ``""`` if no suffix was used.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        end_of_word_suffix: str = "</w>",
    ) -> None:
        self._vocab = vocabulary
        self._eow = end_of_word_suffix

    # ── Public API ────────────────────────────────────────────────────────────

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back to a Unicode string.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Reconstructed text string.
        """
        token_strings = self._ids_to_strings(token_ids)
        return self._join(token_strings)

    def decode_tokens(self, tokens: List[str]) -> str:
        """Convert a list of token strings (not IDs) back to text.

        Useful when you already have the string representations.

        Args:
            tokens: List of BPE token strings.

        Returns:
            Reconstructed text string.
        """
        filtered = [t for t in tokens if t not in self._vocab.special_tokens]
        return self._join(filtered)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ids_to_strings(self, token_ids: List[int]) -> List[str]:
        """Map integer IDs to token strings, skipping specials and unknowns.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            List of token strings (specials excluded).
        """
        result: List[str] = []
        for tid in token_ids:
            token: Optional[str] = self._vocab.get_token(tid)
            if token is None:
                logger.debug("Unknown token ID %d — skipped", tid)
                continue
            if token in self._vocab.special_tokens:
                continue
            result.append(token)
        return result

    def _join(self, tokens: List[str]) -> str:
        """Concatenate tokens and recover whitespace from end-of-word markers.

        The concatenated string has the end-of-word suffix embedded within it.
        We replace each suffix with a single ASCII space and strip trailing
        whitespace.

        Example (end_of_word_suffix = "</w>")::

            ['বাং', 'লা</w>', 'দেশ</w>'] → "বাংলা দেশ"

        Args:
            tokens: List of token strings (no special tokens).

        Returns:
            Reconstructed text.
        """
        text = "".join(tokens)
        if self._eow:
            text = text.replace(self._eow, " ")
        return text.strip()
