"""BPETokenizer - the public-facing Tokenizer built on a trained BPEModel.

This module brings together :class:`~bpe.encoder.BPEEncoder` and
:class:`~bpe.decoder.BPEDecoder` under a single ergonomic API. The trained
model is loaded from a directory that contains ``vocab.json``,
``merges.txt`` and ``metadata.json`` (produced by
:class:`~bpe.trainer.BPETrainer`).

Quick start::

    from bpe.tokenizer import BPETokenizer

    tok = BPETokenizer("output/")
    ids = tok.encode("আমি বাংলাদেশে থাকি।")
    print(ids)               # [45, 120, 7, ...]
    print(tok.decode(ids))   # "আমি বাংলাদেশে থাকি।"
    print(tok.tokenize("বাংলাদেশ"))  # ['বাংলা', 'দেশ</w>']
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .decoder import BPEDecoder
from .encoder import BPEEncoder
from .trainer import BPEModel


class BPETokenizer:
    """Full Tokenizer: encode text → IDs, decode IDs → text.

    Loads a trained model from disk and exposes a simple ``encode`` /
    ``decode`` / ``tokenize`` interface.

    Args:
        model_dir: Path to the directory written by
            :meth:`~bpe.trainer.BPEModel.save`.

    Attributes:
        model: The underlying :class:`~bpe.trainer.BPEModel`.
        vocab_size: Number of tokens in the vocabulary.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model: BPEModel = BPEModel.load(Path(model_dir))

        cfg = self.model.config
        self._encoder = BPEEncoder(
            vocabulary=self.model.vocabulary,
            merges=self.model.merges,
            end_of_word_suffix=cfg.end_of_word_suffix,
            normalization=cfg.normalization,
            normalize_ws=cfg.normalize_whitespace,
        )
        self._decoder = BPEDecoder(
            vocabulary=self.model.vocabulary,
            end_of_word_suffix=cfg.end_of_word_suffix,
        )

    # ── Core API ──────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.model.vocabulary)

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of integer token IDs.

        Args:
            text: Input text (any UTF-8 language).

        Returns:
            List of integer token IDs.

        Example::

            >>> tok.encode("বাংলাদেশ")
            [245, 891]
        """
        return self._encoder.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back to a Unicode string.

        Args:
            token_ids: List of integer IDs produced by :meth:`encode`.

        Returns:
            Reconstructed text.

        Example::

            >>> tok.decode([245, 891])
            'বাংলাদেশ'
        """
        return self._decoder.decode(token_ids)

    def tokenize(self, text: str) -> List[str]:
        """Segment *text* into BPE token strings (not IDs).

        Useful for visualisation, debugging or downstream processing.

        Args:
            text: Input text.

        Returns:
            List of token strings.

        Example::

            >>> tok.tokenize("বাংলাদেশ")
            ['বাংলা', 'দেশ</w>']
        """
        return self._encoder.tokenize(text)

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[Optional[str]]:
        """Map integer IDs to their string representations.

        Unknown IDs produce ``None``.

        Args:
            token_ids: List of token IDs.

        Returns:
            List of token strings (``None`` for unknown IDs).
        """
        return [self.model.vocabulary.get_token(i) for i in token_ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[Optional[int]]:
        """Map token strings to their integer IDs.

        Tokens not in the vocabulary produce ``None``.

        Args:
            tokens: List of token strings.

        Returns:
            List of integer IDs (``None`` for unknown tokens).
        """
        return [self.model.vocabulary.get_id(t) for t in tokens]

    # ── Inspection ────────────────────────────────────────────────────────────

    def inspect(
        self, word: str
    ) -> List[Tuple[List[str], Optional[Tuple[str, str]]]]:
        """Trace the step-by-step BPE merge sequence for *word*.

        Returns a list of ``(symbols, merged_pair)`` tuples (one per merge
        step), starting with the initial character split.

        Used by the ``bpe inspect`` CLI command.

        Args:
            word: A single word (no whitespace).

        Returns:
            Merge trace.  Each element is ``(current_symbols, pair_merged)``.
        """
        return self._encoder.encode_with_trace(word)
