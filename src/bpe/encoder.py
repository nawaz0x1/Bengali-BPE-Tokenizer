"""BPE Encoder converts raw text into sequences of token IDs.

Encoding pipeline
~~~~~~~~~~~~~~~~~
1. **Normalise** the input text with the same settings used during training.
2. **Pre-tokenise** into word-level units (splitting at whitespace and
   punctuation, identical to training).
3. For each word, apply the **greedy BPE merge** algorithm:

   a. Start with the word split into individual Unicode characters plus the
      end-of-word marker (e.g. ``['ব','া','ং','</w>']``).
   b. In a loop, find the pair with the *lowest merge rank* (i.e. the pair
      that was merged earliest during training).
   c. Merge **all** occurrences of that pair in the current symbol list.
   d. Repeat until no mergeable pair remains.

4. Look up each final symbol in the vocabulary to obtain its integer ID.
   Unknown symbols fall back to the ``<unk>`` ID.

Caching
~~~~~~~
Word-level BPE results are cached with ``functools.lru_cache`` (default
capacity 65 536 entries).  For Bengali text where the same word forms recur
heavily, this gives a large speedup.

Complexity
~~~~~~~~~~
Encoding a single word of length *L* with *M* merge rules applied takes
O(L²) in the worst case (each merge pass scans the full symbol list).
In practice, L is small (< 20 characters per word) so this is negligible.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .unicode_utils import normalize, normalize_whitespace, pretokenize_words, split_chars
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


MergeRankMap = Dict[Tuple[str, str], int]


class BPEEncoder:
    """Applies trained BPE merge rules to encode text as token IDs.

    Args:
        vocabulary: The trained :class:`~bpe.vocabulary.Vocabulary`.
        merges: Ordered list of ``(left, right)`` merge rules (earliest first).
        end_of_word_suffix: The end-of-word marker appended during training
            (e.g. ``"</w>"``).  Pass ``""`` if training used no suffix.
        normalization: Unicode normalisation form to apply before encoding.
        normalize_ws: Whether to collapse whitespace before encoding.
        cache_size: Maximum number of word-level results to cache.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        merges: List[Tuple[str, str]],
        end_of_word_suffix: str = "</w>",
        normalization: str = "NFC",
        normalize_ws: bool = True,
        cache_size: int = 65_536,
    ) -> None:
        self._vocab = vocabulary
        self._merges = merges
        self._eow = end_of_word_suffix
        self._norm_form = normalization
        self._normalize_ws = normalize_ws

        # Build merge rank map: pair → rank (lower = applied earlier)
        self._bpe_ranks: MergeRankMap = {pair: rank for rank, pair in enumerate(merges)}

        # Word-level cache (bound method cannot use @lru_cache directly,
        # so we store a module-level cached function per instance)
        self._encode_word_cached = lru_cache(maxsize=cache_size)(self._encode_word_uncached)

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of token IDs.

        Unknown tokens are mapped to the ``<unk>`` ID.  If the vocabulary
        has no ``<unk>`` token, unknown tokens are silently skipped.

        Args:
            text: Input text (any language, UTF-8).

        Returns:
            List of integer token IDs.
        """
        token_ids: List[int] = []
        for token_str in self.tokenize(text):
            tid = self._vocab.get_id(token_str)
            if tid is None:
                # Unknown sub-token: map to <unk> or skip
                tid = self._vocab.unk_id
            if tid is not None:
                token_ids.append(tid)
        return token_ids

    def tokenize(self, text: str) -> List[str]:
        """Segment *text* into BPE token strings (without converting to IDs).

        Useful for inspection or display purposes.

        Args:
            text: Input text.

        Returns:
            List of token strings (e.g. ``['বাংলা', 'দেশ</w>']``).
        """
        text = self._preprocess(text)
        token_strs: List[str] = []
        for word in pretokenize_words(text):
            token_strs.extend(self._encode_word_cached(word))
        return token_strs

    def encode_with_trace(self, word: str) -> List[Tuple[List[str], Optional[Tuple[str, str]]]]:
        """Encode a single *word* and record each merge step.

        Returns a list of ``(symbols, merged_pair)`` pairs where:

        * ``symbols`` is the symbol list *after* the merge.
        * ``merged_pair`` is the ``(A, B)`` pair that was merged to reach
          this state (``None`` for the initial state).

        This is used by the ``inspect`` CLI command to show step-by-step
        merge visualisation.

        Args:
            word: A single word (no whitespace).

        Returns:
            Trace list starting with the initial character split.
        """
        if self._eow:
            symbols: List[str] = split_chars(word) + [self._eow]
        else:
            symbols = split_chars(word)

        trace: List[Tuple[List[str], Optional[Tuple[str, str]]]] = [(list(symbols), None)]

        while len(symbols) > 1:
            best_rank = float("inf")
            best_pair: Optional[Tuple[str, str]] = None

            for i in range(len(symbols) - 1):
                candidate = (symbols[i], symbols[i + 1])
                rank = self._bpe_ranks.get(candidate, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = candidate

            if best_pair is None or best_rank == float("inf"):
                break  # No more mergeable pairs

            symbols = _merge_symbols(symbols, best_pair)
            trace.append((list(symbols), best_pair))

        return trace

    def _preprocess(self, text: str) -> str:
        """Apply normalisation consistent with training."""
        text = normalize(text, self._norm_form)
        if self._normalize_ws:
            text = normalize_whitespace(text)
        return text

    def _encode_word_uncached(self, word: str) -> Tuple[str, ...]:
        """Encode a single *word* using greedy BPE merges.

        This function is wrapped by ``lru_cache`` through the instance
        attribute ``_encode_word_cached``.

        The greedy algorithm:
        1. Start with the word split into characters + end-of-word suffix.
        2. Each iteration, find the pair with the lowest merge rank.
        3. Merge **all** occurrences of that pair in the symbol list.
        4. Repeat until no ranked pair remains.

        Args:
            word: Single pre-token word string.

        Returns:
            Tuple of BPE token strings.
        """
        if self._eow:
            symbols: List[str] = split_chars(word) + [self._eow]
        else:
            symbols = split_chars(word)

        # Handle single-character words immediately
        if len(symbols) <= 1:
            return tuple(symbols)

        while len(symbols) > 1:
            # Find the pair with the lowest (earliest) merge rank
            best_rank = float("inf")
            best_pair: Optional[Tuple[str, str]] = None

            for i in range(len(symbols) - 1):
                candidate = (symbols[i], symbols[i + 1])
                rank = self._bpe_ranks.get(candidate, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = candidate

            if best_pair is None or best_rank == float("inf"):
                break  # No applicable merge rule found

            symbols = _merge_symbols(symbols, best_pair)

        return tuple(symbols)

    def cache_info(self) -> str:
        """Return a string summary of the word-level LRU cache statistics."""
        info = self._encode_word_cached.cache_info()
        return (
            f"hits={info.hits}, misses={info.misses}, "
            f"maxsize={info.maxsize}, currsize={info.currsize}"
        )

    def clear_cache(self) -> None:
        """Evict all cached word-level encodings."""
        self._encode_word_cached.cache_clear()


def _merge_symbols(symbols: List[str], pair: Tuple[str, str]) -> List[str]:
    """Replace every occurrence of *pair* in *symbols* with its merged form.

    All non-overlapping occurrences are merged in a single left-to-right
    pass, consistent with the training procedure.

    Args:
        symbols: Current symbol list.
        pair: ``(A, B)`` pair to merge.

    Returns:
        New symbol list with all occurrences of ``(A, B)`` replaced by
        ``AB``.
    """
    a, b = pair
    merged = a + b
    new_symbols: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            new_symbols.append(merged)
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return new_symbols
