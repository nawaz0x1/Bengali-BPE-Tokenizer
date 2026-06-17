"""BPE Trainer learns merge rules from a raw text corpus.

Algorithm (Sennrich, Haddow & Birch, 2016 "Neural Machine Translation of
Rare Words with Subword Units"):

1.  **Pre-tokenise** the corpus into words separated by whitespace/punctuation.
2.  **Represent** each word as a tuple of Unicode characters plus an
    end-of-word marker (e.g. ``("ব","া","ং","</w>")``).
3.  **Count** the word-type frequencies.
4.  **Initialise** the vocabulary with all unique characters seen.
5.  **Iterate** until the target vocabulary size is reached:

    a.  Count the frequency of every adjacent symbol pair across all words
        (weighted by word frequency).
    b.  Select the most frequent pair.  Ties are broken **lexicographically**
        to guarantee deterministic training.
    c.  Merge every occurrence of that pair in the word-frequency dictionary.
    d.  Update pair-frequency counts **incrementally** (only the counts
        affected by the merge are changed - O(k·C) per step where k is
        average word length and C is the number of affected word types).
    e.  Add the new token to the vocabulary.

6.  **Save** the resulting :class:`BPEModel` (vocab, merges, metadata).

Complexity
~~~~~~~~~~
* Time:  O(V × N) where V = number of merges, N ≈ corpus size.
* Space: O(W × L) where W = unique word types, L = max word length.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False

from .unicode_utils import (
    corpus_unicode_stats,
    normalize,
    normalize_whitespace,
    pretokenize_words,
    split_chars,
)
from .vocabulary import DEFAULT_SPECIAL_TOKENS, Vocabulary

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

#: A word represented as an immutable tuple of BPE symbols.
Word = Tuple[str, ...]

#: Maps a word (symbol-tuple) → its frequency in the corpus.
WordFreqDict = Dict[Word, int]

#: Maps an adjacent symbol pair → its weighted frequency.
PairFreqDict = Dict[Tuple[str, str], int]

#: Ordered list of merge rules produced by training.
MergeList = List[Tuple[str, str]]


# ── Training configuration ────────────────────────────────────────────────────


@dataclass
class TrainerConfig:
    """All hyper-parameters for a BPE training run.

    Attributes:
        vocab_size: Target vocabulary size **including** special tokens and
            the initial character-level symbols.  Training stops as soon as
            ``len(vocabulary) >= vocab_size``.
        min_frequency: Words that appear fewer than this many times in the
            corpus are excluded from frequency counting.  Setting this to 2
            removes hapax legomena and reduces noise.
        end_of_word_suffix: Symbol appended to the last character of every
            word before BPE merges are applied.  Defaults to ``"</w>"``.
            Set to ``""`` to disable end-of-word markers (not recommended
            for word-level reconstruction).
        normalization: Unicode normalisation form applied to the corpus
            before training.  ``"NFC"`` is strongly recommended for Bengali.
        normalize_whitespace: Whether to collapse whitespace runs.
        special_tokens: List of special tokens to include in the vocabulary.
            They are always assigned the lowest IDs (0, 1, 2, …).
        language: Informational language tag stored in ``metadata.json``.
        show_progress: Whether to display a tqdm progress bar.
        log_interval: Log a status line every N merge operations.
        seed: Tie-breaking seed.  Currently used for deterministic
            lexicographic tie-breaking (not a random seed).
    """

    vocab_size: int = 8000
    min_frequency: int = 2
    end_of_word_suffix: str = "</w>"
    normalization: str = "NFC"
    normalize_whitespace: bool = True
    special_tokens: List[str] = field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    language: str = "bengali"
    show_progress: bool = True
    log_interval: int = 500
    seed: int = 42


# ── Trained model artefact ────────────────────────────────────────────────────


@dataclass
class BPEModel:
    """Container for a trained BPE model.

    Holds all the information needed to save, load, encode, or inspect a
    trained Tokenizer.

    Attributes:
        vocabulary: The full token vocabulary.
        merges: Ordered list of merge rules as ``(left, right)`` pairs.
        merge_freq_history: Frequency of the merged pair at each step -
            useful for analysis.
        config: The :class:`TrainerConfig` used during training.
        training_time: Wall-clock seconds spent in :meth:`BPETrainer.train`.
        corpus_size: Number of characters in the (normalised) corpus.
        unicode_stats: Output of :func:`~bpe.unicode_utils.corpus_unicode_stats`.
    """

    vocabulary: Vocabulary
    merges: MergeList
    merge_freq_history: List[Tuple[Tuple[str, str], int]]
    config: TrainerConfig
    training_time: float
    corpus_size: int
    unicode_stats: Dict[str, int]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> None:
        """Persist model to *output_dir*.

        Creates three files:

        * ``vocab.json``    - token → ID mapping and special-token list.
        * ``merges.txt``    - one merge rule per line: ``LEFT RIGHT``
          (tab-separated for robustness against tokens that contain spaces).
        * ``metadata.json`` - training statistics and configuration.

        Args:
            output_dir: Directory to write files to.  Created if absent.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # vocab.json
        self.vocabulary.save(output_dir / "vocab.json")

        # merges.txt - tab-separated so spaces inside tokens are unambiguous
        merges_path = output_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as fh:
            fh.write("#version: 0.2\n")
            for left, right in self.merges:
                fh.write(f"{left}\t{right}\n")

        # metadata.json
        metadata: Dict = {
            "language": self.config.language,
            "corpus_size_chars": self.corpus_size,
            "vocabulary_size": len(self.vocabulary),
            "merge_operations": len(self.merges),
            "training_time_seconds": round(self.training_time, 3),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "unicode_stats": self.unicode_stats,
            "config": {
                "vocab_size": self.config.vocab_size,
                "min_frequency": self.config.min_frequency,
                "end_of_word_suffix": self.config.end_of_word_suffix,
                "normalization": self.config.normalization,
                "normalize_whitespace": self.config.normalize_whitespace,
            },
            "top_merges": [
                {"pair": list(pair), "frequency": freq}
                for pair, freq in self.merge_freq_history[:50]
            ],
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Model saved to %s  (vocab=%d, merges=%d)",
            output_dir,
            len(self.vocabulary),
            len(self.merges),
        )

    @classmethod
    def load(cls, model_dir: Path) -> "BPEModel":
        """Load a model previously saved with :meth:`save`.

        Args:
            model_dir: Directory containing ``vocab.json``, ``merges.txt``
                and ``metadata.json``.

        Returns:
            Loaded :class:`BPEModel` instance.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        model_dir = Path(model_dir)

        vocab_path = model_dir / "vocab.json"
        merges_path = model_dir / "merges.txt"
        meta_path = model_dir / "metadata.json"

        for p in (vocab_path, merges_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file not found: {p}\n"
                    "Make sure you trained a model first with: bpe train ..."
                )

        vocabulary = Vocabulary.load(vocab_path)

        merges: MergeList = []
        with open(merges_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                # Tab-separated format
                if "\t" in line:
                    left, right = line.split("\t", 1)
                else:
                    # Legacy space-separated fallback
                    left, right = line.split(" ", 1)
                merges.append((left, right))

        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        cfg_data = meta.get("config", {})
        config = TrainerConfig(
            vocab_size=cfg_data.get("vocab_size", len(vocabulary)),
            min_frequency=cfg_data.get("min_frequency", 2),
            end_of_word_suffix=cfg_data.get("end_of_word_suffix", "</w>"),
            normalization=cfg_data.get("normalization", "NFC"),
            normalize_whitespace=cfg_data.get("normalize_whitespace", True),
            language=meta.get("language", "unknown"),
        )

        merge_freq_history = [
            (tuple(entry["pair"]), entry["frequency"]) for entry in meta.get("top_merges", [])
        ]

        return cls(
            vocabulary=vocabulary,
            merges=merges,
            merge_freq_history=merge_freq_history,  # type: ignore[arg-type]
            config=config,
            training_time=meta.get("training_time_seconds", 0.0),
            corpus_size=meta.get("corpus_size_chars", 0),
            unicode_stats=meta.get("unicode_stats", {}),
        )


# ── BPE Trainer ───────────────────────────────────────────────────────────────


class BPETrainer:
    """Trains a Byte Pair Encoding model from a raw text corpus.

    Example::

        config = TrainerConfig(vocab_size=8000, language="bengali")
        trainer = BPETrainer(config)
        with open("corpus.txt", encoding="utf-8") as f:
            text = f.read()
        model = trainer.train(text)
        model.save("output/")

    After training, the *output/* directory will contain ``vocab.json``,
    ``merges.txt`` and ``metadata.json``.
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config: TrainerConfig = config or TrainerConfig()
        self.vocabulary: Vocabulary = Vocabulary.from_special_tokens(self.config.special_tokens)
        self.merges: MergeList = []
        self._merge_freq_history: List[Tuple[Tuple[str, str], int]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, text: str) -> BPEModel:
        """Run BPE training on *text*.

        Args:
            text: Raw corpus text (UTF-8 string, any language).

        Returns:
            A :class:`BPEModel` containing the trained vocabulary and merges.
        """
        start_time = time.perf_counter()
        logger.info("BPE training started  (target vocab size: %d)", self.config.vocab_size)

        # ── Step 1: normalise ─────────────────────────────────────────────────
        text = self._normalize_text(text)
        logger.info("Corpus length after normalisation: %d characters", len(text))

        # ── Step 2: Unicode statistics ────────────────────────────────────────
        unicode_stats = corpus_unicode_stats(text)
        logger.info(
            "Unicode stats - Bengali: %d, Latin: %d, Unique codepoints: %d",
            unicode_stats["bengali_chars"],
            unicode_stats["latin_chars"],
            unicode_stats["unique_codepoints"],
        )

        # ── Step 3: word frequencies ──────────────────────────────────────────
        word_freqs = self._build_word_freq(text)
        logger.info("Unique word types (≥ min_freq): %d", len(word_freqs))

        # ── Step 4: initialise character vocabulary ───────────────────────────
        self._init_char_vocab(word_freqs)
        logger.info("Character vocabulary size: %d", len(self.vocabulary))

        # ── Step 5: convert words to symbol sequences ─────────────────────────
        word_syms: WordFreqDict = self._words_to_symbol_seqs(word_freqs)

        # ── Step 6: initial pair counts ───────────────────────────────────────
        pair_freqs: PairFreqDict = self._count_pairs(word_syms)

        # ── Step 7: iterative merge ───────────────────────────────────────────
        merges_needed = self.config.vocab_size - len(self.vocabulary)
        logger.info("Merges needed: %d", merges_needed)

        pbar = None
        if self.config.show_progress and _TQDM_AVAILABLE:
            pbar = _tqdm(total=merges_needed, desc="BPE merges", unit="merge")

        merge_count = 0
        while len(self.vocabulary) < self.config.vocab_size and pair_freqs:
            best_pair, best_freq = self._select_best_pair(pair_freqs)

            # Stop if only singleton pairs remain (no compression possible)
            if best_freq < 2:
                logger.info("Stopping early: best pair frequency = %d (< 2)", best_freq)
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self._merge_freq_history.append((best_pair, best_freq))
            self.vocabulary.add_token(new_token)

            # Incremental update of word_syms and pair_freqs
            word_syms, pair_freqs = self._apply_merge(best_pair, word_syms, pair_freqs)

            merge_count += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"token": new_token, "freq": best_freq, "vocab": len(self.vocabulary)}
                )
            elif merge_count % self.config.log_interval == 0:
                logger.info(
                    "Merge %6d  %r + %r → %r  (freq=%d, vocab=%d)",
                    merge_count,
                    best_pair[0],
                    best_pair[1],
                    new_token,
                    best_freq,
                    len(self.vocabulary),
                )

        if pbar is not None:
            pbar.close()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Training complete - %d merges in %s (final vocab size: %d)",
            merge_count,
            _fmt_time(elapsed),
            len(self.vocabulary),
        )

        return BPEModel(
            vocabulary=self.vocabulary,
            merges=self.merges,
            merge_freq_history=self._merge_freq_history,
            config=self.config,
            training_time=elapsed,
            corpus_size=len(text),
            unicode_stats=unicode_stats,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """Apply configured Unicode normalisation and optional whitespace collapse."""
        text = normalize(text, self.config.normalization)
        if self.config.normalize_whitespace:
            text = normalize_whitespace(text)
        return text

    def _build_word_freq(self, text: str) -> Dict[str, int]:
        """Count word-type frequencies, dropping low-frequency words.

        Args:
            text: Normalised corpus text.

        Returns:
            Mapping of word string → count.
        """
        freq: Counter = Counter(pretokenize_words(text))
        if self.config.min_frequency > 1:
            freq = Counter({w: c for w, c in freq.items() if c >= self.config.min_frequency})
        return dict(freq)

    def _init_char_vocab(self, word_freqs: Dict[str, int]) -> None:
        """Add all unique characters from *word_freqs* to the vocabulary.

        Also adds the end-of-word suffix if one is configured.

        Args:
            word_freqs: Word → frequency mapping.
        """
        for word in word_freqs:
            for ch in word:
                if ch not in self.vocabulary:
                    self.vocabulary.add_token(ch)
        suffix = self.config.end_of_word_suffix
        if suffix and suffix not in self.vocabulary:
            self.vocabulary.add_token(suffix)

    def _words_to_symbol_seqs(self, word_freqs: Dict[str, int]) -> WordFreqDict:
        """Convert each word to a tuple of initial symbols (characters + suffix).

        ``"বাং"`` becomes ``('ব', 'া', 'ং', '</w>')`` (with end-of-word suffix).

        Args:
            word_freqs: Word → frequency mapping.

        Returns:
            Symbol-tuple → frequency mapping.
        """
        suffix = self.config.end_of_word_suffix
        result: WordFreqDict = {}
        for word, freq in word_freqs.items():
            symbols: Word = (
                tuple(split_chars(word)) + (suffix,) if suffix else tuple(split_chars(word))
            )
            result[symbols] = result.get(symbols, 0) + freq
        return result

    @staticmethod
    def _count_pairs(vocab: WordFreqDict) -> PairFreqDict:
        """Count weighted frequency of every adjacent symbol pair.

        For a word ``('ব', 'া', 'ং', '</w>')`` with frequency *f*, the pairs
        ``('ব','া')``, ``('া','ং')``, ``('ং','</w>')`` each receive *f* added
        to their count.

        Args:
            vocab: Symbol-sequence → frequency mapping.

        Returns:
            Pair → weighted-frequency Counter.
        """
        pairs: Counter = Counter()
        for symbols, freq in vocab.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _select_best_pair(
        pair_freqs: PairFreqDict,
    ) -> Tuple[Tuple[str, str], int]:
        """Return the most frequent pair, with lexicographic tie-breaking.

        Deterministic tie-breaking guarantees reproducible training even when
        multiple pairs share the same frequency.

        Args:
            pair_freqs: Current pair → frequency mapping.

        Returns:
            ``(best_pair, frequency)`` tuple.
        """
        best = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        return best, pair_freqs[best]

    @staticmethod
    def _apply_merge(
        pair: Tuple[str, str],
        vocab: WordFreqDict,
        pair_freqs: PairFreqDict,
    ) -> Tuple[WordFreqDict, PairFreqDict]:
        """Apply a merge rule and update pair frequencies incrementally.

        For each word that contains ``pair``:

        1.  Subtract its contribution from all current pair counts.
        2.  Build the merged symbol sequence (replace every ``(A, B)`` with
            ``AB``).
        3.  Add the new word's contribution to pair counts.
        4.  Store the new sequence in the output vocabulary dict.

        Words that do *not* contain ``pair`` are copied unchanged without
        touching pair_freqs.

        Args:
            pair: The ``(A, B)`` pair being merged.
            vocab: Current symbol-sequence → frequency mapping.
            pair_freqs: Current pair → frequency mapping (mutated in place).

        Returns:
            ``(new_vocab, pair_freqs)`` - updated data structures.
        """
        a, b = pair
        merged = a + b
        new_vocab: WordFreqDict = {}

        for symbols, freq in vocab.items():
            # Fast path: skip words that cannot contain this pair.
            if a not in symbols:
                new_vocab[symbols] = new_vocab.get(symbols, 0) + freq
                continue

            # Check whether (a, b) actually occurs as adjacent pair.
            has_pair = any(
                symbols[i] == a and symbols[i + 1] == b for i in range(len(symbols) - 1)
            )
            if not has_pair:
                new_vocab[symbols] = new_vocab.get(symbols, 0) + freq
                continue

            # ── Subtract old pair contributions ───────────────────────────────
            for i in range(len(symbols) - 1):
                old_p = (symbols[i], symbols[i + 1])
                pair_freqs[old_p] = pair_freqs.get(old_p, 0) - freq
                if pair_freqs[old_p] <= 0:
                    pair_freqs.pop(old_p, None)

            # ── Build merged sequence ─────────────────────────────────────────
            new_syms: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            new_sym_tuple: Word = tuple(new_syms)

            # ── Add new pair contributions ────────────────────────────────────
            for i in range(len(new_sym_tuple) - 1):
                new_p = (new_sym_tuple[i], new_sym_tuple[i + 1])
                pair_freqs[new_p] = pair_freqs.get(new_p, 0) + freq

            new_vocab[new_sym_tuple] = new_vocab.get(new_sym_tuple, 0) + freq

        return new_vocab, pair_freqs


# ── Private helpers ───────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"
