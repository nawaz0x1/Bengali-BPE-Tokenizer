"""Unicode utilities for BPE tokenisation.

This module is the foundation of correct non-Latin text handling. Every
function here operates exclusively on Unicode *code points* (Python ``str``),
never on raw UTF-8 bytes. Python 3's ``str`` type is already a sequence of
Unicode code points, so ``list(text)`` always gives the correct per-character
split regardless of script.

Bengali Unicode considerations
================================
Bengali is encoded in the Unicode block **U+0980-U+09FF**. Key sub-ranges:

===================  ========  ============================================
Character type        Range     Notes
===================  ========  ============================================
Independent vowels    0985-0994  স্বরবর্ণ
Consonants            0995-09B9  ব্যঞ্জনবর্ণ
Nukta                 09BC       Modifies consonants (like "ড়" = ড + ়)
Dependent vowels      09BE-09CC  কার (vowel signs attached to consonants)
Virama / Hasanta      09CD       ্ - suppresses the inherent /a/ vowel;
                                 consonant + virama + consonant = conjunct
Anusvara              0982       ং (nasalisation)
Visarga               0983       ঃ (aspiration)
Chandrabindu          0981       ঁ (nasalised vowel)
Khanda Ta             09CE       ৎ (final form of ত)
Bengali digits        09E6-09EF  ০১২৩৪৫৬৭৮৯
===================  ========  ============================================

Conjuncts (যুক্তাক্ষর)
   Formed as: consonant + U+09CD (virama) + consonant
   Example:   ক (U+0995) + ্ (U+09CD) + ষ (U+09B7) = ক্ষ

Zero-width characters
   U+200C  ZWNJ - prevents conjunct formation when placed between
            consonant + virama + consonant.
   U+200D  ZWJ  - forces conjunct or ligature formation.

The BPE Tokenizer treats each code point as an indivisible atom.  It never
attempts to form or split grapheme clusters - that is left to the caller if
needed (e.g. for display-width calculations).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterator, List, Set

# ── Bengali Unicode block ─────────────────────────────────────────────────────

BENGALI_BLOCK_START: int = 0x0980
BENGALI_BLOCK_END: int = 0x09FF

# Specific character constants
BENGALI_VIRAMA: str = "\u09cd"  # ্  (virama / hasanta)
BENGALI_ANUSVARA: str = "\u0982"  # ং
BENGALI_VISARGA: str = "\u0983"  # ঃ
BENGALI_CHANDRABINDU: str = "\u0981"  # ঁ
BENGALI_NUKTA: str = "\u09bc"  # ়

ZERO_WIDTH_NON_JOINER: str = "\u200c"  # ZWNJ
ZERO_WIDTH_JOINER: str = "\u200d"  # ZWJ
ZERO_WIDTH_SPACE: str = "\u200b"  # ZWSP

# Character sets for category checks
BENGALI_INDEPENDENT_VOWELS: frozenset[str] = frozenset(chr(c) for c in range(0x0985, 0x0995))
BENGALI_CONSONANTS: frozenset[str] = frozenset(chr(c) for c in range(0x0995, 0x09BA))
BENGALI_DEPENDENT_VOWELS: frozenset[str] = frozenset(
    chr(c)
    for c in [
        0x09BE,
        0x09BF,
        0x09C0,
        0x09C1,
        0x09C2,
        0x09C3,
        0x09C4,
        0x09C7,
        0x09C8,
        0x09CB,
        0x09CC,
    ]
)

# Unicode general-category groups
_SPACE_CATEGORIES: frozenset[str] = frozenset({"Zs", "Zl", "Zp"})
_PUNCT_CATEGORIES: frozenset[str] = frozenset({"Po", "Ps", "Pe", "Pi", "Pf", "Pd", "Pc"})
_COMBINING_CATEGORIES: frozenset[str] = frozenset({"Mn", "Mc", "Me"})

# ── Normalisation ─────────────────────────────────────────────────────────────


def normalize(text: str, form: str = "NFC") -> str:
    """Normalise *text* to the given Unicode normal form.

    **NFC** is strongly recommended for Bengali (and most non-Latin scripts).
    It ensures pre-composed characters are used consistently. For example,
    the vowel sign ো can be represented as a single code point (U+09CB) *or*
    as the two-code-point sequence া (U+09BE) + ে (U+09CB). NFC chooses the
    canonical composed form, eliminating this ambiguity.

    Args:
        text: Input Unicode string.
        form: One of ``"NFC"``, ``"NFD"``, ``"NFKC"``, ``"NFKD"``.

    Returns:
        Normalised string.

    Raises:
        ValueError: If *form* is not a recognised normalisation form.
    """
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(
            f"Unknown Unicode normalisation form {form!r}. "
            "Expected one of: NFC, NFD, NFKC, NFKD."
        )
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single ASCII space.

    Handles all Unicode whitespace classes including:
    * ASCII space, tab, newline, carriage return
    * Unicode Zs / Zl / Zp category characters (e.g. non-breaking space U+00A0,
      em-space U+2003, ideographic space U+3000)

    Args:
        text: Input text.

    Returns:
        Text with whitespace runs collapsed and leading/trailing space stripped.
    """
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()


def remove_zero_width_chars(text: str) -> str:
    """Strip ZWNJ, ZWJ and ZWSP from *text*.

    .. warning::
        ZWNJ (U+200C) has **orthographic significance** in Bengali: it prevents
        conjunct formation. For example ``ক্‌ষ`` (ক + ্ + ZWNJ + ষ) renders as
        two separate half-consonants rather than the conjunct ক্ষ. Remove only
        when you are sure the distinction is irrelevant (e.g. for a
        bag-of-words model on already-rendered text).

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    return (
        text.replace(ZERO_WIDTH_NON_JOINER, "")
        .replace(ZERO_WIDTH_JOINER, "")
        .replace(ZERO_WIDTH_SPACE, "")
    )


# ── Character classification ──────────────────────────────────────────────────


def is_bengali(char: str) -> bool:
    """Return ``True`` if *char* is in the Bengali Unicode block (U+0980-U+09FF)."""
    return BENGALI_BLOCK_START <= ord(char) <= BENGALI_BLOCK_END


def is_bengali_virama(char: str) -> bool:
    """Return ``True`` if *char* is the Bengali virama ্ (U+09CD)."""
    return char == BENGALI_VIRAMA


def is_bengali_dependent_vowel(char: str) -> bool:
    """Return ``True`` if *char* is a Bengali dependent vowel sign (কার)."""
    return char in BENGALI_DEPENDENT_VOWELS


def is_combining(char: str) -> bool:
    """Return ``True`` if *char* is a Unicode combining mark (Mn / Mc / Me)."""
    return unicodedata.category(char) in _COMBINING_CATEGORIES


def is_whitespace(char: str) -> bool:
    """Return ``True`` if *char* is any Unicode whitespace."""
    return unicodedata.category(char) in _SPACE_CATEGORIES or char in "\t\n\r "


def is_punctuation(char: str) -> bool:
    """Return ``True`` if *char* is a Unicode punctuation character."""
    return unicodedata.category(char) in _PUNCT_CATEGORIES


def is_digit(char: str) -> bool:
    """Return ``True`` if *char* is a Unicode decimal digit (category Nd).

    This includes Bengali digits ০-৯ (U+09E6-U+09EF) as well as ASCII 0-9.
    """
    return unicodedata.category(char) == "Nd"


# ── Pre-tokenisation ──────────────────────────────────────────────────────────

# This regex splits text into meaningful units.
# Order matters: more-specific patterns come before \S+.
# The final alternative ("word") explicitly excludes characters that are
# handled by the other alternatives so they are never swallowed inside a word.
_PUNCT_CHARS = r".,!?;:'\"()\[\]{}<>\/\\@#$%^&*+=|`~"
_PRETOKENIZE_RE = re.compile(
    r"\s+"
    r"|[\u0964\u0965]"  # Bengali danda / double danda
    r"|[0-9]+"  # ASCII numeral run
    r"|[\u09E6-\u09EF]+"  # Bengali numeral run
    r"|[.,!?;:'\"()\[\]{}<>\/\\@#$%^&*+=|`~]+"  # ASCII punctuation cluster
    r"|[^\s.,!?;:'\"()\[\]{}<>\/\\@#$%^&*+=|`~\u0964\u0965\u09E6-\u09EF]+",  # word
    re.UNICODE,
)


def pretokenize(text: str) -> List[str]:
    """Split *text* into a flat list of pre-tokens.

    Pre-tokens are the smallest units that BPE will *not* merge across.
    Splitting at whitespace and punctuation boundaries is the standard
    approach (following Sennrich et al. 2016).

    Python's ``re`` module operates on Unicode code points, so this function
    never splits in the middle of a multi-byte UTF-8 character.

    Args:
        text: Input text (any language, any script).

    Returns:
        List of pre-token strings (including whitespace tokens).

    Example::

        >>> pretokenize("আমি বাংলাদেশে থাকি।")
        ['আমি', ' ', 'বাংলাদেশে', ' ', 'থাকি', '।']
    """
    return [m.group() for m in _PRETOKENIZE_RE.finditer(text)]


def pretokenize_words(text: str) -> List[str]:
    """Like :func:`pretokenize` but returns only non-whitespace tokens.

    Useful for building word-frequency dictionaries during training.

    Args:
        text: Input text.

    Returns:
        List of word/punctuation tokens (whitespace excluded).
    """
    return [t for t in pretokenize(text) if not t.isspace()]


# ── Character-level splitting ─────────────────────────────────────────────────


def split_chars(text: str) -> List[str]:
    """Split a string into individual Unicode code points.

    This is the correct way to iterate character-by-character over any
    Unicode string. Python 3's ``str`` is already a sequence of code
    points, so ``list(text)`` never breaks multi-byte UTF-8 characters.

    Args:
        text: Input string.

    Returns:
        List of single-character strings.

    Example::

        >>> split_chars("বাং")
        ['ব', 'া', 'ং']
    """
    return list(text)


def iter_chars(text: str) -> Iterator[str]:
    """Iterate over Unicode code points in *text* one at a time."""
    yield from text


# ── Unicode statistics ────────────────────────────────────────────────────────


def corpus_unicode_stats(text: str) -> Dict[str, int]:
    """Compute per-category character counts for an entire corpus.

    Returns a dict with the following keys:

    * ``total_chars``      - total code points (excluding nothing)
    * ``bengali_chars``    - chars in U+0980-U+09FF
    * ``latin_chars``      - ASCII a-z / A-Z
    * ``digits``           - Unicode decimal digits (any script)
    * ``whitespace``       - whitespace characters
    * ``punctuation``      - Unicode punctuation characters
    * ``combining_marks``  - combining diacritics (Mn / Mc / Me)
    * ``zero_width_chars`` - ZWNJ + ZWJ + ZWSP
    * ``unique_codepoints``- number of distinct code points seen

    Args:
        text: Corpus text.

    Returns:
        Statistics dictionary.
    """
    counts: Dict[str, int] = {
        "total_chars": 0,
        "bengali_chars": 0,
        "latin_chars": 0,
        "digits": 0,
        "whitespace": 0,
        "punctuation": 0,
        "combining_marks": 0,
        "zero_width_chars": 0,
        "unique_codepoints": 0,
    }
    seen: Set[str] = set()

    for ch in text:
        counts["total_chars"] += 1
        seen.add(ch)

        if is_bengali(ch):
            counts["bengali_chars"] += 1
        elif "a" <= ch.lower() <= "z":
            counts["latin_chars"] += 1

        if is_digit(ch):
            counts["digits"] += 1
        if is_whitespace(ch):
            counts["whitespace"] += 1
        if is_punctuation(ch):
            counts["punctuation"] += 1
        if is_combining(ch):
            counts["combining_marks"] += 1
        if ch in {ZERO_WIDTH_NON_JOINER, ZERO_WIDTH_JOINER, ZERO_WIDTH_SPACE}:
            counts["zero_width_chars"] += 1

    counts["unique_codepoints"] = len(seen)
    return counts


def char_frequency(text: str) -> Dict[str, int]:
    """Compute per-character frequency for *text*, sorted descending by count.

    Args:
        text: Input text.

    Returns:
        Dict mapping character → count, sorted by count descending.
    """
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: -kv[1]))
