"""Tests for bpe.unicode_utils."""

import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from bpe.unicode_utils import (
    BENGALI_VIRAMA,
    char_frequency,
    corpus_unicode_stats,
    is_bengali,
    is_bengali_virama,
    is_combining,
    is_digit,
    is_whitespace,
    normalize,
    normalize_whitespace,
    pretokenize,
    pretokenize_words,
    remove_zero_width_chars,
    split_chars,
)

# ── normalize ─────────────────────────────────────────────────────────────────


class TestNormalize:
    def test_nfc_preserves_bengali(self):
        text = "বাংলাদেশ"
        assert normalize(text, "NFC") == text

    def test_nfc_nfd_roundtrip(self):
        text = "বাংলাদেশ"
        decomposed = normalize(text, "NFD")
        recomposed = normalize(decomposed, "NFC")
        assert recomposed == text

    def test_nfc_is_default(self):
        text = "hello"
        assert normalize(text) == text

    def test_invalid_form_raises(self):
        with pytest.raises(ValueError, match="Unknown Unicode normalisation"):
            normalize("text", "XYZ")

    def test_latin_nfc(self):
        # café composed vs decomposed
        composed = "caf\u00e9"  # é as single code point
        decomposed = "cafe\u0301"  # e + combining acute
        assert normalize(composed, "NFC") == composed
        assert normalize(decomposed, "NFC") == composed


# ── normalize_whitespace ──────────────────────────────────────────────────────


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_newlines_collapsed(self):
        assert normalize_whitespace("a\n\nb") == "a b"

    def test_tabs_collapsed(self):
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_bengali_with_spaces(self):
        text = "আমি  বাংলাদেশে   থাকি"
        assert normalize_whitespace(text) == "আমি বাংলাদেশে থাকি"

    def test_non_breaking_space(self):
        # U+00A0 is a non-breaking space - should be collapsed
        assert normalize_whitespace("a\u00a0b") == "a b"


# ── remove_zero_width_chars ───────────────────────────────────────────────────


class TestRemoveZeroWidth:
    def test_removes_zwnj(self):
        text = "ক্\u200cষ"
        result = remove_zero_width_chars(text)
        assert "\u200c" not in result

    def test_removes_zwj(self):
        text = "a\u200db"
        assert remove_zero_width_chars(text) == "ab"

    def test_leaves_other_chars(self):
        text = "বাংলা"
        assert remove_zero_width_chars(text) == text


# ── is_bengali ────────────────────────────────────────────────────────────────


class TestIsBengali:
    def test_bengali_consonant(self):
        assert is_bengali("ব") is True

    def test_bengali_vowel(self):
        assert is_bengali("আ") is True

    def test_bengali_dependent_vowel(self):
        assert is_bengali("া") is True  # U+09BE

    def test_bengali_virama(self):
        assert is_bengali("্") is True

    def test_latin_char(self):
        assert is_bengali("a") is False

    def test_digit(self):
        assert is_bengali("0") is False

    def test_bengali_digit(self):
        assert is_bengali("০") is True  # U+09E6

    def test_space(self):
        assert is_bengali(" ") is False


# ── is_bengali_virama ─────────────────────────────────────────────────────────


class TestIsBengaliVirama:
    def test_virama(self):
        assert is_bengali_virama("\u09cd") is True

    def test_not_virama(self):
        assert is_bengali_virama("ব") is False
        assert is_bengali_virama("া") is False


# ── is_combining ─────────────────────────────────────────────────────────────


class TestIsCombining:
    def test_bengali_virama_is_combining(self):
        # U+09CD has category Mn (non-spacing mark)
        assert is_combining("\u09cd") is True

    def test_bengali_dependent_vowel_aa(self):
        # U+09BE (আ-কার) has category Mc (spacing combining)
        assert is_combining("\u09be") is True

    def test_latin_not_combining(self):
        assert is_combining("a") is False


# ── is_digit ──────────────────────────────────────────────────────────────────


class TestIsDigit:
    def test_ascii_digit(self):
        assert is_digit("5") is True

    def test_bengali_digit(self):
        assert is_digit("৭") is True  # U+09ED

    def test_letter(self):
        assert is_digit("a") is False

    def test_bengali_letter(self):
        assert is_digit("ব") is False


# ── is_whitespace ─────────────────────────────────────────────────────────────


class TestIsWhitespace:
    def test_space(self):
        assert is_whitespace(" ") is True

    def test_newline(self):
        assert is_whitespace("\n") is True

    def test_tab(self):
        assert is_whitespace("\t") is True

    def test_letter(self):
        assert is_whitespace("a") is False

    def test_bengali(self):
        assert is_whitespace("ব") is False


# ── split_chars ───────────────────────────────────────────────────────────────


class TestSplitChars:
    def test_bengali_word(self):
        result = split_chars("বাং")
        assert result == ["ব", "া", "ং"]

    def test_latin_word(self):
        assert split_chars("hello") == ["h", "e", "l", "l", "o"]

    def test_empty(self):
        assert split_chars("") == []

    def test_single_char(self):
        assert split_chars("ব") == ["ব"]

    def test_mixed(self):
        # Mixing Bengali and ASCII
        result = split_chars("a ব")
        assert result == ["a", " ", "ব"]

    def test_conjunct(self):
        # ক্ষ is 3 code points: ক + ্ + ষ
        result = split_chars("ক্ষ")
        assert len(result) == 3
        assert result[0] == "ক"
        assert result[1] == BENGALI_VIRAMA
        assert result[2] == "ষ"


# ── pretokenize ───────────────────────────────────────────────────────────────


class TestPretokenize:
    def test_bengali_sentence(self):
        tokens = pretokenize("আমি বাংলাদেশে থাকি")
        # Should split on spaces
        non_ws = [t for t in tokens if not t.isspace()]
        assert "আমি" in non_ws
        assert "বাংলাদেশে" in non_ws
        assert "থাকি" in non_ws

    def test_danda_separated(self):
        tokens = pretokenize("থাকি।")
        non_ws = [t for t in tokens if not t.isspace()]
        assert "থাকি" in non_ws
        assert "।" in non_ws

    def test_ascii_punctuation(self):
        tokens = pretokenize("hello, world!")
        non_ws = [t for t in tokens if not t.isspace()]
        assert "hello" in non_ws
        assert "world" in non_ws

    def test_empty_string(self):
        assert pretokenize("") == []

    def test_whitespace_only(self):
        tokens = pretokenize("   ")
        # All tokens are whitespace
        assert all(t.isspace() for t in tokens)


class TestPretokenizeWords:
    def test_no_whitespace_tokens(self):
        words = pretokenize_words("আমি বাংলাদেশে থাকি।")
        assert all(not t.isspace() for t in words)

    def test_content_preserved(self):
        words = pretokenize_words("বাংলাদেশ")
        assert "বাংলাদেশ" in words


# ── corpus_unicode_stats ──────────────────────────────────────────────────────


class TestCorpusUnicodeStats:
    def test_bengali_counting(self):
        text = "বাংলাদেশ"
        stats = corpus_unicode_stats(text)
        assert stats["bengali_chars"] == len("বাংলাদেশ")
        assert stats["total_chars"] == len("বাংলাদেশ")
        assert stats["latin_chars"] == 0

    def test_mixed_counting(self):
        text = "a ব"  # 1 latin, 1 space, 1 bengali
        stats = corpus_unicode_stats(text)
        assert stats["latin_chars"] == 1
        assert stats["bengali_chars"] == 1
        assert stats["whitespace"] == 1

    def test_unique_codepoints(self):
        stats = corpus_unicode_stats("aabb")
        assert stats["unique_codepoints"] == 2  # only 'a' and 'b'


# ── char_frequency ────────────────────────────────────────────────────────────


class TestCharFrequency:
    def test_counts_correctly(self):
        freq = char_frequency("আআব")
        assert freq["আ"] == 2
        assert freq["ব"] == 1

    def test_sorted_descending(self):
        freq = char_frequency("bbbaa")
        keys = list(freq.keys())
        assert keys[0] == "b"  # higher frequency first

    def test_empty_string(self):
        assert char_frequency("") == {}
