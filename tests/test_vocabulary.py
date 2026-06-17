"""Tests for bpe.vocabulary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from bpe.vocabulary import (
    BOS_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
)


class TestVocabularyConstruction:
    def test_empty_vocab(self):
        v = Vocabulary()
        assert len(v) == 0

    def test_from_special_tokens(self):
        v = Vocabulary.from_special_tokens()
        assert PAD_TOKEN in v
        assert UNK_TOKEN in v
        assert BOS_TOKEN in v
        assert EOS_TOKEN in v

    def test_special_tokens_get_lowest_ids(self):
        v = Vocabulary.from_special_tokens()
        assert v.pad_id == 0
        assert v.unk_id == 1

    def test_custom_special_tokens(self):
        v = Vocabulary.from_special_tokens(["<blank>", "<sep>"])
        assert "<blank>" in v
        assert "<sep>" in v
        assert v.get_id("<blank>") == 0

    def test_bidirectional_sync_from_token_to_id(self):
        v = Vocabulary(token_to_id={"a": 0, "b": 1})
        assert v.get_token(0) == "a"
        assert v.get_token(1) == "b"


class TestAddToken:
    def test_add_new_token(self):
        v = Vocabulary()
        tid = v.add_token("ব")
        assert tid == 0
        assert "ব" in v

    def test_idempotent(self):
        v = Vocabulary()
        tid1 = v.add_token("ব")
        tid2 = v.add_token("ব")
        assert tid1 == tid2
        assert len(v) == 1

    def test_sequential_ids(self):
        v = Vocabulary()
        v.add_token("ব")
        v.add_token("া")
        v.add_token("ং")
        assert v.get_id("ব") == 0
        assert v.get_id("া") == 1
        assert v.get_id("ং") == 2

    def test_special_flag(self):
        v = Vocabulary()
        v.add_token("<pad>", special=True)
        assert "<pad>" in v.special_tokens

    def test_non_special_not_in_specials(self):
        v = Vocabulary()
        v.add_token("ব")
        assert "ব" not in v.special_tokens


class TestLookup:
    def test_get_id_existing(self):
        v = Vocabulary.from_special_tokens()
        assert v.get_id(PAD_TOKEN) == 0

    def test_get_id_missing_default(self):
        v = Vocabulary()
        assert v.get_id("ব") is None
        assert v.get_id("ব", default=-1) == -1

    def test_get_token_existing(self):
        v = Vocabulary()
        v.add_token("ব")
        assert v.get_token(0) == "ব"

    def test_get_token_missing_default(self):
        v = Vocabulary()
        assert v.get_token(99) is None
        assert v.get_token(99, default="?") == "?"

    def test_contains(self):
        v = Vocabulary()
        v.add_token("ব")
        assert "ব" in v
        assert "া" not in v


class TestProperties:
    def test_size(self):
        v = Vocabulary.from_special_tokens()
        assert v.size == len(DEFAULT_SPECIAL_TOKENS)

    def test_len(self):
        v = Vocabulary.from_special_tokens()
        assert len(v) == len(DEFAULT_SPECIAL_TOKENS)

    def test_pad_id(self):
        v = Vocabulary.from_special_tokens()
        assert v.pad_id is not None

    def test_unk_id(self):
        v = Vocabulary.from_special_tokens()
        assert v.unk_id is not None

    def test_bos_id(self):
        v = Vocabulary.from_special_tokens()
        assert v.bos_id is not None

    def test_eos_id(self):
        v = Vocabulary.from_special_tokens()
        assert v.eos_id is not None

    def test_special_token_not_present_returns_none(self):
        v = Vocabulary()  # no special tokens added
        assert v.pad_id is None


class TestIterator:
    def test_iterate(self):
        v = Vocabulary()
        v.add_token("a")
        v.add_token("b")
        tokens = list(v)
        assert "a" in tokens
        assert "b" in tokens


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        v = Vocabulary.from_special_tokens()
        v.add_token("ব")
        v.add_token("বা")

        path = tmp_path / "vocab.json"
        v.save(path)
        assert path.exists()

        loaded = Vocabulary.load(path)
        assert len(loaded) == len(v)
        assert loaded.get_id("ব") == v.get_id("ব")
        assert loaded.get_id(PAD_TOKEN) == 0
        assert PAD_TOKEN in loaded.special_tokens

    def test_ensure_ascii_false(self, tmp_path):
        v = Vocabulary()
        v.add_token("বাংলাদেশ")
        path = tmp_path / "vocab.json"
        v.save(path)
        raw = path.read_text(encoding="utf-8")
        # Bengali should be stored as actual Unicode, not \uXXXX escapes
        assert "বাংলাদেশ" in raw

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Vocabulary.load(tmp_path / "nonexistent.json")


class TestStatistics:
    def test_token_length_stats(self):
        v = Vocabulary.from_special_tokens()
        v.add_token("ব")  # length 1
        v.add_token("বা")  # length 2
        v.add_token("বাং")  # length 3

        stats = v.token_length_stats()
        assert stats["max"] == 3
        assert stats["min"] == 1
        assert stats["total_tokens"] == 3
        assert stats["mean"] == pytest.approx(2.0)

    def test_longest_tokens(self):
        v = Vocabulary.from_special_tokens()
        v.add_token("ব")
        v.add_token("বাংলাদেশ")
        v.add_token("বা")

        longest = v.longest_tokens(2)
        assert longest[0][0] == "বাংলাদেশ"
        assert longest[0][1] == len("বাংলাদেশ")

    def test_token_length_stats_excludes_specials(self):
        v = Vocabulary.from_special_tokens()
        stats = v.token_length_stats()
        # With only special tokens, stats should be empty
        assert stats == {}

    def test_export_csv(self, tmp_path):
        v = Vocabulary.from_special_tokens()
        v.add_token("ব")
        csv_path = tmp_path / "vocab.csv"
        v.export_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "ব" in content
        assert "id" in content
