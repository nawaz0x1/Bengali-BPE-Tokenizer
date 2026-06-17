"""Tests for bpe.trainer — BPETrainer and BPEModel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from bpe.trainer import BPETrainer, BPEModel, TrainerConfig

# ── Sample corpora ─────────────────────────────────────────────────────────────

BENGALI_CORPUS = (
    "আমি বাংলাদেশে থাকি। "
    "বাংলাদেশ একটি সুন্দর দেশ। "
    "বাংলা আমার মাতৃভাষা। "
    "আমি বাংলায় কথা বলি। "
    "বাংলাদেশের মানুষ অনেক সংস্কৃতিবান। "
    "আমি বাংলাদেশকে ভালোবাসি। "
    "বাংলাদেশে অনেক নদী আছে। "
    "পদ্মা মেঘনা যমুনা বাংলাদেশের প্রধান নদী। "
)

LATIN_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barked at the fox. "
    "the fox was quick and the dog was lazy. "
    "brown foxes are rare but beautiful. "
)

SMALL_CORPUS = "আ আ ব ব ব"  # minimal corpus for quick tests


# ── TrainerConfig ─────────────────────────────────────────────────────────────

class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.vocab_size == 8000
        assert cfg.min_frequency == 2
        assert cfg.end_of_word_suffix == "</w>"
        assert cfg.normalization == "NFC"

    def test_custom(self):
        cfg = TrainerConfig(vocab_size=100, language="english")
        assert cfg.vocab_size == 100
        assert cfg.language == "english"


# ── BPETrainer ────────────────────────────────────────────────────────────────

class TestBPETrainer:
    def _make_trainer(self, vocab_size=50, min_frequency=1) -> BPETrainer:
        cfg = TrainerConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=False,
        )
        return BPETrainer(cfg)

    def test_train_returns_model(self):
        trainer = self._make_trainer(vocab_size=30)
        model = trainer.train(SMALL_CORPUS)
        assert isinstance(model, BPEModel)

    def test_vocab_size_respected(self):
        # BENGALI_CORPUS has ~34 unique chars + 4 specials + 1 eow ≈ 39 initial tokens.
        # We need vocab_size > initial char count to actually trigger merges.
        trainer = self._make_trainer(vocab_size=60)
        model = trainer.train(BENGALI_CORPUS)
        assert len(model.vocabulary) <= 60

    def test_vocab_contains_special_tokens(self):
        trainer = self._make_trainer()
        model = trainer.train(SMALL_CORPUS)
        assert "<pad>" in model.vocabulary
        assert "<unk>" in model.vocabulary

    def test_vocab_contains_all_chars(self):
        trainer = self._make_trainer(vocab_size=200, min_frequency=1)
        model = trainer.train(BENGALI_CORPUS)
        # Every character that appears should be in the vocabulary
        for ch in set("".join(BENGALI_CORPUS.split())):
            assert ch in model.vocabulary, f"Character {ch!r} not in vocabulary"

    def test_merges_are_ordered(self):
        trainer = self._make_trainer(vocab_size=50)
        model = trainer.train(BENGALI_CORPUS)
        # Merges should be a list of (str, str) tuples
        for pair in model.merges:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_merge_count(self):
        # Use a vocab_size larger than the initial char vocabulary so merges happen.
        trainer = self._make_trainer(vocab_size=60)
        model = trainer.train(BENGALI_CORPUS)
        assert len(model.merges) > 0

    def test_latin_corpus(self):
        trainer = self._make_trainer(vocab_size=80, min_frequency=1)
        model = trainer.train(LATIN_CORPUS)
        assert "t" in model.vocabulary
        assert "h" in model.vocabulary
        assert "e" in model.vocabulary

    def test_no_eow(self):
        cfg = TrainerConfig(
            vocab_size=30,
            end_of_word_suffix="",
            min_frequency=1,
            show_progress=False,
        )
        trainer = BPETrainer(cfg)
        model = trainer.train(SMALL_CORPUS)
        # No </w> token should be in vocabulary
        assert "</w>" not in model.vocabulary

    def test_eow_in_vocab(self):
        trainer = self._make_trainer(vocab_size=30, min_frequency=1)
        model = trainer.train(SMALL_CORPUS)
        assert "</w>" in model.vocabulary

    def test_deterministic(self):
        """Same corpus + same config must produce identical models."""
        cfg = TrainerConfig(vocab_size=40, min_frequency=1, show_progress=False)

        m1 = BPETrainer(cfg).train(BENGALI_CORPUS)
        m2 = BPETrainer(cfg).train(BENGALI_CORPUS)

        assert m1.merges == m2.merges
        assert m1.vocabulary.token_to_id == m2.vocabulary.token_to_id

    def test_training_time_recorded(self):
        trainer = self._make_trainer(vocab_size=30)
        model = trainer.train(SMALL_CORPUS)
        assert model.training_time > 0

    def test_unicode_stats_recorded(self):
        trainer = self._make_trainer(vocab_size=30)
        model = trainer.train(BENGALI_CORPUS)
        assert model.unicode_stats["bengali_chars"] > 0


# ── BPEModel save / load ──────────────────────────────────────────────────────

class TestBPEModelSaveLoad:
    def _train(self, corpus=BENGALI_CORPUS, vocab_size=40):
        cfg = TrainerConfig(vocab_size=vocab_size, min_frequency=1, show_progress=False)
        return BPETrainer(cfg).train(corpus)

    def test_save_creates_files(self, tmp_path):
        model = self._train()
        model.save(tmp_path)
        assert (tmp_path / "vocab.json").exists()
        assert (tmp_path / "merges.txt").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_load_roundtrip(self, tmp_path):
        model = self._train()
        model.save(tmp_path)
        loaded = BPEModel.load(tmp_path)

        assert len(loaded.vocabulary) == len(model.vocabulary)
        assert loaded.merges == model.merges

    def test_load_config_preserved(self, tmp_path):
        model = self._train()
        model.save(tmp_path)
        loaded = BPEModel.load(tmp_path)
        assert loaded.config.end_of_word_suffix == "</w>"
        assert loaded.config.normalization == "NFC"

    def test_missing_vocab_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BPEModel.load(tmp_path)

    def test_merges_tab_separated(self, tmp_path):
        model = self._train()
        model.save(tmp_path)
        merges_text = (tmp_path / "merges.txt").read_text(encoding="utf-8")
        # All non-comment lines should be tab-separated
        for line in merges_text.splitlines():
            if line and not line.startswith("#"):
                assert "\t" in line, f"Expected tab in: {line!r}"

    def test_metadata_language(self, tmp_path):
        cfg = TrainerConfig(
            vocab_size=40, min_frequency=1, show_progress=False, language="test_lang"
        )
        model = BPETrainer(cfg).train(BENGALI_CORPUS)
        model.save(tmp_path)
        loaded = BPEModel.load(tmp_path)
        assert loaded.config.language == "test_lang"

    def test_vocab_json_non_ascii(self, tmp_path):
        model = self._train()
        model.save(tmp_path)
        content = (tmp_path / "vocab.json").read_text(encoding="utf-8")
        # Bengali chars should appear as-is, not as \uXXXX escapes
        assert "ব" in content or any(
            is_bengali_in_content(content) for _ in [1]
        )


def is_bengali_in_content(content: str) -> bool:
    """Check if any Bengali character appears literally in the content."""
    return any(0x0980 <= ord(c) <= 0x09FF for c in content)


# ── _apply_merge (internal) ───────────────────────────────────────────────────

class TestApplyMerge:
    """Tests for the incremental merge update logic."""

    def test_simple_merge(self):
        vocab = {("a", "b", "c"): 5}
        pair_freqs = {("a", "b"): 5, ("b", "c"): 5}
        new_vocab, new_freqs = BPETrainer._apply_merge(("a", "b"), vocab, pair_freqs)
        assert ("ab", "c") in new_vocab or tuple(["ab", "c"]) in new_vocab
        assert ("ab", "c") in new_freqs or new_freqs.get(("ab", "c"), 0) > 0

    def test_merge_reduces_vocab(self):
        """After merging (a, b), the word ('a','b','a','b') becomes ('ab','ab')."""
        vocab = {("a", "b", "a", "b"): 3}
        pair_freqs = {("a", "b"): 6, ("b", "a"): 3}
        new_vocab, new_freqs = BPETrainer._apply_merge(("a", "b"), vocab, pair_freqs)
        # ('ab', 'ab') should be in new_vocab
        assert ("ab", "ab") in new_vocab
        assert new_vocab[("ab", "ab")] == 3

    def test_unrelated_words_unchanged(self):
        vocab = {("x", "y"): 10, ("a", "b"): 5}
        pair_freqs = {("x", "y"): 10, ("a", "b"): 5}
        new_vocab, _ = BPETrainer._apply_merge(("a", "b"), vocab, pair_freqs)
        assert ("x", "y") in new_vocab
        assert new_vocab[("x", "y")] == 10

    def test_pair_freq_updated(self):
        vocab = {("a", "b", "c"): 3}
        pair_freqs = {("a", "b"): 3, ("b", "c"): 3}
        _, new_freqs = BPETrainer._apply_merge(("a", "b"), vocab, pair_freqs)
        # ("a", "b") should be removed
        assert ("a", "b") not in new_freqs
        # ("ab", "c") should be added
        assert new_freqs.get(("ab", "c"), 0) == 3
