"""Tests for BPETokenizer (encode / decode / inspect round-trips)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from bpe.trainer import BPETrainer, TrainerConfig
from bpe.tokenizer import BPETokenizer
from bpe.encoder import BPEEncoder, _merge_symbols
from bpe.decoder import BPEDecoder
from bpe.vocabulary import Vocabulary

# ── Corpus for integration tests ──────────────────────────────────────────────

CORPUS = (
    "আমি বাংলাদেশে থাকি। "
    "বাংলাদেশ একটি সুন্দর দেশ। "
    "বাংলা আমার মাতৃভাষা। "
    "আমি বাংলায় কথা বলি। "
    "বাংলাদেশের মানুষ অনেক ভালো। "
    "আমি বাংলাদেশকে ভালোবাসি। "
    "পদ্মা মেঘনা যমুনা বাংলাদেশের নদী। "
    "ঢাকা বাংলাদেশের রাজধানী। "
) * 5  # repeat to increase frequency counts


def _train_and_save(tmp_path: Path, vocab_size: int = 80) -> Path:
    """Helper: train a small model and return the output directory."""
    cfg = TrainerConfig(vocab_size=vocab_size, min_frequency=1, show_progress=False)
    model = BPETrainer(cfg).train(CORPUS)
    model.save(tmp_path)
    return tmp_path


# ── BPETokenizer integration ──────────────────────────────────────────────────


class TestBPETokenizerIntegration:
    @pytest.fixture
    def tokenizer(self, tmp_path):
        _train_and_save(tmp_path)
        return BPETokenizer(tmp_path)

    def test_encode_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("বাংলাদেশ")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_nonempty(self, tokenizer):
        ids = tokenizer.encode("বাংলাদেশ")
        assert len(ids) > 0

    def test_decode_roundtrip(self, tokenizer):
        text = "বাংলাদেশ"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        # Decoded should contain the original characters
        # (may differ in whitespace due to </w> handling)
        assert "বাংলাদেশ" in decoded

    def test_encode_empty_string(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_tokenize_returns_strings(self, tokenizer):
        tokens = tokenizer.tokenize("বাংলাদেশ")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_covers_input(self, tokenizer):
        """Joining tokens (removing </w>) should recover the original word."""
        word = "বাংলা"
        tokens = tokenizer.tokenize(word)
        joined = "".join(tokens).replace("</w>", "")
        assert joined == word

    def test_vocab_size_property(self, tokenizer):
        assert tokenizer.vocab_size > 0

    def test_convert_ids_to_tokens(self, tokenizer):
        ids = tokenizer.encode("বাংলাদেশ")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        assert len(tokens) == len(ids)
        assert all(t is not None for t in tokens)

    def test_convert_tokens_to_ids(self, tokenizer):
        ids = tokenizer.encode("বাংলাদেশ")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        back_ids = tokenizer.convert_tokens_to_ids(tokens)
        assert back_ids == ids

    def test_inspect_returns_trace(self, tokenizer):
        trace = tokenizer.inspect("বাংলাদেশ")
        assert isinstance(trace, list)
        assert len(trace) >= 1
        # First step should be the initial split (no merged pair)
        symbols_0, pair_0 = trace[0]
        assert pair_0 is None
        # Initial symbols should be individual characters + </w>
        expected_chars = list("বাংলাদেশ") + ["</w>"]
        assert symbols_0 == expected_chars

    def test_inspect_final_state(self, tokenizer):
        trace = tokenizer.inspect("বাংলা")
        # Final symbols joined (after removing </w>) should equal original word
        final_symbols, _ = trace[-1]
        joined = "".join(final_symbols).replace("</w>", "")
        assert joined == "বাংলা"


# ── Encoder unit tests ────────────────────────────────────────────────────────


class TestBPEEncoder:
    @pytest.fixture
    def simple_encoder(self):
        """A small encoder with known merges."""
        vocab = Vocabulary.from_special_tokens()
        for ch in ["a", "b", "c", "ab", "abc", "</w>", "ab</w>", "abc</w>"]:
            vocab.add_token(ch)
        merges = [("a", "b"), ("ab", "c"), ("ab", "</w>"), ("abc", "</w>")]
        return BPEEncoder(vocabulary=vocab, merges=merges, end_of_word_suffix="</w>")

    def test_encode_abc(self, simple_encoder):
        simple_encoder.encode("abc")
        tokens = simple_encoder.tokenize("abc")
        # "abc" → ['abc</w>']  (merge a+b=ab, ab+c=abc, abc+</w>=abc</w>)
        assert "abc</w>" in tokens

    def test_encode_ab(self, simple_encoder):
        tokens = simple_encoder.tokenize("ab")
        assert "ab</w>" in tokens

    def test_cache_hit(self, simple_encoder):
        simple_encoder.encode("abc")
        simple_encoder.encode("abc")  # second call should hit cache
        info = simple_encoder.cache_info()
        assert "hits=1" in info

    def test_clear_cache(self, simple_encoder):
        simple_encoder.encode("abc")
        simple_encoder.clear_cache()
        info = simple_encoder.cache_info()
        assert "currsize=0" in info

    def test_unknown_char_uses_unk(self):
        vocab = Vocabulary.from_special_tokens()
        vocab.add_token("a")
        vocab.add_token("</w>")
        vocab.add_token("a</w>")
        encoder = BPEEncoder(vocabulary=vocab, merges=[("a", "</w>")], end_of_word_suffix="</w>")
        # 'z' is not in vocab → splits to ['z', '</w>'] → [unk_id, id_of_</w>]
        ids = encoder.encode("z")
        unk_id = vocab.unk_id
        eow_id = vocab.get_id("</w>")
        assert unk_id in ids
        # The </w> suffix token is a valid vocab token and should appear in output
        assert ids == [unk_id, eow_id]


# ── Decoder unit tests ────────────────────────────────────────────────────────


class TestBPEDecoder:
    @pytest.fixture
    def decoder_with_vocab(self):
        vocab = Vocabulary.from_special_tokens()
        vocab.add_token("বাংলা")
        vocab.add_token("দেশ</w>")
        return BPEDecoder(vocabulary=vocab, end_of_word_suffix="</w>")

    def test_decode_two_tokens(self, decoder_with_vocab):
        vocab = decoder_with_vocab._vocab
        tid1 = vocab.get_id("বাংলা")
        tid2 = vocab.get_id("দেশ</w>")
        result = decoder_with_vocab.decode([tid1, tid2])
        assert result == "বাংলাদেশ"

    def test_skip_special_tokens(self, decoder_with_vocab):
        vocab = decoder_with_vocab._vocab
        pad_id = vocab.pad_id
        tid = vocab.get_id("বাংলা")
        result = decoder_with_vocab.decode([pad_id, tid])
        assert "বাংলা" in result

    def test_unknown_id_skipped(self, decoder_with_vocab):
        result = decoder_with_vocab.decode([99999])
        assert result == ""

    def test_decode_tokens_strings(self, decoder_with_vocab):
        result = decoder_with_vocab.decode_tokens(["বাংলা", "দেশ</w>"])
        assert result == "বাংলাদেশ"


# ── _merge_symbols helper ─────────────────────────────────────────────────────


class TestMergeSymbols:
    def test_basic_merge(self):
        result = _merge_symbols(["a", "b", "c"], ("a", "b"))
        assert result == ["ab", "c"]

    def test_no_match(self):
        result = _merge_symbols(["a", "b", "c"], ("x", "y"))
        assert result == ["a", "b", "c"]

    def test_consecutive_pairs(self):
        # "a b a b" with pair (a, b) should become "ab ab"
        result = _merge_symbols(["a", "b", "a", "b"], ("a", "b"))
        assert result == ["ab", "ab"]

    def test_overlapping_avoided(self):
        # "a a a" with pair (a, a) → ["aa", "a"] (greedy, no overlap)
        result = _merge_symbols(["a", "a", "a"], ("a", "a"))
        assert result == ["aa", "a"]

    def test_single_element(self):
        result = _merge_symbols(["a"], ("a", "b"))
        assert result == ["a"]

    def test_bengali_merge(self):
        result = _merge_symbols(["ব", "া", "ং"], ("ব", "া"))
        assert result == ["বা", "ং"]
