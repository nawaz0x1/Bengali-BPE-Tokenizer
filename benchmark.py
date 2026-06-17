#!/usr/bin/env python3
"""Benchmark: Bengali BPE Tokenizer vs tiktoken (GPT-2 / GPT-4o).

Usage:
    python benchmark.py                        # uses examples/corpus.txt & output/
    python benchmark.py --corpus path/to/file  # custom corpus
    python benchmark.py --model  path/to/dir   # custom model directory
"""

import argparse
import sys
import re
from pathlib import Path
from collections import defaultdict

# ── make src/ importable when run directly ──────────────────────────────────
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bpe import BPETokenizer  # noqa: E402

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken not found. Install it with:  pip install tiktoken")

# ── Unicode Bengali range ────────────────────────────────────────────────────
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]+")


# ────────────────────────────────────────────────────────────────────────────
def _tokenize_tiktoken(enc, text: str) -> list[int]:
    return enc.encode(text)


def _words(text: str) -> list[str]:
    """Return whitespace-split words (non-empty)."""
    return [w for w in text.split() if w]


def _bengali_words(text: str) -> list[str]:
    """Return tokens that are purely Bengali script."""
    return [w for w in _words(text) if _BENGALI_RE.fullmatch(w)]


# ── per-word token counts ────────────────────────────────────────────────────
def _word_token_counts(tokenizer, enc_gpt2, enc_cl100k, words: list[str]):
    """Return dict word → (bengali_bpe_count, gpt2_count, cl100k_count)."""
    results = {}
    for w in words:
        bpe_count = len(tokenizer.tokenize(w))
        gpt2_count = len(enc_gpt2.encode(w))
        cl100k_count = len(enc_cl100k.encode(w))
        results[w] = (bpe_count, gpt2_count, cl100k_count)
    return results


# ── helpers ──────────────────────────────────────────────────────────────────
def _bar(value: float, best: float, width: int = 20) -> str:
    filled = round(value / best * width) if best else 0
    return "█" * filled + "░" * (width - filled)


def _ratio_label(our: int, theirs: int) -> str:
    if theirs == 0:
        return "N/A"
    r = theirs / our
    return f"{r:.2f}× fewer" if r >= 1 else f"{1/r:.2f}× more"


# ── main benchmark ───────────────────────────────────────────────────────────
def run_benchmark(corpus_path: Path, model_dir: Path):
    print("\n" + "═" * 62)
    print("  Bengali BPE  vs  tiktoken  -  Benchmark")
    print("═" * 62)

    # ── load corpus ──────────────────────────────────────────────────────────
    text = corpus_path.read_text(encoding="utf-8")
    char_count = len(text)
    word_list = _words(text)
    bengali_word_list = _bengali_words(text)
    unique_bengali = list(dict.fromkeys(bengali_word_list))  # preserve order

    print(f"\n  Corpus : {corpus_path}")
    print(f"  Chars  : {char_count:,}")
    print(f"  Words  : {len(word_list):,}  ({len(bengali_word_list):,} Bengali)")
    print(f"  Unique Bengali words: {len(unique_bengali):,}")

    # ── load tokenizers ──────────────────────────────────────────────────────
    tok = BPETokenizer(str(model_dir))
    enc_gpt2    = tiktoken.get_encoding("gpt2")
    enc_cl100k  = tiktoken.get_encoding("cl100k_base")   # GPT-4 / ChatGPT

    print(f"\n  Bengali-BPE vocab size : {tok.vocab_size:,}")
    print(f"  GPT-2 vocab size       : {enc_gpt2.n_vocab:,}")
    print(f"  cl100k (GPT-4) vocab   : {enc_cl100k.n_vocab:,}")

    # ── full-corpus token counts ─────────────────────────────────────────────
    bpe_ids    = tok.encode(text)
    gpt2_ids   = _tokenize_tiktoken(enc_gpt2,   text)
    cl100k_ids = _tokenize_tiktoken(enc_cl100k, text)

    bpe_n    = len(bpe_ids)
    gpt2_n   = len(gpt2_ids)
    cl100k_n = len(cl100k_ids)

    print("\n" + "─" * 62)
    print("  FULL-CORPUS TOKEN COUNT")
    print("─" * 62)
    best_n = min(bpe_n, gpt2_n, cl100k_n)
    print(f"  Bengali-BPE  {bpe_n:>7,}  {_bar(bpe_n, best_n)}  (ours)")
    print(f"  GPT-2        {gpt2_n:>7,}  {_bar(gpt2_n, best_n)}")
    print(f"  cl100k/GPT-4 {cl100k_n:>7,}  {_bar(cl100k_n, best_n)}")

    # ── compression ratio (chars / tokens) ───────────────────────────────────
    bpe_cpr    = char_count / bpe_n
    gpt2_cpr   = char_count / gpt2_n
    cl100k_cpr = char_count / cl100k_n

    print("\n" + "─" * 62)
    print("  COMPRESSION RATIO  (chars per token - higher = better)")
    print("─" * 62)
    best_cpr = max(bpe_cpr, gpt2_cpr, cl100k_cpr)
    print(f"  Bengali-BPE  {bpe_cpr:>6.2f}  {_bar(bpe_cpr, best_cpr)}  (ours)")
    print(f"  GPT-2        {gpt2_cpr:>6.2f}  {_bar(gpt2_cpr, best_cpr)}")
    print(f"  cl100k/GPT-4 {cl100k_cpr:>6.2f}  {_bar(cl100k_cpr, best_cpr)}")

    # ── average tokens per Bengali word ──────────────────────────────────────
    if unique_bengali:
        counts = _word_token_counts(tok, enc_gpt2, enc_cl100k, unique_bengali)
        bpe_avg    = sum(v[0] for v in counts.values()) / len(counts)
        gpt2_avg   = sum(v[1] for v in counts.values()) / len(counts)
        cl100k_avg = sum(v[2] for v in counts.values()) / len(counts)

        print("\n" + "─" * 62)
        print("  AVG TOKENS PER BENGALI WORD  (lower = better)")
        print("─" * 62)
        best_avg = min(bpe_avg, gpt2_avg, cl100k_avg)
        print(f"  Bengali-BPE  {bpe_avg:>5.2f}  {_bar(best_avg, bpe_avg)}  (ours)")
        print(f"  GPT-2        {gpt2_avg:>5.2f}  {_bar(best_avg, gpt2_avg)}")
        print(f"  cl100k/GPT-4 {cl100k_avg:>5.2f}  {_bar(best_avg, cl100k_avg)}")

        # ── per-word spotlight: 15 most "painful" GPT-2 words ────────────────
        sorted_words = sorted(counts.items(), key=lambda kv: kv[1][1], reverse=True)
        spotlight = sorted_words[:15]

        print("\n" + "─" * 62)
        print("  TOP-15 WORDS - GPT-2 TOKEN COUNT (worst cases for GPT-2)")
        print("─" * 62)
        print(f"  {'Word':<22} {'Ours':>5}  {'GPT-2':>6}  {'GPT-4':>6}  Savings")
        print(f"  {'─'*22} {'─'*5}  {'─'*6}  {'─'*6}  {'─'*15}")
        for word, (bpe_c, g2_c, cl_c) in spotlight:
            print(f"  {word:<22} {bpe_c:>5}  {g2_c:>6}  {cl_c:>6}  "
                  f"{_ratio_label(bpe_c, g2_c)} vs GPT-2")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  SUMMARY")
    print("═" * 62)
    print(f"  vs GPT-2   : {_ratio_label(bpe_n, gpt2_n)} tokens on full corpus")
    print(f"  vs GPT-4   : {_ratio_label(bpe_n, cl100k_n)} tokens on full corpus")
    if unique_bengali:
        print(f"  vs GPT-2   : {_ratio_label(bpe_avg, gpt2_avg)} tokens/word (avg Bengali)")
        print(f"  vs GPT-4   : {_ratio_label(bpe_avg, cl100k_avg)} tokens/word (avg Bengali)")
    print("═" * 62 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark Bengali BPE vs tiktoken")
    parser.add_argument("--corpus", default="examples/corpus.txt",
                        help="Path to UTF-8 corpus file (default: examples/corpus.txt)")
    parser.add_argument("--model", default="output/",
                        help="Path to trained model directory (default: output/)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    model_dir   = Path(args.model)

    if not corpus_path.exists():
        sys.exit(f"Corpus not found: {corpus_path}")
    if not model_dir.exists():
        sys.exit(f"Model directory not found: {model_dir}")

    run_benchmark(corpus_path, model_dir)


if __name__ == "__main__":
    main()
