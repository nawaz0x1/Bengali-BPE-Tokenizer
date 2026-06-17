"""Command-line interface for the Bengali BPE Tokenizer.

Available sub-commands
~~~~~~~~~~~~~~~~~~~~~~

``train``    learn BPE merge rules from a corpus file.
``encode``   tokenise text using a trained model.
``decode``   reconstruct text from token IDs.
``stats``    display vocabulary and training statistics.
``inspect``  visualise the step-by-step merge sequence for one word.

Run ``python bpe.py <command> --help`` for per-command help.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, NoReturn

from .trainer import BPEModel, BPETrainer, TrainerConfig
from .tokenizer import BPETokenizer
from .utils import (
    compression_ratio,
    count_sentences,
    count_words,
    format_duration,
    format_number,
    print_banner,
    read_text_file,
    render_merge_step,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser construction
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all sub-commands."""

    parser = argparse.ArgumentParser(
        prog="bpe",
        description=(
            "Bengali BPE Tokenizer - Byte Pair Encoding from scratch.\n"
            "Supports Bengali (Bangla) and any UTF-8 language."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bpe.py train --file corpus.txt --vocab-size 8000 --output output/\n"
            "  python bpe.py encode --model output/ --text 'আমি বাংলাদেশে থাকি।'\n"
            "  python bpe.py decode --model output/ --tokens 245 891 7\n"
            "  python bpe.py stats  --model output/\n"
            "  python bpe.py inspect --model output/ --word বাংলাদেশ\n"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    _add_train_parser(sub)
    _add_encode_parser(sub)
    _add_decode_parser(sub)
    _add_stats_parser(sub)
    _add_inspect_parser(sub)

    return parser


def _add_train_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "train",
        help="Train a BPE model from a UTF-8 corpus file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Read a text corpus, learn BPE merge rules, and save the trained\n"
            "model (vocab.json, merges.txt, metadata.json) to --output.\n\n"
            "Example:\n"
            "  python bpe.py train --file corpus.txt --vocab-size 8000 --output output/"
        ),
    )
    p.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Path to UTF-8 training corpus.",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        metavar="N",
        help="Target vocabulary size (default: 8000).",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory to write model files to.",
    )
    p.add_argument(
        "--language",
        default="bengali",
        metavar="LANG",
        help="Language tag stored in metadata (default: bengali).",
    )
    p.add_argument(
        "--min-freq",
        type=int,
        default=2,
        metavar="N",
        help="Minimum word frequency for inclusion in training (default: 2).",
    )
    p.add_argument(
        "--normalization",
        default="NFC",
        choices=["NFC", "NFD", "NFKC", "NFKD"],
        help="Unicode normalisation form (default: NFC).",
    )
    p.add_argument(
        "--no-eow",
        action="store_true",
        help="Disable the end-of-word suffix </w>. Not recommended.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the tqdm progress bar.",
    )
    p.add_argument(
        "--export-vocab-csv",
        action="store_true",
        help="Also export vocabulary as vocab.csv.",
    )


def _add_encode_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "encode",
        help="Encode text using a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Tokenise text and output token IDs.\n\n"
            "Example:\n"
            "  python bpe.py encode --model output/ --text 'আমি বাংলাদেশে থাকি।'"
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="Model directory (written by 'train').",
    )
    p.add_argument(
        "--text",
        metavar="TEXT",
        help="Text to encode (UTF-8). Reads from stdin if omitted.",
    )
    p.add_argument(
        "--file",
        metavar="PATH",
        help="Encode entire file instead of --text.",
    )
    p.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print token strings alongside IDs.",
    )


def _add_decode_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "decode",
        help="Decode token IDs back to text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Reconstruct text from a space-separated list of token IDs.\n\n"
            "Example:\n"
            "  python bpe.py decode --model output/ --tokens 245 891 7"
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="Model directory.",
    )
    p.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        metavar="ID",
        help="One or more integer token IDs.",
    )


def _add_stats_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "stats",
        help="Display vocabulary and training statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Show detailed statistics for a trained model including vocabulary\n"
            "composition, top merge rules, and corpus metadata.\n\n"
            "Example:\n"
            "  python bpe.py stats --model output/"
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="Model directory.",
    )
    p.add_argument(
        "--top-merges",
        type=int,
        default=20,
        metavar="N",
        help="Number of top merge rules to display (default: 20).",
    )
    p.add_argument(
        "--export-csv",
        action="store_true",
        help="Export vocabulary to vocab.csv inside the model directory.",
    )
    p.add_argument(
        "--corpus-file",
        metavar="PATH",
        help=(
            "Provide original corpus to compute compression ratio and "
            "tokens-per-word / tokens-per-sentence statistics."
        ),
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Plot merge frequency curve (requires matplotlib).",
    )


def _add_inspect_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "inspect",
        help="Visualise BPE merge steps for a single word.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Show how a word is broken down step-by-step by BPE merges.\n\n"
            "Example:\n"
            "  python bpe.py inspect --model output/ --word বাংলাদেশ"
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="DIR",
        help="Model directory.",
    )
    p.add_argument(
        "--word",
        required=True,
        metavar="WORD",
        help="Single word to inspect (no spaces).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> None:
    """Handler for the ``train`` sub-command."""
    corpus_path = Path(args.file)
    if not corpus_path.exists():
        _die(f"Corpus file not found: {corpus_path}")

    print_banner(f"BPE Training - {corpus_path.name}")

    text = read_text_file(corpus_path)
    print(f"  Corpus size : {format_number(len(text))} characters")
    print(f"  Word count  : {format_number(count_words(text))}")

    config = TrainerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        end_of_word_suffix="" if args.no_eow else "</w>",
        normalization=args.normalization,
        language=args.language,
        show_progress=not args.no_progress,
    )

    trainer = BPETrainer(config)
    model = trainer.train(text)

    output_dir = Path(args.output)
    model.save(output_dir)

    if args.export_vocab_csv:
        csv_path = output_dir / "vocab.csv"
        model.vocabulary.export_csv(csv_path)
        print(f"  Vocabulary CSV exported → {csv_path}")

    print()
    print(f"  Final vocab size : {format_number(len(model.vocabulary))}")
    print(f"  Merge operations : {format_number(len(model.merges))}")
    print(f"  Training time    : {format_duration(model.training_time)}")
    print(f"  Output directory : {output_dir.resolve()}")


def _cmd_encode(args: argparse.Namespace) -> None:
    """Handler for the ``encode`` sub-command."""
    model_dir = Path(args.model)

    if args.file:
        text = read_text_file(Path(args.file))
    elif args.text:
        text = args.text
    else:
        print("Reading from stdin (Ctrl-D to finish)…", file=sys.stderr)
        text = sys.stdin.read()

    tokenizer = BPETokenizer(model_dir)

    if args.show_tokens:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        pairs = list(zip(tokens, ids))
        print(f"{'Token':<25} {'ID':>8}")
        print("─" * 35)
        for tok, tid in pairs:
            print(f"{tok:<25} {tid:>8}")
        print(f"\nTotal tokens: {len(ids)}")
    else:
        ids = tokenizer.encode(text)
        print(" ".join(str(i) for i in ids))


def _cmd_decode(args: argparse.Namespace) -> None:
    """Handler for the ``decode`` sub-command."""
    model_dir = Path(args.model)

    if not args.tokens:
        _die("Please provide token IDs via --tokens ID [ID ...]")

    tokenizer = BPETokenizer(model_dir)
    text = tokenizer.decode(args.tokens)
    print(text)


def _cmd_stats(args: argparse.Namespace) -> None:
    """Handler for the ``stats`` sub-command."""
    model_dir = Path(args.model)
    model = BPEModel.load(model_dir)

    print_banner(f"Model Statistics - {model_dir.resolve()}")

    # ── Basic info ────────────────────────────────────────────────────────────
    print("\n  TRAINING CONFIGURATION")
    print(f"    Language           : {model.config.language}")
    print(f"    Normalisation      : {model.config.normalization}")
    print(f"    End-of-word suffix : {model.config.end_of_word_suffix!r}")
    print(f"    Min word frequency : {model.config.min_frequency}")

    print("\n  VOCABULARY")
    print(f"    Total size         : {format_number(len(model.vocabulary))}")
    print(f"    Special tokens     : {sorted(model.vocabulary.special_tokens)}")

    tls = model.vocabulary.token_length_stats()
    if tls:
        print(f"    Avg token length   : {tls['mean']:.2f} characters")
        print(f"    Longest token      : {tls['max']} characters")
        print(f"    Shortest token     : {tls['min']} characters")

    print("\n  LONGEST TOKENS")
    for tok, length in model.vocabulary.longest_tokens(10):
        print(f"    {tok:<30} ({length} chars)")

    print("\n  TRAINING RUN")
    print(f"    Corpus size        : {format_number(model.corpus_size)} characters")
    print(f"    Merge operations   : {format_number(len(model.merges))}")
    print(f"    Training time      : {format_duration(model.training_time)}")

    us = model.unicode_stats
    if us:
        print("\n  UNICODE STATISTICS (training corpus)")
        print(f"    Bengali chars      : {format_number(us.get('bengali_chars', 0))}")
        print(f"    Latin chars        : {format_number(us.get('latin_chars', 0))}")
        print(f"    Digits             : {format_number(us.get('digits', 0))}")
        print(f"    Combining marks    : {format_number(us.get('combining_marks', 0))}")
        print(f"    Zero-width chars   : {format_number(us.get('zero_width_chars', 0))}")
        print(f"    Unique codepoints  : {format_number(us.get('unique_codepoints', 0))}")

    # ── Top merges ────────────────────────────────────────────────────────────
    n_merges = args.top_merges
    history = model.merge_freq_history[:n_merges]
    if history:
        print(f"\n  TOP {n_merges} MERGE RULES (by training frequency)")
        print(f"    {'#':<6} {'Left':<20} {'Right':<20} {'Freq':>10}")
        print("    " + "─" * 60)
        for rank, ((left, right), freq) in enumerate(history, 1):
            print(f"    {rank:<6} {left:<20} {right:<20} {format_number(freq):>10}")

    # ── Corpus-level compression stats ────────────────────────────────────────
    if args.corpus_file:
        corpus_path = Path(args.corpus_file)
        if not corpus_path.exists():
            print(f"\n  Warning: corpus file not found: {corpus_path}", file=sys.stderr)
        else:
            print("\n  COMPRESSION STATISTICS")
            tokenizer = BPETokenizer(model_dir)
            corpus_text = read_text_file(corpus_path)
            n_words = count_words(corpus_text)
            n_sents = count_sentences(corpus_text)
            ids = tokenizer.encode(corpus_text)
            n_tokens = len(ids)

            avg_per_word = round(n_tokens / n_words, 3) if n_words else 0
            avg_per_sent = round(n_tokens / n_sents, 3) if n_sents else 0
            ratio = compression_ratio(n_words, n_tokens)

            print(f"    Words in corpus    : {format_number(n_words)}")
            print(f"    Sentences (est.)   : {format_number(n_sents)}")
            print(f"    Total tokens       : {format_number(n_tokens)}")
            print(f"    Tokens / word      : {avg_per_word:.2f}")
            print(f"    Tokens / sentence  : {avg_per_sent:.2f}")
            print(f"    Compression ratio  : {ratio:.4f}")

    # ── Optional CSV export ───────────────────────────────────────────────────
    if args.export_csv:
        csv_path = model_dir / "vocab.csv"
        model.vocabulary.export_csv(csv_path)
        print(f"\n  Vocabulary exported → {csv_path}")

    # ── Optional merge-frequency plot ─────────────────────────────────────────
    if args.plot:
        _plot_merge_frequencies(model)


def _cmd_inspect(args: argparse.Namespace) -> None:
    """Handler for the ``inspect`` sub-command."""
    model_dir = Path(args.model)
    word = args.word.strip()

    if not word:
        _die("--word must be a non-empty string.")
    if " " in word:
        _die("--word must be a single word without spaces.")

    print_banner(f"BPE Inspection - {word!r}")

    tokenizer = BPETokenizer(model_dir)
    trace = tokenizer.inspect(word)

    print()
    for step, (symbols, pair) in enumerate(trace):
        print(render_merge_step(step, symbols, pair))

    final_symbols = trace[-1][0]
    ids = [
        tokenizer.model.vocabulary.get_id(s, tokenizer.model.vocabulary.unk_id)
        for s in final_symbols
    ]
    print()
    print(f"  Final tokens : {final_symbols}")
    print(f"  Token IDs    : {ids}")
    print(f"  Merge steps  : {len(trace) - 1}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional visualisation
# ─────────────────────────────────────────────────────────────────────────────


def _plot_merge_frequencies(model: BPEModel) -> None:
    """Plot the frequency of merged pairs over training iterations."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print(
            "\n  matplotlib is not installed. " "Run: pip install matplotlib",
            file=sys.stderr,
        )
        return

    freqs = [freq for _, freq in model.merge_freq_history]
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(freqs) + 1), freqs, linewidth=0.8, color="steelblue")
    plt.yscale("log")
    plt.xlabel("Merge iteration")
    plt.ylabel("Pair frequency (log scale)")
    plt.title(f"BPE Merge Frequency - {model.config.language}")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _die(message: str) -> NoReturn:
    """Print an error message and exit with code 1."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def main(argv: List[str] | None = None) -> None:
    """Main entry point for the ``bpe`` CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    dispatch = {
        "train": _cmd_train,
        "encode": _cmd_encode,
        "decode": _cmd_decode,
        "stats": _cmd_stats,
        "inspect": _cmd_inspect,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except FileNotFoundError as exc:
        _die(str(exc))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
