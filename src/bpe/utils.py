"""General-purpose utilities for the Bengali BPE package.

This module contains helpers that do not belong to a single algorithmic
component: file I/O, timing, compression metrics, pretty-printing, and
progress-bar wrappers.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# ── Logging setup ─────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a concise, readable format.

    Args:
        level: Logging level string, e.g. ``"DEBUG"``, ``"INFO"``,
               ``"WARNING"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
    )


# ── Timing ────────────────────────────────────────────────────────────────────


@contextmanager
def timer(label: str = "elapsed") -> Generator[Dict[str, float], None, None]:
    """Context manager that measures elapsed wall-clock time.

    Usage::

        with timer("training") as t:
            do_something()
        print(f"Took {t['seconds']:.2f}s")

    Args:
        label: Human-readable label (used only for display).

    Yields:
        Dict with key ``"seconds"`` updated after the block exits.
    """
    result: Dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def format_duration(seconds: float) -> str:
    """Format *seconds* as a human-readable duration string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String like ``"1h 23m 45s"`` or ``"2m 05s"`` or ``"0.43s"``.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


# ── File I/O ──────────────────────────────────────────────────────────────────


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Read and return the full contents of a text file.

    Args:
        path: File path.
        encoding: Character encoding (default UTF-8).

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        UnicodeDecodeError: If the file cannot be decoded as *encoding*.
    """
    path = Path(path)
    return path.read_text(encoding=encoding)


def write_text_file(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path*, creating parent directories as needed.

    Args:
        path: Destination file path.
        text: Content to write.
        encoding: Character encoding (default UTF-8).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def load_json(path: Path) -> Any:
    """Load and return a JSON file.

    Args:
        path: Path to a UTF-8 JSON file.

    Returns:
        Deserialised Python object.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialise *obj* to a UTF-8 JSON file.

    Non-ASCII characters (e.g. Bengali script) are written as-is so the
    file stays human-readable.

    Args:
        obj: JSON-serialisable Python object.
        path: Destination file path.
        indent: Indentation level for pretty-printing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


# ── Corpus metrics ────────────────────────────────────────────────────────────


def count_words(text: str) -> int:
    """Count whitespace-separated tokens in *text*.

    This is a rough estimate of word count, suitable for compression metrics.

    Args:
        text: Input text.

    Returns:
        Token count.
    """
    return len(text.split())


def count_sentences(text: str) -> int:
    """Estimate sentence count using common sentence-ending punctuation.

    Handles ASCII ``.``, ``!``, ``?`` and the Bengali danda ``।``
    (U+0964) and double danda ``॥`` (U+0965).

    Args:
        text: Input text.

    Returns:
        Estimated number of sentences.
    """
    import re

    return len(re.findall(r"[.!?।॥]", text))


def compression_ratio(original_words: int, token_count: int) -> float:
    """Compute the BPE compression ratio.

    Defined as ``token_count / original_words``.  A value < 1 means BPE
    produced *fewer* tokens than words (aggressive merging). A value close
    to 1 means little compression.

    For a well-trained Bengali BPE model this is typically 1.5-3.0 (i.e.
    each word is represented by 1.5-3 subword tokens on average).

    Args:
        original_words: Number of whitespace-delimited words in the corpus.
        token_count: Number of BPE tokens produced from the same corpus.

    Returns:
        Compression ratio (float).
    """
    if original_words == 0:
        return 0.0
    return round(token_count / original_words, 4)


# ── Pretty-printing helpers ───────────────────────────────────────────────────


def format_number(n: int) -> str:
    """Format *n* with thousands separators (e.g. ``1_234_567``)."""
    return f"{n:,}"


def render_merge_step(
    step: int,
    symbols: List[str],
    merged_pair: Optional[tuple[str, str]] = None,
) -> str:
    """Render a single BPE merge step for the ``inspect`` command.

    Args:
        step: Step number (0 = initial state).
        symbols: Current list of symbols.
        merged_pair: The ``(A, B)`` pair that was just merged (or ``None``
                     for the initial display).

    Returns:
        Formatted string for terminal output.
    """
    joined = " + ".join(symbols)
    if step == 0:
        label = "Initial"
    elif merged_pair is not None:
        label = f"Merge ({merged_pair[0]!r} + {merged_pair[1]!r})"
    else:
        label = f"Step {step}"
    return f"  Step {step:>3}  [{label:>35}]  →  {joined}"


def print_banner(title: str, width: int = 60) -> None:
    """Print a simple titled banner to stdout.

    Args:
        title: Text to display in the banner.
        width: Total banner width in characters.
    """
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
