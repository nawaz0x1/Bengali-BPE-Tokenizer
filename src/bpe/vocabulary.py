"""Vocabulary management for BPE.

A :class:`Vocabulary` holds the bidirectional mapping between token strings
and integer IDs.  Special tokens (``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``)
are always assigned the **lowest** IDs so that they remain stable across
incremental vocabulary updates.

File format
-----------
Vocabularies are serialised as JSON::

    {
      "token_to_id": {"<pad>": 0, "<unk>": 1, "ব": 4, "বা": 156, ...},
      "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"]
    }

The ``id_to_token`` reverse mapping is reconstructed on load and is never
written to disk (it is always derived from ``token_to_id``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

# ── Special tokens ────────────────────────────────────────────────────────────

PAD_TOKEN: str = "<pad>"   # padding / ignored position
UNK_TOKEN: str = "<unk>"   # unknown / out-of-vocabulary token
BOS_TOKEN: str = "<bos>"   # beginning-of-sequence marker
EOS_TOKEN: str = "<eos>"   # end-of-sequence marker

DEFAULT_SPECIAL_TOKENS: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ── Vocabulary class ──────────────────────────────────────────────────────────


@dataclass
class Vocabulary:
    """Bidirectional token ↔ ID mapping.

    Attributes:
        token_to_id: Maps token string → integer ID.
        id_to_token: Maps integer ID → token string (always kept in sync).
        special_tokens: Set of special token strings (e.g. ``"<pad>"``).
    """

    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    special_tokens: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # If only one direction is supplied, derive the other.
        if self.token_to_id and not self.id_to_token:
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        elif self.id_to_token and not self.token_to_id:
            self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_special_tokens(
        cls, special_tokens: Optional[List[str]] = None
    ) -> "Vocabulary":
        """Create a :class:`Vocabulary` pre-populated with special tokens.

        Args:
            special_tokens: List of special token strings. Defaults to
                ``DEFAULT_SPECIAL_TOKENS``.

        Returns:
            New :class:`Vocabulary` instance.
        """
        vocab = cls()
        for tok in special_tokens or DEFAULT_SPECIAL_TOKENS:
            vocab.add_token(tok, special=True)
        return vocab

    # ── Core API ──────────────────────────────────────────────────────────────

    def add_token(self, token: str, special: bool = False) -> int:
        """Add *token* to the vocabulary and return its integer ID.

        If the token already exists, its existing ID is returned unchanged
        (i.e. this method is idempotent).

        Args:
            token: The token string to add.
            special: Whether to mark this as a special token.

        Returns:
            Integer ID for *token*.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        if special:
            self.special_tokens.add(token)
        return token_id

    def get_id(self, token: str, default: Optional[int] = None) -> Optional[int]:
        """Return the ID for *token*, or *default* if not in vocabulary."""
        return self.token_to_id.get(token, default)

    def get_token(self, token_id: int, default: Optional[str] = None) -> Optional[str]:
        """Return the token string for *token_id*, or *default* if not found."""
        return self.id_to_token.get(token_id, default)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __iter__(self) -> Iterator[str]:
        """Iterate over all token strings in insertion order."""
        return iter(self.token_to_id)

    @property
    def size(self) -> int:
        """Number of tokens (including special tokens)."""
        return len(self.token_to_id)

    # ── Named-token shortcuts ─────────────────────────────────────────────────

    @property
    def unk_id(self) -> Optional[int]:
        """ID of the ``<unk>`` token, or ``None`` if not present."""
        return self.token_to_id.get(UNK_TOKEN)

    @property
    def pad_id(self) -> Optional[int]:
        """ID of the ``<pad>`` token, or ``None`` if not present."""
        return self.token_to_id.get(PAD_TOKEN)

    @property
    def bos_id(self) -> Optional[int]:
        """ID of the ``<bos>`` token, or ``None`` if not present."""
        return self.token_to_id.get(BOS_TOKEN)

    @property
    def eos_id(self) -> Optional[int]:
        """ID of the ``<eos>`` token, or ``None`` if not present."""
        return self.token_to_id.get(EOS_TOKEN)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise vocabulary to a UTF-8 JSON file.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        path = Path(path)
        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": sorted(self.special_tokens),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Deserialise a vocabulary from a JSON file.

        Args:
            path: Path to a vocabulary JSON file created by :meth:`save`.

        Returns:
            Loaded :class:`Vocabulary` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the JSON is missing the ``token_to_id`` field.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        token_to_id: Dict[str, int] = data["token_to_id"]
        special_tokens: Set[str] = set(data.get("special_tokens", []))
        id_to_token: Dict[int, str] = {v: k for k, v in token_to_id.items()}
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=special_tokens,
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def token_length_stats(self) -> Dict[str, float]:
        """Compute statistics about token character lengths.

        Excludes special tokens from the calculation.

        Returns:
            Dict with keys ``mean``, ``max``, ``min``, ``median``,
            ``total_tokens``.
        """
        lengths = [
            len(t) for t in self.token_to_id if t not in self.special_tokens
        ]
        if not lengths:
            return {}
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        return {
            "mean": round(sum(lengths) / n, 4),
            "max": lengths_sorted[-1],
            "min": lengths_sorted[0],
            "median": lengths_sorted[n // 2],
            "total_tokens": n,
        }

    def longest_tokens(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the *n* longest tokens (by Unicode character count).

        Special tokens are excluded.

        Args:
            n: Number of tokens to return.

        Returns:
            List of ``(token, length)`` tuples sorted by length descending.
        """
        return sorted(
            [(t, len(t)) for t in self.token_to_id if t not in self.special_tokens],
            key=lambda x: -x[1],
        )[:n]

    def export_csv(self, path: Path) -> None:
        """Export vocabulary to a CSV file.

        Columns: ``id``, ``token``, ``length``, ``is_special``.

        Args:
            path: Destination CSV file path.
        """
        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "token", "length", "is_special"])
            for token, token_id in sorted(
                self.token_to_id.items(), key=lambda kv: kv[1]
            ):
                writer.writerow(
                    [token_id, token, len(token), token in self.special_tokens]
                )
