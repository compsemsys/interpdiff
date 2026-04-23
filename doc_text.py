"""Word-level slicing for long documents (whitespace-separated words)."""

from __future__ import annotations

import re


_WS = re.compile(r"\s+")


def first_n_words(text: str, n: int = 200) -> str:
    """
    Return the first `n` words of `text` (split on any whitespace), stripped.

    Empty or very short strings return as much as exists after stripping.
    """
    if n <= 0:
        return ""
    if not text:
        return ""
    # Treat any run of whitespace as a boundary for consistent excerpt sizing.
    parts = _WS.split(text.strip())
    if not parts or parts == [""]:
        return ""
    return " ".join(parts[:n])
