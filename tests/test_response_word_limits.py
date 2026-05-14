"""Unit tests for response word counting / truncation (generation helpers)."""

from __future__ import annotations

from local_llm_utils import response_word_count, truncate_completion_to_max_generated_words


def test_response_word_count_hyphenated_single_word() -> None:
    assert response_word_count("well-known co-op") == 2


def test_response_word_count_whitespace() -> None:
    assert response_word_count("  a\tb\nc  ") == 3


def test_response_word_count_empty() -> None:
    assert response_word_count("") == 0
    assert response_word_count("   \n") == 0


def test_truncate_noop_when_under_limit() -> None:
    assert truncate_completion_to_max_generated_words("one two", 5) == "one two"


def test_truncate_hard_cap() -> None:
    assert truncate_completion_to_max_generated_words("a b c d", 3) == "a b c"


def test_truncate_preserves_hyphenated_units() -> None:
    s = "first well-known second"
    assert truncate_completion_to_max_generated_words(s, 2) == "first well-known"
