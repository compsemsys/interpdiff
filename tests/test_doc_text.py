"""Tests for doc_text.first_n_words."""

from __future__ import annotations

import pytest

from doc_text import first_n_words


def test_first_n_words_basic():
    assert first_n_words("a b c d e", 3) == "a b c"


def test_first_n_words_strips_and_collapses():
    assert first_n_words("  hello   world  foo  ", 2) == "hello world"


def test_empty():
    assert first_n_words("", 10) == ""
    assert first_n_words("   ", 10) == ""


def test_zero_n():
    assert first_n_words("hello world", 0) == ""


def test_longer_than_text():
    assert first_n_words("one two", 200) == "one two"
