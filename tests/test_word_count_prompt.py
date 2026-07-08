from __future__ import annotations

import pytest

from local_llm_utils import build_explain_prompt
from run_categorized_corpus import (
    WORD_COUNT_INSTRUCTION,
    resolve_instruction_args,
)


class _PlainTokenizer:
    chat_template = None


def test_build_explain_prompt_substitutes_word_count() -> None:
    tok = _PlainTokenizer()
    prompt = build_explain_prompt(
        tok,
        WORD_COUNT_INSTRUCTION,
        title="Natural history",
        word_count=200,
    )
    assert prompt == "In 200 words, explain the following: Natural history\n\n"


def test_build_explain_prompt_custom_word_count_placeholder() -> None:
    tok = _PlainTokenizer()
    prompt = build_explain_prompt(
        tok,
        "Summarize in {word_count} words: {title}",
        title="Algebra",
        word_count=50,
    )
    assert prompt == "Summarize in 50 words: Algebra\n\n"


def test_resolve_instruction_args_word_count_prompt_requires_max_generated_words() -> None:
    with pytest.raises(ValueError, match="requires --max_generated_words"):
        resolve_instruction_args(
            instruction="Explain the following: {title}",
            max_generated_words=None,
            word_count_prompt=True,
        )


def test_resolve_instruction_args_custom_instruction_with_word_count_requires_max() -> None:
    with pytest.raises(ValueError, match="uses \\{word_count\\}"):
        resolve_instruction_args(
            instruction="In {word_count} words: {title}",
            max_generated_words=None,
            word_count_prompt=False,
        )


def test_resolve_instruction_args_word_count_prompt_sets_preset() -> None:
    instruction, word_count_prompt = resolve_instruction_args(
        instruction="Explain the following: {title}",
        max_generated_words=250,
        word_count_prompt=True,
    )
    assert instruction == WORD_COUNT_INSTRUCTION
    assert word_count_prompt is True


def test_resolve_instruction_args_custom_word_count_instruction() -> None:
    instruction, word_count_prompt = resolve_instruction_args(
        instruction="In {word_count} words: {excerpt}",
        max_generated_words=100,
        word_count_prompt=False,
    )
    assert instruction == "In {word_count} words: {excerpt}"
    assert word_count_prompt is False


def test_format_cli_command_quotes_spaces_for_cmd() -> None:
    from run_categorized_corpus import format_cli_command

    cmd = format_cli_command(
        [r"python.exe", r"run_categorized_corpus.py", "--corpus", r"F:\code\Independent Study\data.jsonl"]
    )
    assert cmd == (
        r'python.exe run_categorized_corpus.py --corpus "F:\code\Independent Study\data.jsonl"'
    )


def test_format_cli_command_cmd_embedded_quote_doubling() -> None:
    from run_categorized_corpus import format_cli_command

    cmd = format_cli_command([r"script.py", r'say "hi" there'])
    assert cmd == r'script.py "say ""hi"" there"'
