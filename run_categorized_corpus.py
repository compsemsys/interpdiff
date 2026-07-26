"""
Categorized corpus pipeline (repeatable, all artifacts under ``--out_dir``).

Stages (use ``--stage`` to run one at a time; later stages require the same
``--out_dir`` and compatible ``--models`` as recorded in ``pipeline_config.json``):

1. **init** — excerpts JSONL, corpus copy, ``pipeline_config.json``, ``pipeline_state.json``.
2. **generate** — causal LM outputs per model for each configured generation task
   (the explain-style ``response`` task, plus a ``summary`` task when ``--summarize`` is set),
   unless ``skip_generate`` in config. Each task has its own word cap (``--max_generated_words``
   for ``response``, ``--summarize_words`` for ``summary``), or per-excerpt caps when
   ``--match_abstract_length`` is set; ``--max_new_tokens`` is a shared ceiling.
   Word caps early-stop once the decoded **model completion** reaches N whitespace-delimited words
   (still capped by ``--max_new_tokens`` and EOS); hyphenated forms count as one word.
3. **embed_excerpts** — token + aggregated npy under ``excerpts/<slug>/``.
4. **embed_responses** — token + aggregated npy for each generation task under its dir
   (``responses/<slug>/``, ``summaries/<slug>/``, …), tagged with the task's segment label.
5. **cka** — pairwise linear CKA under ``cka/`` across all model × segment pairs.

Default ``--stage all`` runs the full pipeline in one process. With ``--resume``
(default: on for single stages), existing outputs are skipped unless ``--overwrite``.

Key functions:
- ``run_stage_init``: validates/records run config and writes excerpt corpus artifacts.
- ``run_stage_generate``: generates model responses from excerpt-derived prompts.
- ``run_stage_embed_excerpts`` / ``run_stage_embed_responses``: writes token + pooled embeddings.
- ``run_stage_cka``: computes pairwise CKA outputs and optional document-by-category matrix.
- ``main``: CLI parsing, staged/full-run dispatch, and resume/overwrite behavior.

``pipeline_embed_explain_cka.py`` is the same CLI.

Example (pause between heavy steps):
  python run_categorized_corpus.py --corpus data/corpus.jsonl --out_dir outputs/run_a \\
    --models F:/m/gemma-3-1b-it F:/m/Qwen3.5-0.8B --stage init
  python run_categorized_corpus.py --corpus data/corpus.jsonl --out_dir outputs/run_a \\
    --models F:/m/gemma-3-1b-it F:/m/Qwen3.5-0.8B --stage generate
  python run_categorized_corpus.py --corpus data/corpus.jsonl --out_dir outputs/run_a \\
    --models F:/m/gemma-3-1b-it F:/m/Qwen3.5-0.8B --stage embed_excerpts
  python run_categorized_corpus.py --corpus data/corpus.jsonl --out_dir outputs/run_a \\
    --models F:/m/gemma-3-1b-it F:/m/Qwen3.5-0.8B --stage embed_responses
  python run_categorized_corpus.py --corpus data/corpus.jsonl --out_dir outputs/run_a \\
    --models F:/m/gemma-3-1b-it F:/m/Qwen3.5-0.8B --stage cka
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

from cka_word_embeddings import (
    cka_row_slice_file_suffix,
    linear_cka_chunked,
    pairwise_linear_cka_from_paths,
    parse_doc_ids_arg,
    parse_meta_filter_arg,
)
from doc_text import first_n_words
from doc_merge_agnostic import merge_token_embeddings_to_docs
from embed_corpus import embed_labeled_texts
from local_llm_utils import (
    build_explain_prompt,
    generate_completion,
    load_causal_lm,
    response_word_count,
    sanitize_model_slug,
)
from token_embed_utils import pick_device
from word_merge_agnostic import merge_token_embeddings_to_words

STAGES_ORDER = (
    "init",
    "generate",
    "embed_excerpts",
    "embed_responses",
    "cka",
)
# The config captures immutable run inputs; state tracks which stages completed.
CONFIG_NAME = "pipeline_config.json"
STATE_NAME = "pipeline_state.json"


def load_corpus_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_token_npy(
    path: str, embeddings: np.ndarray, token_meta: list[dict[str, Any]]
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, {"embeddings": embeddings, "token_meta": token_meta}, allow_pickle=True)


def save_word_npy(path: str, embeddings: np.ndarray, meta: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, {"embeddings": embeddings, "meta": meta}, allow_pickle=True)


def _instruction_placeholders(s: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", s))


_ALLOWED_INSTRUCTION_KEYS = frozenset({"excerpt", "title", "word_count"})
_DEFAULT_INSTRUCTION = "Explain the following: {title}"
WORD_COUNT_INSTRUCTION = "In {word_count} words, explain the following: {title}"
# Preset for the additional summarization task (segment "summary").
SUMMARIZE_INSTRUCTION = "Summarize the following in {word_count} words: {excerpt}"
# Segment label / output subdirectory for the primary explain-style generation task.
RESPONSE_SEGMENT = "response"
RESPONSE_SUBDIR = "responses"
SUMMARY_SEGMENT = "summary"
SUMMARY_SUBDIR = "summaries"


def _validate_instruction_template(
    instruction: str,
    *,
    flag: str,
    word_count: int | None,
    match_abstract_length: bool = False,
) -> None:
    """Shared placeholder validation for any generation-task instruction template."""
    keys = _instruction_placeholders(instruction)
    bad = keys - _ALLOWED_INSTRUCTION_KEYS
    if bad:
        allowed = ", ".join(sorted(_ALLOWED_INSTRUCTION_KEYS))
        raise ValueError(f"{flag} has unknown placeholder(s): {bad}; allowed: {allowed}")
    if not keys & {"excerpt", "title"}:
        raise ValueError(f"{flag} must contain at least one of {{excerpt}}, {{title}}")
    if "word_count" in keys and word_count is None and not match_abstract_length:
        raise ValueError(f"{flag} uses {{word_count}} but no word count is set")


def resolve_instruction_args(
    *,
    instruction: str,
    max_generated_words: int | None,
    word_count_prompt: bool,
    match_abstract_length: bool = False,
) -> tuple[str, bool]:
    """Return the effective instruction template and word_count_prompt flag."""
    if word_count_prompt:
        if max_generated_words is None and not match_abstract_length:
            raise ValueError(
                "--word_count_prompt requires --max_generated_words or --match_abstract_length"
            )
        instruction = WORD_COUNT_INSTRUCTION
    keys = _instruction_placeholders(instruction)
    bad = keys - _ALLOWED_INSTRUCTION_KEYS
    if bad:
        allowed = ", ".join(sorted(_ALLOWED_INSTRUCTION_KEYS))
        raise ValueError(
            f"--instruction has unknown placeholder(s): {bad}; allowed: {allowed}"
        )
    if not keys & {"excerpt", "title"}:
        raise ValueError("--instruction must contain at least one of {excerpt}, {title}")
    if "word_count" in keys and max_generated_words is None and not match_abstract_length:
        raise ValueError(
            "--instruction uses {word_count} but --max_generated_words is not set "
            "(or pass --match_abstract_length)"
        )
    return instruction, word_count_prompt


def build_generation_tasks(
    *,
    instruction: str,
    max_generated_words: int | None,
    word_count_prompt: bool,
    summarize: bool = False,
    summarize_instruction: str = SUMMARIZE_INSTRUCTION,
    summarize_words: int | None = None,
    match_abstract_length: bool = False,
) -> list[dict[str, Any]]:
    """Build the ordered list of generation tasks frozen into ``pipeline_config.json``.

    The first task is always the primary explain-style ``response`` task (unchanged
    behavior). When ``summarize`` is set, an additional ``summary`` task is appended.
    Each task is self-describing: ``name`` doubles as the CKA segment label and
    ``subdir`` is where its ``responses.jsonl`` / embeddings live under ``out_dir``.
    ``summarize_words`` defaults to ``max_generated_words`` (the primary task's cap).
    When ``match_abstract_length`` is set, task ``max_generated_words`` values are
    frozen as ``None`` and generation uses each excerpt's word count instead.
    """
    tasks: list[dict[str, Any]] = [
        {
            "name": RESPONSE_SEGMENT,
            "subdir": RESPONSE_SUBDIR,
            "instruction": instruction,
            "max_generated_words": None if match_abstract_length else max_generated_words,
            "word_count_prompt": bool(word_count_prompt),
        }
    ]
    if summarize:
        if match_abstract_length:
            sw = None
        else:
            sw = summarize_words if summarize_words is not None else max_generated_words
        _validate_instruction_template(
            summarize_instruction,
            flag="--summarize_instruction",
            word_count=sw,
            match_abstract_length=match_abstract_length,
        )
        if sw is not None and sw < 1:
            raise ValueError("summarize word count must be >= 1 when set")
        tasks.append(
            {
                "name": SUMMARY_SEGMENT,
                "subdir": SUMMARY_SUBDIR,
                "instruction": summarize_instruction,
                "max_generated_words": sw,
                "word_count_prompt": False,
            }
        )
    return tasks


def _effective_task_max_generated_words(
    task: dict[str, Any],
    excerpt_text: str,
    *,
    match_abstract_length: bool,
) -> int | None:
    """Resolve the word target/cap for one doc under one generation task."""
    if match_abstract_length:
        return response_word_count(excerpt_text)
    mgw = task.get("max_generated_words")
    return int(mgw) if mgw is not None else None


def _quote_cmd_arg(arg: str) -> str:
    """Quote one argv token for Windows cmd.exe (not PowerShell)."""
    if arg == "":
        return '""'
    if any(c in arg for c in " \t"):
        return '"' + arg.replace('"', '""') + '"'
    return arg


def format_cli_command(argv: list[str] | None = None) -> str:
    """Join argv into a single copy-pasteable Windows cmd.exe one-liner."""
    parts = sys.argv if argv is None else argv
    return " ".join(_quote_cmd_arg(p) for p in parts)


def _existing_run_cli(out_dir: str) -> tuple[str | None, list[str] | None]:
    path = os.path.join(out_dir, "run_info.json")
    if not os.path.isfile(path):
        return None, None
    prev = _load_json(path)
    cli_argv = prev.get("cli_argv")
    if isinstance(cli_argv, list) and all(isinstance(x, str) for x in cli_argv):
        return prev.get("cli_command") or format_cli_command(cli_argv), cli_argv
    cli_command = prev.get("cli_command")
    if isinstance(cli_command, str):
        return cli_command, None
    return None, None


def _pairwise_cka_word_files(
    path_a: str,
    path_b: str,
    *,
    chunk_rows: int,
    filters: list[tuple[str, str]] | None = None,
    doc_ids: set[int] | None = None,
) -> dict[str, Any]:
    return pairwise_linear_cka_from_paths(
        path_a,
        path_b,
        chunk_rows=chunk_rows,
        filters=filters,
        doc_ids=doc_ids,
    )


def _cka_record_merge_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    rs = rec.get("row_slice")
    rs_s = json.dumps(rs, sort_keys=True, default=str) if rs is not None else ""
    return (
        rec.get("segment"),
        rec.get("aggregation", "word"),
        rec.get("model_a"),
        rec.get("model_b"),
        rs_s,
    )


def _abspaths(paths: list[str]) -> list[str]:
    return [os.path.abspath(p) for p in paths]


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_pipeline_config(out_dir: str) -> dict[str, Any]:
    p = os.path.join(out_dir, CONFIG_NAME)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing {CONFIG_NAME}. Run --stage init first (same --out_dir after first create)."
        )
    return _load_json(p)


def _save_pipeline_config(out_dir: str, cfg: dict[str, Any]) -> None:
    _save_json(os.path.join(out_dir, CONFIG_NAME), cfg)


def _effective_max_generated_words(cfg: dict[str, Any]) -> int | None:
    """Frozen generation word cap from ``pipeline_config.json``.

    Prefers ``max_generated_words``; falls back to legacy ``max_words`` so older runs resume.
    """
    if "max_generated_words" in cfg:
        return cfg["max_generated_words"]
    return cfg.get("max_words")


def _effective_match_abstract_length(cfg: dict[str, Any]) -> bool:
    """Whether generation word caps match each excerpt's word count."""
    return bool(cfg.get("match_abstract_length", False))


def _effective_word_count_prompt(cfg: dict[str, Any]) -> bool:
    """Whether the primary response task used the word-count preset instruction."""
    return bool(cfg.get("word_count_prompt", False))


def _effective_generation_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen generation-task list, deriving one for legacy configs.

    Configs written before the task-list schema only carry the single-response fields
    (``instruction`` / ``word_count_prompt`` / ``max_generated_words``); those resume as
    a one-element list describing the ``response`` task so old runs keep working.
    """
    tasks = cfg.get("generation_tasks")
    if isinstance(tasks, list) and tasks:
        return tasks
    return [
        {
            "name": RESPONSE_SEGMENT,
            "subdir": RESPONSE_SUBDIR,
            "instruction": cfg.get("instruction", _DEFAULT_INSTRUCTION),
            "max_generated_words": _effective_max_generated_words(cfg),
            "word_count_prompt": _effective_word_count_prompt(cfg),
        }
    ]


def _load_pipeline_state(out_dir: str) -> dict[str, Any]:
    p = os.path.join(out_dir, STATE_NAME)
    if not os.path.isfile(p):
        return {"completed": [], "updated_iso": None}
    return _load_json(p)


def _save_pipeline_state(out_dir: str, state: dict[str, Any]) -> None:
    state["updated_iso"] = datetime.now().isoformat()
    _save_json(os.path.join(out_dir, STATE_NAME), state)


def _mark_stage_done(out_dir: str, stage: str) -> None:
    st = _load_pipeline_state(out_dir)
    done = set(st.get("completed", []))
    done.add(stage)
    # Keep deterministic order so resumed runs are easier to inspect/diff.
    st["completed"] = sorted(done, key=lambda s: STAGES_ORDER.index(s) if s in STAGES_ORDER else 99)
    _save_pipeline_state(out_dir, st)


def _stage_done(out_dir: str, stage: str) -> bool:
    st = _load_pipeline_state(out_dir)
    return stage in set(st.get("completed", []))


def _verify_models_match(cfg_models: list[str], cli_models: list[str]) -> None:
    a, b = _abspaths(cfg_models), _abspaths(cli_models)
    if a != b:
        raise SystemExit(
            f"--models must match pipeline_config.json exactly.\n  config: {a}\n  cli:    {b}"
        )


def _cfg_matches_init(
    cfg: dict[str, Any],
    *,
    corpus: str,
    words: int,
    models: list[str],
    instruction: str,
    word_count_prompt: bool,
    skip_generate: bool,
    skip_cka: bool,
    cka_chunk_rows: int,
    max_new_tokens: int,
    max_generated_words: int | None,
    match_abstract_length: bool,
    batch_size: int,
    chunk_size: int,
    local_only: bool,
    aggregation_level: str,
    generation_tasks: list[dict[str, Any]],
) -> bool:
    return (
        cfg.get("corpus") == os.path.abspath(corpus)
        and cfg.get("words") == words
        and cfg.get("models") == _abspaths(models)
        and cfg.get("instruction") == instruction
        and _effective_word_count_prompt(cfg) == word_count_prompt
        and _effective_generation_tasks(cfg) == generation_tasks
        and cfg.get("skip_generate") == skip_generate
        and cfg.get("skip_cka") == skip_cka
        and cfg.get("cka_chunk_rows") == cka_chunk_rows
        and cfg.get("max_new_tokens") == max_new_tokens
        and _effective_max_generated_words(cfg) == max_generated_words
        and _effective_match_abstract_length(cfg) == match_abstract_length
        and cfg.get("batch_size") == batch_size
        and cfg.get("chunk_size") == chunk_size
        and cfg.get("local_only") == local_only
        and cfg.get("aggregation_level", "word") == aggregation_level
    )


def _build_excerpt_rows(raw_rows: list[dict[str, Any]], words: int) -> list[dict[str, Any]]:
    excerpt_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        ex = first_n_words(r["text"], words)
        excerpt_rows.append(
            {
                "doc_id": int(r["doc_id"]),
                "category": r.get("category", ""),
                "title": r.get("title", ""),
                "text": ex,
                "segment": "excerpt",
            }
        )
    return excerpt_rows


def _load_excerpt_rows(out_dir: str) -> list[dict[str, Any]]:
    return load_corpus_jsonl(os.path.join(out_dir, "corpus_excerpts_used.jsonl"))


def _task_jsonl_path(out_dir: str, subdir: str, slug: str) -> str:
    return os.path.join(out_dir, subdir, slug, "responses.jsonl")


def _responses_jsonl_path(out_dir: str, slug: str) -> str:
    return _task_jsonl_path(out_dir, RESPONSE_SUBDIR, slug)


def _load_task_by_model(
    out_dir: str, subdir: str, model_paths: list[str]
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for mp in model_paths:
        slug = sanitize_model_slug(mp)
        path = _task_jsonl_path(out_dir, subdir, slug)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing generated text for {slug}: {path}")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[slug] = rows
    return out


def _load_response_by_model(
    out_dir: str, model_paths: list[str]
) -> dict[str, list[dict[str, Any]]]:
    return _load_task_by_model(out_dir, RESPONSE_SUBDIR, model_paths)


def _excerpt_word_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "excerpts", slug, "word_embeddings_merged_agnostic.npy")


def _task_word_npy(out_dir: str, subdir: str, slug: str) -> str:
    return os.path.join(out_dir, subdir, slug, "word_embeddings_merged_agnostic.npy")


def _task_doc_npy(out_dir: str, subdir: str, slug: str) -> str:
    return os.path.join(out_dir, subdir, slug, "document_embeddings_merged_agnostic.npy")


def _response_word_npy(out_dir: str, slug: str) -> str:
    return _task_word_npy(out_dir, RESPONSE_SUBDIR, slug)


def _excerpt_doc_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "excerpts", slug, "document_embeddings_merged_agnostic.npy")


def _response_doc_npy(out_dir: str, slug: str) -> str:
    return _task_doc_npy(out_dir, RESPONSE_SUBDIR, slug)


def _aggregation_outputs(aggregation_level: str) -> tuple[bool, bool]:
    want_word = aggregation_level in ("word", "both")
    want_doc = aggregation_level in ("document", "both")
    return want_word, want_doc


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_")


def _load_embeddings_meta(path: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.load(path, allow_pickle=True).item()
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    meta = list(data["meta"])
    if emb.shape[0] != len(meta):
        raise ValueError(f"embeddings/meta length mismatch for {path}: {emb.shape[0]} vs {len(meta)}")
    return emb, meta


def _doc_meta_category_indices(meta: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        cat = str(m.get("category", ""))
        out.setdefault(cat, []).append(i)
    return out


def _align_doc_embeddings_by_doc_id(
    emb_a: np.ndarray,
    meta_a: list[dict[str, Any]],
    emb_b: np.ndarray,
    meta_b: list[dict[str, Any]],
    *,
    doc_ids_allow: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    idx_a: dict[int, int] = {}
    idx_b: dict[int, int] = {}
    for i, m in enumerate(meta_a):
        idx_a[int(m["doc_id"])] = i
    for i, m in enumerate(meta_b):
        idx_b[int(m["doc_id"])] = i
    shared = sorted(set(idx_a).intersection(idx_b))
    if doc_ids_allow is not None:
        shared = [d for d in shared if int(d) in doc_ids_allow]
    if not shared:
        return (
            np.zeros((0, emb_a.shape[1]), dtype=emb_a.dtype),
            np.zeros((0, emb_b.shape[1]), dtype=emb_b.dtype),
            [],
        )
    ia = [idx_a[d] for d in shared]
    ib = [idx_b[d] for d in shared]
    return emb_a[ia], emb_b[ib], shared


def _document_cka_run_params(
    cfg: dict[str, Any], run_info: dict[str, Any]
) -> dict[str, Any]:
    """Subset of pipeline settings stored on document_cka_by_category.json.

    Written at CKA time from the frozen config / run_info for that run. Omits
    top-level ``instruction`` (response-only; incomplete when a summary task
    exists). Each generation task keeps its own ``instruction`` plus word cap.
    Omits per-task ``word_count_prompt`` (that flag means the response CLI
    preset was used, not whether ``{word_count}`` appears in the prompt).
    """
    keys = (
        "corpus",
        "words",
        "models",
        "aggregation_level",
        "max_new_tokens",
        "max_generated_words",
        "match_abstract_length",
        "word_count_prompt",
        "generation_tasks",
        "num_docs",
        "cli_command",
        "device",
        "batch_size",
        "chunk_size",
        "skip_generate",
        "skip_cka",
        "cka_chunk_rows",
        "local_only",
    )
    merged: dict[str, Any] = {}
    for src in (cfg, run_info):
        for k in keys:
            if k in src:
                merged[k] = src[k]
    tasks = merged.get("generation_tasks")
    if isinstance(tasks, list):
        compact: list[dict[str, Any]] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            entry: dict[str, Any] = {
                "name": t.get("name"),
                "max_generated_words": t.get("max_generated_words"),
            }
            if "instruction" in t:
                entry["instruction"] = t.get("instruction")
            compact.append(entry)
        if compact:
            merged["generation_tasks"] = compact
    return merged


def _document_category_matrix(
    *,
    out_dir: str,
    slugs: list[str],
    skip_generate: bool,
    chunk_rows: int,
    generation_tasks: list[dict[str, Any]] | None = None,
    doc_ids_allow: set[int] | None = None,
) -> dict[str, Any]:
    generation_tasks = generation_tasks or []
    paths_by_key: dict[tuple[str, str], str] = {}
    for slug in slugs:
        ex = _excerpt_doc_npy(out_dir, slug)
        if os.path.isfile(ex):
            paths_by_key[(slug, "excerpt")] = ex
        if not skip_generate:
            for task in generation_tasks:
                rp = _task_doc_npy(out_dir, task["subdir"], slug)
                if os.path.isfile(rp):
                    paths_by_key[(slug, task["name"])] = rp

    if len(paths_by_key) < 2:
        return {
            "results": [],
            "aggregated": [],
            "aggregated_focused": {
                "cross_model_same_segment": [],
                "within_model_different_segments": [],
            },
            "focused": {
                "cross_model_same_segment": [],
                "within_model_different_segments": [],
            },
        }

    loaded: dict[tuple[str, str], tuple[np.ndarray, list[dict[str, Any]], dict[str, list[int]]]] = {}
    categories: set[str] = set()
    for key, path in paths_by_key.items():
        emb, meta = _load_embeddings_meta(path)
        cidx = _doc_meta_category_indices(meta)
        loaded[key] = (emb, meta, cidx)
        categories.update(cidx.keys())

    keys = sorted(loaded.keys(), key=lambda x: (x[0], x[1]))
    records: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    cka_dir = os.path.join(out_dir, "cka")
    os.makedirs(cka_dir, exist_ok=True)

    pool_label = "(all documents)"

    for (ma, sa), (mb, sb) in itertools.combinations(keys, 2):
        ea, meta_a, _cidx_a = loaded[(ma, sa)]
        eb, meta_b, _cidx_b = loaded[(mb, sb)]
        xa, xb, shared_doc_ids = _align_doc_embeddings_by_doc_id(
            ea,
            meta_a,
            eb,
            meta_b,
            doc_ids_allow=doc_ids_allow,
        )
        n = int(xa.shape[0])
        rec: dict[str, Any] = {
            "analysis": "document_pooled",
            "aggregation": "document",
            "segment": f"{sa}__vs__{sb}",
            "segment_a": sa,
            "segment_b": sb,
            "category": pool_label,
            "pooled": True,
            "model_a": ma,
            "model_b": mb,
            "path_a": paths_by_key[(ma, sa)],
            "path_b": paths_by_key[(mb, sb)],
            "n_rows": n,
            "n_rows_total_a": int(ea.shape[0]),
            "n_rows_total_b": int(eb.shape[0]),
            "pairing": "doc_id_aligned",
            "doc_ids": shared_doc_ids,
            "row_slice": {
                "pooled": True,
                "doc_ids": sorted(doc_ids_allow) if doc_ids_allow else None,
            },
        }
        if n < 2:
            rec["linear_cka"] = float("nan")
            rec["error"] = "fewer than 2 aligned docs"
        else:
            rec["linear_cka"] = float(linear_cka_chunked(xa, xb, chunk_rows))
        aggregated.append(rec)

        file_name = f"pooled__{_safe_slug(ma)}_{sa}__vs__{_safe_slug(mb)}_{sb}.json"
        with open(os.path.join(cka_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    for cat in sorted(categories):
        cat_keys = [k for k in keys if cat in loaded[k][2]]
        for (ma, sa), (mb, sb) in itertools.combinations(cat_keys, 2):
            ea, meta_a, cidx_a = loaded[(ma, sa)]
            eb, meta_b, cidx_b = loaded[(mb, sb)]
            ia = cidx_a[cat]
            ib = cidx_b[cat]
            ea_cat = ea[ia]
            eb_cat = eb[ib]
            meta_a_cat = [meta_a[i] for i in ia]
            meta_b_cat = [meta_b[i] for i in ib]
            xa, xb, shared_doc_ids = _align_doc_embeddings_by_doc_id(
                ea_cat,
                meta_a_cat,
                eb_cat,
                meta_b_cat,
                doc_ids_allow=doc_ids_allow,
            )
            n = int(xa.shape[0])
            rec: dict[str, Any] = {
                "analysis": "document_by_category",
                "aggregation": "document",
                "segment": f"{sa}__vs__{sb}",
                "segment_a": sa,
                "segment_b": sb,
                "category": cat,
                "model_a": ma,
                "model_b": mb,
                "path_a": paths_by_key[(ma, sa)],
                "path_b": paths_by_key[(mb, sb)],
                "n_rows": n,
                "n_rows_total_a": int(ea_cat.shape[0]),
                "n_rows_total_b": int(eb_cat.shape[0]),
                "pairing": "doc_id_aligned",
                "doc_ids": shared_doc_ids,
                "row_slice": {"filters": [["category", cat]], "doc_ids": sorted(doc_ids_allow) if doc_ids_allow else None},
            }
            if n < 2:
                rec["linear_cka"] = float("nan")
                rec["error"] = "fewer than 2 aligned docs"
            else:
                rec["linear_cka"] = float(linear_cka_chunked(xa, xb, chunk_rows))
            records.append(rec)

            file_name = (
                f"by_category__{_safe_slug(cat)}__"
                f"{_safe_slug(ma)}_{sa}__vs__{_safe_slug(mb)}_{sb}.json"
            )
            with open(os.path.join(cka_dir, file_name), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)

    focused: dict[str, list[dict[str, Any]]] = {
        "cross_model_same_segment": [],
        "within_model_different_segments": [],
    }
    for r in records:
        same_model = r["model_a"] == r["model_b"]
        same_segment = r["segment_a"] == r["segment_b"]
        if not same_model and same_segment:
            focused["cross_model_same_segment"].append(r)
        elif same_model and not same_segment:
            focused["within_model_different_segments"].append(r)

    aggregated_focused: dict[str, list[dict[str, Any]]] = {
        "cross_model_same_segment": [],
        "within_model_different_segments": [],
    }
    for r in aggregated:
        same_model = r["model_a"] == r["model_b"]
        same_segment = r["segment_a"] == r["segment_b"]
        if not same_model and same_segment:
            aggregated_focused["cross_model_same_segment"].append(r)
        elif same_model and not same_segment:
            aggregated_focused["within_model_different_segments"].append(r)

    return {
        "results": records,
        "aggregated": aggregated,
        "aggregated_focused": aggregated_focused,
        "focused": focused,
    }


def run_stage_init(
    *,
    out_dir: str,
    corpus: str,
    words: int,
    models: list[str],
    instruction: str,
    instruction_keys: list[str],
    word_count_prompt: bool,
    generation_tasks: list[dict[str, Any]],
    skip_generate: bool,
    skip_cka: bool,
    cka_chunk_rows: int,
    max_new_tokens: int,
    max_generated_words: int | None,
    match_abstract_length: bool,
    batch_size: int,
    chunk_size: int,
    local_only: bool,
    device: str,
    aggregation_level: str,
    force_redo: bool,
    cli_command: str | None = None,
    cli_argv: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_path = os.path.join(out_dir, "corpus_excerpts_used.jsonl")
    cfg_path = os.path.join(out_dir, CONFIG_NAME)
    if (
        os.path.isfile(used_path)
        and os.path.isfile(cfg_path)
        and not force_redo
    ):
        cfg = _load_json(cfg_path)
        if not _cfg_matches_init(
            cfg,
            corpus=corpus,
            words=words,
            models=models,
            instruction=instruction,
            word_count_prompt=word_count_prompt,
            skip_generate=skip_generate,
            skip_cka=skip_cka,
            cka_chunk_rows=cka_chunk_rows,
            max_new_tokens=max_new_tokens,
            max_generated_words=max_generated_words,
            match_abstract_length=match_abstract_length,
            batch_size=batch_size,
            chunk_size=chunk_size,
            local_only=local_only,
            aggregation_level=aggregation_level,
            generation_tasks=generation_tasks,
        ):
            raise SystemExit(
                f"[init] {CONFIG_NAME} exists but CLI args differ from saved config. "
                f"Use --overwrite to re-init, or match the original init flags."
            )
        print(f"[init] reuse existing excerpts and {CONFIG_NAME}")
        excerpt_rows = _load_excerpt_rows(out_dir)
    else:
        raw_rows = load_corpus_jsonl(corpus)
        excerpt_rows = _build_excerpt_rows(raw_rows, words)
        os.makedirs(out_dir, exist_ok=True)
        with open(used_path, "w", encoding="utf-8") as f:
            for row in excerpt_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        shutil.copy2(corpus, os.path.join(out_dir, "corpus_source.jsonl"))
        print(f"[init] wrote {used_path}")

    cfg = {
        "corpus": os.path.abspath(corpus),
        "words": words,
        "models": _abspaths(models),
        "instruction": instruction,
        "instruction_placeholders": instruction_keys,
        "word_count_prompt": word_count_prompt,
        "generation_tasks": generation_tasks,
        "skip_generate": skip_generate,
        "skip_cka": skip_cka,
        "cka_chunk_rows": cka_chunk_rows,
        "max_new_tokens": max_new_tokens,
        "max_generated_words": max_generated_words,
        "match_abstract_length": match_abstract_length,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "local_only": local_only,
        "device": device,
        "aggregation_level": aggregation_level,
    }
    _save_pipeline_config(out_dir, cfg)
    _mark_stage_done(out_dir, "init")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = {
        "corpus_source": os.path.join(out_dir, "corpus_source.jsonl"),
        "corpus_excerpts_used": used_path,
        "pipeline_config": os.path.join(out_dir, CONFIG_NAME),
        "pipeline_state": os.path.join(out_dir, STATE_NAME),
    }
    run_info: dict[str, Any] = {
        "timestamp": ts,
        "stage_mode": True,
        "corpus": cfg["corpus"],
        "words": words,
        "device": device,
        "local_only": local_only,
        "models": cfg["models"],
        "skip_generate": skip_generate,
        "skip_cka": skip_cka,
        "cka_chunk_rows": cka_chunk_rows,
        "max_new_tokens": max_new_tokens,
        "max_generated_words": max_generated_words,
        "match_abstract_length": match_abstract_length,
        "instruction": instruction,
        "instruction_placeholders": instruction_keys,
        "word_count_prompt": word_count_prompt,
        "generation_tasks": generation_tasks,
        "num_docs": len(excerpt_rows),
        "aggregation_level": aggregation_level,
        "artifacts": artifacts,
        "timings_sec": {"generation_total": 0.0, "embedding_total": 0.0, "models": {}},
        "cka": [],
    }
    prev_cli_command, prev_cli_argv = _existing_run_cli(out_dir)
    if prev_cli_command and not force_redo:
        run_info["cli_command"] = prev_cli_command
        if prev_cli_argv is not None:
            run_info["cli_argv"] = prev_cli_argv
    elif cli_command:
        run_info["cli_command"] = cli_command
        if cli_argv is not None:
            run_info["cli_argv"] = cli_argv
    return excerpt_rows, run_info


def run_stage_generate(
    *,
    out_dir: str,
    args_models: list[str],
    excerpt_rows: list[dict[str, Any]],
    generation_tasks: list[dict[str, Any]],
    max_new_tokens: int,
    match_abstract_length: bool,
    device: str,
    local_only: bool,
    force_redo: bool,
) -> None:
    cfg = _load_pipeline_config(out_dir)
    _verify_models_match(cfg["models"], args_models)
    for model_path in args_models:
        slug = sanitize_model_slug(model_path)
        # Figure out which tasks still need generation before paying to load the model.
        pending = []
        for task in generation_tasks:
            resp_path = _task_jsonl_path(out_dir, task["subdir"], slug)
            if os.path.isfile(resp_path) and not force_redo:
                print(f"[generate] skip (exists): {resp_path}")
            else:
                pending.append(task)
        if not pending:
            continue
        tok, gen = load_causal_lm(model_path, device, local_only=local_only)
        # Generate every pending task from a single model load.
        for task in pending:
            task_start = time.perf_counter()
            resp_path = _task_jsonl_path(out_dir, task["subdir"], slug)
            instruction = task["instruction"]
            lines_out = []
            for i, row in enumerate(excerpt_rows):
                mgw = _effective_task_max_generated_words(
                    task,
                    str(row.get("text") or ""),
                    match_abstract_length=match_abstract_length,
                )
                # Prompt templates can mix {title} and {excerpt}; both are populated here.
                prompt = build_explain_prompt(
                    tok,
                    instruction,
                    excerpt=row["text"],
                    title=str(row.get("title") or ""),
                    word_count=mgw,
                )
                resp_text = generate_completion(
                    tok, gen, prompt, device, max_new_tokens, max_generated_words=mgw
                )
                lines_out.append(
                    {
                        "doc_id": row["doc_id"],
                        "category": row["category"],
                        "title": row["title"],
                        "segment": task["name"],
                        "prompt": prompt,
                        "response": resp_text,
                    }
                )
                print(
                    f"  [{slug}/{task['name']}] generated {i + 1}/{len(excerpt_rows)}",
                    flush=True,
                )
            os.makedirs(os.path.dirname(resp_path), exist_ok=True)
            with open(resp_path, "w", encoding="utf-8") as rf:
                for ln in lines_out:
                    rf.write(json.dumps(ln, ensure_ascii=False) + "\n")
            print(
                f"[generate] saved {resp_path} ({time.perf_counter() - task_start:.1f}s)"
            )
        del gen
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()
    _mark_stage_done(out_dir, "generate")


def run_stage_embed_excerpts(
    *,
    out_dir: str,
    args_models: list[str],
    excerpt_rows: list[dict[str, Any]],
    device: str,
    local_only: bool,
    batch_size: int,
    chunk_size: int,
    aggregation_level: str,
    run_info: dict[str, Any],
    force_redo: bool,
) -> None:
    cfg = _load_pipeline_config(out_dir)
    _verify_models_match(cfg["models"], args_models)
    artifacts = run_info.setdefault("artifacts", {})
    for model_path in args_models:
        slug = sanitize_model_slug(model_path)
        w_path = _excerpt_word_npy(out_dir, slug)
        d_path = _excerpt_doc_npy(out_dir, slug)
        want_word, want_doc = _aggregation_outputs(aggregation_level)
        required_paths = []
        if want_word:
            required_paths.append(w_path)
        if want_doc:
            required_paths.append(d_path)
        if required_paths and all(os.path.isfile(p) for p in required_paths) and not force_redo:
            print(f"[embed_excerpts] skip (exists): {', '.join(required_paths)}")
            continue
        ex_dir = os.path.join(out_dir, "excerpts", slug)
        os.makedirs(ex_dir, exist_ok=True)
        t0 = time.perf_counter()
        emb_e, meta_e = embed_labeled_texts(
            excerpt_rows,
            model_path,
            device=device,
            local_files_only=local_only,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        # Save token-level arrays first; word/doc aggregation is derived from this.
        tok_e = os.path.join(ex_dir, "token_embeddings.npy")
        save_token_npy(tok_e, emb_e, meta_e)
        if want_word:
            w_e, m_e = merge_token_embeddings_to_words(emb_e, meta_e)
            save_word_npy(w_path, w_e, m_e)
        if want_doc:
            d_e, dm_e = merge_token_embeddings_to_docs(emb_e, meta_e)
            save_word_npy(d_path, d_e, dm_e)
        artifacts[f"excerpts_{slug}_token_embeddings"] = tok_e
        if want_word:
            artifacts[f"excerpts_{slug}_word_embeddings"] = w_path
        if want_doc:
            artifacts[f"excerpts_{slug}_document_embeddings"] = d_path
        run_info["timings_sec"].setdefault("models", {}).setdefault(slug, {})[
            "embedding_excerpt"
        ] = round(time.perf_counter() - t0, 3)
        print(f"[embed_excerpts] {slug} ({run_info['timings_sec']['models'][slug]['embedding_excerpt']}s)")
    _mark_stage_done(out_dir, "embed_excerpts")


def run_stage_embed_responses(
    *,
    out_dir: str,
    args_models: list[str],
    generation_tasks: list[dict[str, Any]],
    device: str,
    local_only: bool,
    batch_size: int,
    chunk_size: int,
    aggregation_level: str,
    run_info: dict[str, Any],
    force_redo: bool,
) -> None:
    cfg = _load_pipeline_config(out_dir)
    if cfg.get("skip_generate"):
        print("[embed_responses] skip (skip_generate in pipeline_config)")
        _mark_stage_done(out_dir, "embed_responses")
        return
    _verify_models_match(cfg["models"], args_models)
    want_word, want_doc = _aggregation_outputs(aggregation_level)
    artifacts = run_info.setdefault("artifacts", {})
    # Embed each generation task's output (response, summary, ...) with its segment label.
    for task in generation_tasks:
        subdir = task["subdir"]
        segment = task["name"]
        docs_by_model = _load_task_by_model(out_dir, subdir, args_models)
        for model_path in args_models:
            slug = sanitize_model_slug(model_path)
            w_path = _task_word_npy(out_dir, subdir, slug)
            d_path = _task_doc_npy(out_dir, subdir, slug)
            required_paths = []
            if want_word:
                required_paths.append(w_path)
            if want_doc:
                required_paths.append(d_path)
            if required_paths and all(os.path.isfile(p) for p in required_paths) and not force_redo:
                print(f"[embed_responses] skip (exists): {', '.join(required_paths)}")
                continue
            resp_docs = docs_by_model[slug]
            # Rebuild into the same schema used for excerpts so embedding stays generic.
            resp_rows = [
                {
                    "doc_id": int(x["doc_id"]),
                    "category": x["category"],
                    "title": x["title"],
                    "text": x["response"],
                    "segment": segment,
                }
                for x in resp_docs
            ]
            r_dir = os.path.join(out_dir, subdir, slug)
            os.makedirs(r_dir, exist_ok=True)
            t0 = time.perf_counter()
            emb_r, meta_r = embed_labeled_texts(
                resp_rows,
                model_path,
                device=device,
                local_files_only=local_only,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )
            save_token_npy(os.path.join(r_dir, "token_embeddings.npy"), emb_r, meta_r)
            if want_word:
                w_r, m_r = merge_token_embeddings_to_words(emb_r, meta_r)
                save_word_npy(w_path, w_r, m_r)
            if want_doc:
                d_r, dm_r = merge_token_embeddings_to_docs(emb_r, meta_r)
                save_word_npy(d_path, d_r, dm_r)
            artifacts[f"{subdir}_{slug}_token_embeddings"] = os.path.join(
                r_dir, "token_embeddings.npy"
            )
            if want_word:
                artifacts[f"{subdir}_{slug}_word_embeddings"] = w_path
            if want_doc:
                artifacts[f"{subdir}_{slug}_document_embeddings"] = d_path
            elapsed = round(time.perf_counter() - t0, 3)
            run_info["timings_sec"].setdefault("models", {}).setdefault(slug, {})[
                f"embedding_{segment}"
            ] = elapsed
            print(f"[embed_responses] {slug}/{segment} ({elapsed}s)")
    _mark_stage_done(out_dir, "embed_responses")


def run_stage_cka(
    *,
    out_dir: str,
    args_models: list[str],
    skip_generate: bool,
    skip_cka: bool,
    cka_chunk_rows: int,
    run_info: dict[str, Any],
    force_redo: bool,
    cka_filters: list[tuple[str, str]] | None = None,
    cka_doc_ids: set[int] | None = None,
    cka_slice_label: str | None = None,
) -> None:
    if skip_cka:
        print("[cka] skip (--skip_cka)")
        _mark_stage_done(out_dir, "cka")
        return
    cfg = _load_pipeline_config(out_dir)
    want_word, want_doc = _aggregation_outputs(cfg.get("aggregation_level", "word"))
    if not (want_word or want_doc):
        raise SystemExit("[cka] no aggregation outputs configured for CKA.")
    _verify_models_match(cfg["models"], args_models)
    if len(args_models) < 2:
        print("CKA skipped: need at least two models.")
        return
    cka_filters = list(cka_filters or [])
    has_row_slice = bool(cka_filters) or bool(cka_doc_ids)
    cka_index = os.path.join(out_dir, "cka", "cka_index.json")
    if os.path.isfile(cka_index) and not force_redo and not has_row_slice:
        print(f"[cka] skip (exists): {cka_index}")
        _mark_stage_done(out_dir, "cka")
        return
    cka_dir = os.path.join(out_dir, "cka")
    os.makedirs(cka_dir, exist_ok=True)
    slugs = [sanitize_model_slug(m) for m in args_models]
    slice_suffix = cka_row_slice_file_suffix(cka_filters, cka_doc_ids, cka_slice_label)
    generation_tasks = _effective_generation_tasks(cfg)
    cka_records: list[dict[str, Any]] = []
    # agg_spec = (aggregation name, filename suffix, excerpt path fn, task path fn).
    agg_specs: list[tuple[str, str, Any, Any]] = []
    if want_word:
        agg_specs.append(("word", "", _excerpt_word_npy, _task_word_npy))
    if want_doc:
        agg_specs.append(("document", "_document", _excerpt_doc_npy, _task_doc_npy))

    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            si, sj = slugs[i], slugs[j]
            for agg_name, seg_suffix, excerpt_path_fn, task_path_fn in agg_specs:
                # Compare matched aggregation outputs (word and/or document) per segment.
                ex_a = excerpt_path_fn(out_dir, si)
                ex_b = excerpt_path_fn(out_dir, sj)
                if not (os.path.isfile(ex_a) and os.path.isfile(ex_b)):
                    raise FileNotFoundError(
                        f"Missing excerpt {agg_name} npy for CKA: {ex_a} / {ex_b}"
                    )
                rec_ex = {
                    "segment": "excerpt",
                    "aggregation": agg_name,
                    "model_a": si,
                    "model_b": sj,
                    **_pairwise_cka_word_files(
                        ex_a,
                        ex_b,
                        chunk_rows=cka_chunk_rows,
                        filters=cka_filters or None,
                        doc_ids=cka_doc_ids,
                    ),
                }
                cka_records.append(rec_ex)
                ex_name = f"excerpt{seg_suffix}__{si}__vs__{sj}{slice_suffix}.json"
                with open(os.path.join(cka_dir, ex_name), "w", encoding="utf-8") as cf:
                    json.dump(rec_ex, cf, indent=2)
                if rec_ex.get("error"):
                    print(f"CKA excerpt ({agg_name}) {si} vs {sj}: {rec_ex['error']}")
                else:
                    print(
                        f"CKA excerpt ({agg_name}) {si} vs {sj}: "
                        f"linear_cka={rec_ex['linear_cka']:.6f} n={rec_ex['n_rows']}"
                    )
                if skip_generate:
                    continue
                # One CKA record per generated segment (response, summary, ...).
                for task in generation_tasks:
                    segment = task["name"]
                    r_a = task_path_fn(out_dir, task["subdir"], si)
                    r_b = task_path_fn(out_dir, task["subdir"], sj)
                    if not (os.path.isfile(r_a) and os.path.isfile(r_b)):
                        continue
                    rec_r = {
                        "segment": segment,
                        "aggregation": agg_name,
                        "model_a": si,
                        "model_b": sj,
                        **_pairwise_cka_word_files(
                            r_a,
                            r_b,
                            chunk_rows=cka_chunk_rows,
                            filters=cka_filters or None,
                            doc_ids=cka_doc_ids,
                        ),
                    }
                    cka_records.append(rec_r)
                    resp_name = f"{segment}{seg_suffix}__{si}__vs__{sj}{slice_suffix}.json"
                    with open(os.path.join(cka_dir, resp_name), "w", encoding="utf-8") as cf:
                        json.dump(rec_r, cf, indent=2)
                    if rec_r.get("error"):
                        print(f"CKA {segment} ({agg_name}) {si} vs {sj}: {rec_r['error']}")
                    else:
                        print(
                            f"CKA {segment} ({agg_name}) {si} vs {sj}: "
                            f"linear_cka={rec_r['linear_cka']:.6f} n={rec_r['n_rows']}"
                        )
    if has_row_slice and os.path.isfile(cka_index):
        prev_list = _load_json(cka_index)
        if not isinstance(prev_list, list):
            prev_list = []
        new_keys = {_cka_record_merge_key(r) for r in cka_records}
        prev_kept = [r for r in prev_list if _cka_record_merge_key(r) not in new_keys]
        cka_records = prev_kept + cka_records
    run_info["cka"] = cka_records
    with open(cka_index, "w", encoding="utf-8") as jf:
        json.dump(cka_records, jf, indent=2)
    run_info.setdefault("artifacts", {})["cka_index"] = cka_index

    # Richer document-level matrix by category across model/segment pairs.
    # Keep this separate from cka_index so existing consumers remain unchanged.
    by_cat_index = os.path.join(cka_dir, "document_cka_by_category.json")
    want_word, want_doc = _aggregation_outputs(cfg.get("aggregation_level", "word"))
    if want_doc:
        if cka_filters:
            print(
                "[cka] document_cka_by_category skipped when --cka_filter is provided "
                "(category slicing is built in for this matrix)."
            )
        elif os.path.isfile(by_cat_index) and not force_redo and not has_row_slice:
            print(f"[cka] document-by-category skip (exists): {by_cat_index}")
        else:
            by_cat = _document_category_matrix(
                out_dir=out_dir,
                slugs=slugs,
                skip_generate=skip_generate,
                chunk_rows=cka_chunk_rows,
                generation_tasks=generation_tasks,
                doc_ids_allow=cka_doc_ids,
            )
            by_cat_obj = {
                "analysis": "document_by_category",
                "out_dir": out_dir,
                "models": slugs,
                "skip_generate": bool(skip_generate),
                "chunk_rows": int(cka_chunk_rows),
                "run_params": _document_cka_run_params(cfg, run_info),
                "results": by_cat.get("results", []),
                "aggregated": by_cat.get("aggregated", []),
                "aggregated_focused": by_cat.get("aggregated_focused", {}),
                "focused": by_cat.get("focused", {}),
            }
            with open(by_cat_index, "w", encoding="utf-8") as f:
                json.dump(by_cat_obj, f, indent=2)
            run_info["cka_document_by_category"] = by_cat_obj
            print(
                "[cka] document-by-category (per-category) comparisons:",
                len(by_cat_obj["results"]),
            )
            print(
                "[cka] document Pooled (all doc_id aligned rows) comparisons:",
                len(by_cat_obj.get("aggregated", [])),
            )
            f = by_cat_obj.get("focused", {})
            print(
                "      (per-cat) cross_model_same_segment=",
                len(f.get("cross_model_same_segment", [])),
                "within_model_different_segments=",
                len(
                    f.get("within_model_different_segments")
                    or f.get("within_model_excerpt_vs_response")
                    or []
                ),
            )
            agf = by_cat_obj.get("aggregated_focused", {}) or {}
            print(
                "      (pooled) cross_model_same_segment=",
                len(agf.get("cross_model_same_segment", [])),
                "within_model_different_segments=",
                len(
                    agf.get("within_model_different_segments")
                    or agf.get("within_model_excerpt_vs_response")
                    or []
                ),
            )
        run_info.setdefault("artifacts", {})["cka_document_by_category"] = by_cat_index

    _mark_stage_done(out_dir, "cka")


def _merge_run_info(out_dir: str, run_info: dict[str, Any]) -> None:
    path = os.path.join(out_dir, "run_info.json")
    if os.path.isfile(path):
        prev = _load_json(path)
        prev.setdefault("timings_sec", {}).setdefault("models", {})
        prev["timings_sec"]["models"].update(run_info.get("timings_sec", {}).get("models", {}))
        for k in ("artifacts", "cka", "num_docs", "cli_command", "cli_argv"):
            if k in run_info:
                prev[k] = run_info[k]
        prev["last_stage_update"] = datetime.now().isoformat()
        run_info = prev
    _save_json(path, run_info)


def main() -> None:
    run_start = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="JSONL with doc_id, category, title, text")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Run directory (required for every stage except init, where it defaults if omitted)",
    )
    p.add_argument("--words", type=int, default=200)
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local model paths; must match pipeline_config.json when resuming",
    )
    p.add_argument(
        "--stage",
        choices=("all",) + STAGES_ORDER,
        default="all",
        help="Pipeline stage to run (default: all in one go). Use separate invocations to pause between stages.",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip outputs that already exist (default: true)",
    )
    p.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Recompute even when outputs exist (same as --overwrite for file skips)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo work even if stage outputs already exist (implies re-embedding / regenerating)",
    )
    p.add_argument(
        "--skip_generate",
        action="store_true",
        help="Only embed excerpts (no causal LM responses); stored in pipeline_config",
    )
    p.add_argument(
        "--allow_remote",
        action="store_true",
        help="Allow Hugging Face Hub downloads for model weights (default: local files only)",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--max_generated_words",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional, generate stage only: stop once the model's decoded completion "
            "(the assistant reply, not the prompt or excerpt corpus) has at least N "
            "whitespace-delimited words; hyphenated spellings count as one word. "
            "Still bounded by --max_new_tokens and EOS. Does not cap --words excerpts or "
            "embedding length. Stored as max_generated_words in pipeline_config; must match "
            "on --stage generate. Legacy configs may still have max_words (same meaning). "
            "Incompatible with --match_abstract_length."
        ),
    )
    p.add_argument(
        "--match_abstract_length",
        action="store_true",
        help=(
            "Per-document generation word target/cap: for each excerpt, use that excerpt's "
            "whitespace word count (after --words truncation) as {word_count} and the hard "
            "cap for both the response and summary tasks. Incompatible with "
            "--max_generated_words / --summarize_words. Enables {word_count} / "
            "--word_count_prompt without a fixed N. Stored in pipeline_config; must match "
            "on resumed stages."
        ),
    )
    p.add_argument(
        "--instruction",
        default=_DEFAULT_INSTRUCTION,
        help="Prompt template; {title}, {excerpt}, and/or {word_count}",
    )
    p.add_argument(
        "--word_count_prompt",
        action="store_true",
        help=(
            "Use the preset instruction "
            f"{WORD_COUNT_INSTRUCTION!r}; requires --max_generated_words "
            "or --match_abstract_length"
        ),
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help=(
            "Additive: also generate a summarization task (segment 'summary') alongside "
            "the response task. Outputs go under summaries/<slug>/ and are embedded and "
            "CKA-compared as an extra segment. Stored in pipeline_config; re-pass on "
            "staged runs to match config."
        ),
    )
    p.add_argument(
        "--summarize_instruction",
        default=SUMMARIZE_INSTRUCTION,
        help=(
            "Prompt template for the --summarize task; {title}, {excerpt}, and/or "
            f"{{word_count}} (default: {SUMMARIZE_INSTRUCTION!r})."
        ),
    )
    p.add_argument(
        "--summarize_words",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Word target/cap for the --summarize task (fills {word_count} and hard-caps "
            "the generated summary). Defaults to --max_generated_words when omitted. "
            "Incompatible with --match_abstract_length."
        ),
    )
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument(
        "--aggregation_level",
        choices=("word", "document", "both"),
        default="word",
        help=(
            "Embedding aggregation outputs to write from token embeddings: "
            "word, document, or both (default: word)."
        ),
    )
    p.add_argument(
        "--skip_cka",
        action="store_true",
        help="Skip linear CKA (stored in pipeline_config on init)",
    )
    p.add_argument(
        "--cka_chunk_rows",
        type=int,
        default=4096,
        metavar="N",
        help="Row chunk size for linear_cka_chunked",
    )
    p.add_argument(
        "--cka_filter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "CKA only: keep embedding rows whose meta matches all filters (AND). "
            "Example: --cka_filter category=Category:Statistics"
        ),
    )
    p.add_argument(
        "--cka_doc_ids",
        default=None,
        metavar="IDS",
        help="CKA only: comma-separated doc_id list (intersection with --cka_filter).",
    )
    p.add_argument(
        "--cka_slice_label",
        default=None,
        metavar="NAME",
        help="CKA only: suffix for slice JSON filenames; default is a short hash when slicing.",
    )
    args = p.parse_args()

    cka_filters_list: list[tuple[str, str]] = []
    if args.cka_filter:
        try:
            for spec in args.cka_filter:
                cka_filters_list.append(parse_meta_filter_arg(spec))
        except ValueError as e:
            p.error(str(e))
    cka_doc_ids_parsed = parse_doc_ids_arg(args.cka_doc_ids)

    if args.match_abstract_length and args.max_generated_words is not None:
        p.error("--match_abstract_length cannot be combined with --max_generated_words")
    if args.match_abstract_length and args.summarize_words is not None:
        p.error("--match_abstract_length cannot be combined with --summarize_words")
    try:
        instruction, word_count_prompt = resolve_instruction_args(
            instruction=args.instruction,
            max_generated_words=args.max_generated_words,
            word_count_prompt=args.word_count_prompt,
            match_abstract_length=args.match_abstract_length,
        )
    except ValueError as e:
        p.error(str(e))
    keys = _instruction_placeholders(instruction)
    if args.max_generated_words is not None and args.max_generated_words < 1:
        p.error("--max_generated_words must be >= 1 when set")
    if args.summarize_words is not None and args.summarize_words < 1:
        p.error("--summarize_words must be >= 1 when set")
    try:
        generation_tasks = build_generation_tasks(
            instruction=instruction,
            max_generated_words=args.max_generated_words,
            word_count_prompt=word_count_prompt,
            summarize=args.summarize,
            summarize_instruction=args.summarize_instruction,
            summarize_words=args.summarize_words,
            match_abstract_length=args.match_abstract_length,
        )
    except ValueError as e:
        p.error(str(e))

    force_redo = bool(args.overwrite or not args.resume)

    local_only = not args.allow_remote
    device = pick_device()
    print(f"Device: {device}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.stage == "init":
        out_dir = args.out_dir or os.path.join("outputs", f"categorized_{ts}")
    else:
        if not args.out_dir:
            p.error("--out_dir is required unless --stage init")
        out_dir = args.out_dir
    out_dir = os.path.abspath(out_dir)
    cli_command = format_cli_command()
    cli_argv = list(sys.argv)

    if args.stage == "all":
        # Single-process full run for convenience and simple reproducibility.
        os.makedirs(out_dir, exist_ok=True)
        excerpt_rows, run_info = run_stage_init(
            out_dir=out_dir,
            corpus=args.corpus,
            words=args.words,
            models=args.models,
            instruction=instruction,
            instruction_keys=sorted(keys),
            word_count_prompt=word_count_prompt,
            generation_tasks=generation_tasks,
            skip_generate=args.skip_generate,
            skip_cka=args.skip_cka,
            cka_chunk_rows=args.cka_chunk_rows,
            max_new_tokens=args.max_new_tokens,
            max_generated_words=args.max_generated_words,
            match_abstract_length=args.match_abstract_length,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            local_only=local_only,
            device=device,
            aggregation_level=args.aggregation_level,
            force_redo=force_redo,
            cli_command=cli_command,
            cli_argv=cli_argv,
        )
        if not args.skip_generate:
            run_stage_generate(
                out_dir=out_dir,
                args_models=args.models,
                excerpt_rows=excerpt_rows,
                generation_tasks=generation_tasks,
                max_new_tokens=args.max_new_tokens,
                match_abstract_length=args.match_abstract_length,
                device=device,
                local_only=local_only,
                force_redo=force_redo,
            )
        run_stage_embed_excerpts(
            out_dir=out_dir,
            args_models=args.models,
            excerpt_rows=excerpt_rows,
            device=device,
            local_only=local_only,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            aggregation_level=args.aggregation_level,
            run_info=run_info,
            force_redo=force_redo,
        )
        if not args.skip_generate:
            run_stage_embed_responses(
                out_dir=out_dir,
                args_models=args.models,
                generation_tasks=generation_tasks,
                device=device,
                local_only=local_only,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                aggregation_level=args.aggregation_level,
                run_info=run_info,
                force_redo=force_redo,
            )
        run_stage_cka(
            out_dir=out_dir,
            args_models=args.models,
            skip_generate=args.skip_generate,
            skip_cka=args.skip_cka,
            cka_chunk_rows=args.cka_chunk_rows,
            run_info=run_info,
            force_redo=force_redo,
            cka_filters=cka_filters_list or None,
            cka_doc_ids=cka_doc_ids_parsed,
            cka_slice_label=args.cka_slice_label,
        )
        for s in STAGES_ORDER:
            if not _stage_done(out_dir, s):
                _mark_stage_done(out_dir, s)
        run_info["timings_sec"]["total"] = round(time.perf_counter() - run_start, 3)
        _merge_run_info(out_dir, run_info)
        print(f"Done. Run directory: {out_dir}")
        return

    # --- staged mode ---
    # Each invocation performs one stage against an existing out_dir.
    run_info: dict[str, Any] = {"timings_sec": {"models": {}}, "artifacts": {}}
    if os.path.isfile(os.path.join(out_dir, "run_info.json")):
        run_info = _load_json(os.path.join(out_dir, "run_info.json"))

    excerpt_rows: list[dict[str, Any]] = []

    if args.stage == "init":
        excerpt_rows, run_info = run_stage_init(
            out_dir=out_dir,
            corpus=args.corpus,
            words=args.words,
            models=args.models,
            instruction=instruction,
            instruction_keys=sorted(keys),
            word_count_prompt=word_count_prompt,
            generation_tasks=generation_tasks,
            skip_generate=args.skip_generate,
            skip_cka=args.skip_cka,
            cka_chunk_rows=args.cka_chunk_rows,
            max_new_tokens=args.max_new_tokens,
            max_generated_words=args.max_generated_words,
            match_abstract_length=args.match_abstract_length,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            local_only=local_only,
            device=device,
            aggregation_level=args.aggregation_level,
            force_redo=force_redo,
            cli_command=cli_command,
            cli_argv=cli_argv,
        )
    else:
        cfg = _load_pipeline_config(out_dir)
        _verify_models_match(cfg["models"], args.models)
        if instruction != cfg["instruction"]:
            raise SystemExit(
                f"--instruction must match {CONFIG_NAME} exactly.\n"
                f"  config: {cfg['instruction']!r}\n  cli:    {instruction!r}"
            )
        if word_count_prompt != _effective_word_count_prompt(cfg):
            raise SystemExit(
                f"--word_count_prompt must match {CONFIG_NAME} "
                f"(word_count_prompt={_effective_word_count_prompt(cfg)!r})."
            )
        if args.match_abstract_length != _effective_match_abstract_length(cfg):
            raise SystemExit(
                f"--match_abstract_length must match {CONFIG_NAME} "
                f"(match_abstract_length={_effective_match_abstract_length(cfg)!r})."
            )
        if generation_tasks != _effective_generation_tasks(cfg):
            raise SystemExit(
                f"--summarize / generation task flags must match {CONFIG_NAME}.\n"
                f"  config: {_effective_generation_tasks(cfg)!r}\n"
                f"  cli:    {generation_tasks!r}"
            )
        if local_only != cfg.get("local_only", True):
            raise SystemExit(
                f"--allow_remote / local_only must match {CONFIG_NAME} (local_only={cfg.get('local_only')})."
            )
        excerpt_rows = _load_excerpt_rows(out_dir)

    if args.stage == "generate":
        cfg = _load_pipeline_config(out_dir)
        if args.skip_generate or cfg.get("skip_generate"):
            print("[generate] skipped (--skip_generate or pipeline_config)")
            _mark_stage_done(out_dir, "generate")
        else:
            if args.max_new_tokens != cfg["max_new_tokens"]:
                raise SystemExit(
                    f"--max_new_tokens must match {CONFIG_NAME} ({cfg['max_new_tokens']})."
                )
            cfg_mgw = _effective_max_generated_words(cfg)
            if cfg_mgw != args.max_generated_words:
                raise SystemExit(
                    f"--max_generated_words must match {CONFIG_NAME} ({cfg_mgw!r}; "
                    f"legacy key max_words is accepted when reading config)."
                )
            run_stage_generate(
                out_dir=out_dir,
                args_models=args.models,
                excerpt_rows=excerpt_rows,
                generation_tasks=_effective_generation_tasks(cfg),
                max_new_tokens=cfg["max_new_tokens"],
                match_abstract_length=_effective_match_abstract_length(cfg),
                device=device,
                local_only=local_only,
                force_redo=force_redo,
            )
    elif args.stage == "embed_excerpts":
        cfg = _load_pipeline_config(out_dir)
        run_stage_embed_excerpts(
            out_dir=out_dir,
            args_models=args.models,
            excerpt_rows=excerpt_rows,
            device=device,
            local_only=local_only,
            batch_size=cfg["batch_size"],
            chunk_size=cfg["chunk_size"],
            aggregation_level=cfg.get("aggregation_level", "word"),
            run_info=run_info,
            force_redo=force_redo,
        )
    elif args.stage == "embed_responses":
        cfg = _load_pipeline_config(out_dir)
        run_stage_embed_responses(
            out_dir=out_dir,
            args_models=args.models,
            generation_tasks=_effective_generation_tasks(cfg),
            device=device,
            local_only=local_only,
            batch_size=cfg["batch_size"],
            chunk_size=cfg["chunk_size"],
            aggregation_level=cfg.get("aggregation_level", "word"),
            run_info=run_info,
            force_redo=force_redo,
        )
    elif args.stage == "cka":
        cfg = _load_pipeline_config(out_dir)
        run_stage_cka(
            out_dir=out_dir,
            args_models=args.models,
            skip_generate=cfg.get("skip_generate", False),
            skip_cka=cfg.get("skip_cka", False),
            cka_chunk_rows=cfg["cka_chunk_rows"],
            run_info=run_info,
            force_redo=force_redo,
            cka_filters=cka_filters_list or None,
            cka_doc_ids=cka_doc_ids_parsed,
            cka_slice_label=args.cka_slice_label,
        )

    run_info["timings_sec"]["total"] = round(time.perf_counter() - run_start, 3)
    _merge_run_info(out_dir, run_info)
    print(f"Stage '{args.stage}' finished. Run directory: {out_dir}")


if __name__ == "__main__":
    main()
