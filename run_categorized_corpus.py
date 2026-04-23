"""
Categorized corpus pipeline (repeatable, all artifacts under ``--out_dir``).

Stages (use ``--stage`` to run one at a time; later stages require the same
``--out_dir`` and compatible ``--models`` as recorded in ``pipeline_config.json``):

1. **init** — excerpts JSONL, corpus copy, ``pipeline_config.json``, ``pipeline_state.json``.
2. **generate** — causal LM responses per model (unless ``skip_generate`` in config).
3. **embed_excerpts** — token + aggregated npy under ``excerpts/<slug>/``.
4. **embed_responses** — token + aggregated npy under ``responses/<slug>/`` (needs generation).
5. **cka** — pairwise linear CKA under ``cka/``.

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


_ALLOWED_INSTRUCTION_KEYS = frozenset({"excerpt", "title"})


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
    skip_generate: bool,
    skip_cka: bool,
    cka_chunk_rows: int,
    max_new_tokens: int,
    batch_size: int,
    chunk_size: int,
    local_only: bool,
    aggregation_level: str,
) -> bool:
    return (
        cfg.get("corpus") == os.path.abspath(corpus)
        and cfg.get("words") == words
        and cfg.get("models") == _abspaths(models)
        and cfg.get("instruction") == instruction
        and cfg.get("skip_generate") == skip_generate
        and cfg.get("skip_cka") == skip_cka
        and cfg.get("cka_chunk_rows") == cka_chunk_rows
        and cfg.get("max_new_tokens") == max_new_tokens
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


def _responses_jsonl_path(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "responses", slug, "responses.jsonl")


def _load_response_by_model(
    out_dir: str, model_paths: list[str]
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for mp in model_paths:
        slug = sanitize_model_slug(mp)
        path = _responses_jsonl_path(out_dir, slug)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing responses for {slug}: {path}")
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[slug] = rows
    return out


def _excerpt_word_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "excerpts", slug, "word_embeddings_merged_agnostic.npy")


def _response_word_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "responses", slug, "word_embeddings_merged_agnostic.npy")


def _excerpt_doc_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "excerpts", slug, "document_embeddings_merged_agnostic.npy")


def _response_doc_npy(out_dir: str, slug: str) -> str:
    return os.path.join(out_dir, "responses", slug, "document_embeddings_merged_agnostic.npy")


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


def _document_category_matrix(
    *,
    out_dir: str,
    slugs: list[str],
    skip_generate: bool,
    chunk_rows: int,
    doc_ids_allow: set[int] | None = None,
) -> dict[str, Any]:
    paths_by_key: dict[tuple[str, str], str] = {}
    for slug in slugs:
        ex = _excerpt_doc_npy(out_dir, slug)
        if os.path.isfile(ex):
            paths_by_key[(slug, "excerpt")] = ex
        if not skip_generate:
            rp = _response_doc_npy(out_dir, slug)
            if os.path.isfile(rp):
                paths_by_key[(slug, "response")] = rp

    if len(paths_by_key) < 2:
        return {"results": [], "focused": {}}

    loaded: dict[tuple[str, str], tuple[np.ndarray, list[dict[str, Any]], dict[str, list[int]]]] = {}
    categories: set[str] = set()
    for key, path in paths_by_key.items():
        emb, meta = _load_embeddings_meta(path)
        cidx = _doc_meta_category_indices(meta)
        loaded[key] = (emb, meta, cidx)
        categories.update(cidx.keys())

    keys = sorted(loaded.keys(), key=lambda x: (x[0], x[1]))
    records: list[dict[str, Any]] = []
    cka_dir = os.path.join(out_dir, "cka")
    os.makedirs(cka_dir, exist_ok=True)

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

    focused = {
        "cross_model_same_segment": [],
        "within_model_excerpt_vs_response": [],
        "cross_model_cross_segment": [],
    }
    for r in records:
        same_model = r["model_a"] == r["model_b"]
        same_segment = r["segment_a"] == r["segment_b"]
        if not same_model and same_segment:
            focused["cross_model_same_segment"].append(r)
        elif same_model and not same_segment:
            focused["within_model_excerpt_vs_response"].append(r)
        elif not same_model and not same_segment:
            focused["cross_model_cross_segment"].append(r)

    return {"results": records, "focused": focused}


def run_stage_init(
    *,
    out_dir: str,
    corpus: str,
    words: int,
    models: list[str],
    instruction: str,
    instruction_keys: list[str],
    skip_generate: bool,
    skip_cka: bool,
    cka_chunk_rows: int,
    max_new_tokens: int,
    batch_size: int,
    chunk_size: int,
    local_only: bool,
    device: str,
    aggregation_level: str,
    force_redo: bool,
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
            skip_generate=skip_generate,
            skip_cka=skip_cka,
            cka_chunk_rows=cka_chunk_rows,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            chunk_size=chunk_size,
            local_only=local_only,
            aggregation_level=aggregation_level,
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
        "skip_generate": skip_generate,
        "skip_cka": skip_cka,
        "cka_chunk_rows": cka_chunk_rows,
        "max_new_tokens": max_new_tokens,
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
        "instruction": instruction,
        "instruction_placeholders": instruction_keys,
        "num_docs": len(excerpt_rows),
        "aggregation_level": aggregation_level,
        "artifacts": artifacts,
        "timings_sec": {"generation_total": 0.0, "embedding_total": 0.0, "models": {}},
        "cka": [],
    }
    return excerpt_rows, run_info


def run_stage_generate(
    *,
    out_dir: str,
    args_models: list[str],
    excerpt_rows: list[dict[str, Any]],
    instruction: str,
    max_new_tokens: int,
    device: str,
    local_only: bool,
    force_redo: bool,
) -> dict[str, list[dict[str, Any]]]:
    cfg = _load_pipeline_config(out_dir)
    _verify_models_match(cfg["models"], args_models)
    response_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_path in args_models:
        slug = sanitize_model_slug(model_path)
        resp_path = _responses_jsonl_path(out_dir, slug)
        if os.path.isfile(resp_path) and not force_redo:
            print(f"[generate] skip (exists): {resp_path}")
            with open(resp_path, encoding="utf-8") as f:
                response_by_model[slug] = [json.loads(l) for l in f if l.strip()]
            continue
        gen_model_start = time.perf_counter()
        tok, gen = load_causal_lm(model_path, device, local_only=local_only)
        lines_out = []
        for i, row in enumerate(excerpt_rows):
            # Prompt templates can mix {title} and {excerpt}; both are populated here.
            prompt = build_explain_prompt(
                tok,
                instruction,
                excerpt=row["text"],
                title=str(row.get("title") or ""),
            )
            resp_text = generate_completion(tok, gen, prompt, device, max_new_tokens)
            lines_out.append(
                {
                    "doc_id": row["doc_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "prompt": prompt,
                    "response": resp_text,
                }
            )
            print(f"  [{slug}] generated {i + 1}/{len(excerpt_rows)}", flush=True)
        del gen
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(resp_path), exist_ok=True)
        with open(resp_path, "w", encoding="utf-8") as rf:
            for ln in lines_out:
                rf.write(json.dumps(ln, ensure_ascii=False) + "\n")
        response_by_model[slug] = lines_out
        print(f"[generate] saved {resp_path} ({time.perf_counter() - gen_model_start:.1f}s)")
    _mark_stage_done(out_dir, "generate")
    return response_by_model


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
    response_by_model = _load_response_by_model(out_dir, args_models)
    artifacts = run_info.setdefault("artifacts", {})
    for model_path in args_models:
        slug = sanitize_model_slug(model_path)
        w_path = _response_word_npy(out_dir, slug)
        d_path = _response_doc_npy(out_dir, slug)
        want_word, want_doc = _aggregation_outputs(aggregation_level)
        required_paths = []
        if want_word:
            required_paths.append(w_path)
        if want_doc:
            required_paths.append(d_path)
        if required_paths and all(os.path.isfile(p) for p in required_paths) and not force_redo:
            print(f"[embed_responses] skip (exists): {', '.join(required_paths)}")
            continue
        resp_docs = response_by_model[slug]
        # Rebuild into the same schema used for excerpts so embedding stays generic.
        resp_rows = [
            {
                "doc_id": int(x["doc_id"]),
                "category": x["category"],
                "title": x["title"],
                "text": x["response"],
                "segment": "response",
            }
            for x in resp_docs
        ]
        r_dir = os.path.join(out_dir, "responses", slug)
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
        artifacts[f"responses_{slug}_token_embeddings"] = os.path.join(
            r_dir, "token_embeddings.npy"
        )
        if want_word:
            artifacts[f"responses_{slug}_word_embeddings"] = w_path
        if want_doc:
            artifacts[f"responses_{slug}_document_embeddings"] = d_path
        run_info["timings_sec"].setdefault("models", {}).setdefault(slug, {})[
            "embedding_response"
        ] = round(time.perf_counter() - t0, 3)
        print(f"[embed_responses] {slug} ({run_info['timings_sec']['models'][slug]['embedding_response']}s)")
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
    cka_records: list[dict[str, Any]] = []
    agg_specs: list[tuple[str, str, Any, Any]] = []
    if want_word:
        agg_specs.append(("word", "", _excerpt_word_npy, _response_word_npy))
    if want_doc:
        agg_specs.append(("document", "_document", _excerpt_doc_npy, _response_doc_npy))

    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            si, sj = slugs[i], slugs[j]
            for agg_name, seg_suffix, excerpt_path_fn, response_path_fn in agg_specs:
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
                if not skip_generate:
                    r_a = response_path_fn(out_dir, si)
                    r_b = response_path_fn(out_dir, sj)
                    if os.path.isfile(r_a) and os.path.isfile(r_b):
                        rec_r = {
                            "segment": "response",
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
                        resp_name = f"response{seg_suffix}__{si}__vs__{sj}{slice_suffix}.json"
                        with open(os.path.join(cka_dir, resp_name), "w", encoding="utf-8") as cf:
                            json.dump(rec_r, cf, indent=2)
                        if rec_r.get("error"):
                            print(f"CKA response ({agg_name}) {si} vs {sj}: {rec_r['error']}")
                        else:
                            print(
                                f"CKA response ({agg_name}) {si} vs {sj}: "
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
                doc_ids_allow=cka_doc_ids,
            )
            by_cat_obj = {
                "analysis": "document_by_category",
                "out_dir": out_dir,
                "models": slugs,
                "skip_generate": bool(skip_generate),
                "chunk_rows": int(cka_chunk_rows),
                "results": by_cat.get("results", []),
                "focused": by_cat.get("focused", {}),
            }
            with open(by_cat_index, "w", encoding="utf-8") as f:
                json.dump(by_cat_obj, f, indent=2)
            run_info["cka_document_by_category"] = by_cat_obj
            print(
                "[cka] document-by-category comparisons:",
                len(by_cat_obj["results"]),
            )
            f = by_cat_obj.get("focused", {})
            print(
                "      cross_model_same_segment=",
                len(f.get("cross_model_same_segment", [])),
                "within_model_excerpt_vs_response=",
                len(f.get("within_model_excerpt_vs_response", [])),
                "cross_model_cross_segment=",
                len(f.get("cross_model_cross_segment", [])),
            )
        run_info.setdefault("artifacts", {})["cka_document_by_category"] = by_cat_index

    _mark_stage_done(out_dir, "cka")


def _merge_run_info(out_dir: str, run_info: dict[str, Any]) -> None:
    path = os.path.join(out_dir, "run_info.json")
    if os.path.isfile(path):
        prev = _load_json(path)
        prev.setdefault("timings_sec", {}).setdefault("models", {})
        prev["timings_sec"]["models"].update(run_info.get("timings_sec", {}).get("models", {}))
        for k in ("artifacts", "cka", "num_docs"):
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
        "--instruction",
        default="Explain the following: {title}",
        help="Prompt template; {title} and/or {excerpt}",
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

    keys = _instruction_placeholders(args.instruction)
    bad = keys - _ALLOWED_INSTRUCTION_KEYS
    if bad:
        p.error(f"--instruction has unknown placeholder(s): {bad}; allowed: excerpt, title")
    if not keys:
        p.error("--instruction must contain at least one of {excerpt}, {title}")

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

    if args.stage == "all":
        # Single-process full run for convenience and simple reproducibility.
        os.makedirs(out_dir, exist_ok=True)
        excerpt_rows, run_info = run_stage_init(
            out_dir=out_dir,
            corpus=args.corpus,
            words=args.words,
            models=args.models,
            instruction=args.instruction,
            instruction_keys=sorted(keys),
            skip_generate=args.skip_generate,
            skip_cka=args.skip_cka,
            cka_chunk_rows=args.cka_chunk_rows,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            local_only=local_only,
            device=device,
            aggregation_level=args.aggregation_level,
            force_redo=force_redo,
        )
        response_by_model: dict[str, list[dict[str, Any]]] = {}
        if not args.skip_generate:
            response_by_model = run_stage_generate(
                out_dir=out_dir,
                args_models=args.models,
                excerpt_rows=excerpt_rows,
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
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
            instruction=args.instruction,
            instruction_keys=sorted(keys),
            skip_generate=args.skip_generate,
            skip_cka=args.skip_cka,
            cka_chunk_rows=args.cka_chunk_rows,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            local_only=local_only,
            device=device,
            aggregation_level=args.aggregation_level,
            force_redo=force_redo,
        )
    else:
        cfg = _load_pipeline_config(out_dir)
        _verify_models_match(cfg["models"], args.models)
        if args.instruction != cfg["instruction"]:
            raise SystemExit(
                f"--instruction must match {CONFIG_NAME} exactly.\n"
                f"  config: {cfg['instruction']!r}\n  cli:    {args.instruction!r}"
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
            run_stage_generate(
                out_dir=out_dir,
                args_models=args.models,
                excerpt_rows=excerpt_rows,
                instruction=cfg["instruction"],
                max_new_tokens=cfg["max_new_tokens"],
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
