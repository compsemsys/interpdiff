# Categorized corpus pipeline guide

This document describes how to run the **excerpt → embeddings → (optional) generation → response embeddings → linear CKA** pipeline, including **pausing between stages** and **resuming** without redoing finished work.

## Entry points

| Script | Role |
|--------|------|
| `run_categorized_corpus.py` | Main implementation and CLI. |


## Corpus format (JSONL)

Each line is one JSON object with at least:

- `doc_id` (integer)
- `category` (string; may be empty)
- `title` (string)
- `text` (full article or passage; the pipeline uses the first **N** words as the excerpt)

Example sources in this repo: `data/wiki_tree_corpus_for_cka.jsonl`, or JSONL produced by `wiki_fetch.py` / `wiki_tree_random_articles.py` after mapping fields.

## Pipeline stages

| Stage | What it does |
|-------|----------------|
| `init` | Writes excerpt JSONL, copies/records corpus metadata, creates `pipeline_config.json` and `pipeline_state.json` under `--out_dir`. |
| `generate` | For each `--models` entry: loads the causal LM, builds prompts from `--instruction`, writes `responses/<slug>/responses.jsonl` (includes `prompt` and `response`). Skipped if `skip_generate` was set at init. |
| `embed_excerpts` | Token embeddings + aggregation (`--aggregation_level word|document|both`) for excerpt text; artifacts under `excerpts/<slug>/`. |
| `embed_responses` | Same for generated responses under `responses/<slug>/`. Requires generation outputs unless `skip_generate`. |
| `cka` | Pairwise **linear CKA** on aligned embedding rows from the configured aggregation level (`word`, `document`, or both); writes `cka/cka_index.json` and per-pair JSON files. For document aggregation, also writes a category-stratified matrix at `cka/document_cka_by_category.json`. Honors `skip_cka` from `pipeline_config.json` on staged runs. |

**Default:** `--stage all` runs `init` through `cka` in one process (same order as above).

## Full run in one command

From the repo root (adjust paths and venv):

```powershell
cd F:\code\Independent Study
.\.venv313\Scripts\python.exe .\run_categorized_corpus.py `
  --corpus "F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl" `
  --out_dir "F:\code\Independent Study\outputs\my_run2" `
  --models "F:\path\to\gemma-3-1b-it" "F:\path\to\Qwen3.5-0.8B" `
  --aggregation_level "document" `
  --words 200 `
  --max_new_tokens 128 `
  --batch_size 4 `
  --chunk_size 768 `
  --stage "init"
```

If `--out_dir` is omitted on **`--stage init` only**, a timestamped directory under `outputs/categorized_<timestamp>/` is created.

## Pausing between stages (staged runs)

Run **one stage per invocation**, reusing the same `--out_dir` and the **same** `--models` paths as recorded in `pipeline_config.json`.

```powershell
$CORPUS = "F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl"
$OUT = "F:\code\Independent Study\outputs\my_run"
$M1 = "F:\path\to\gemma-3-1b-it"
$M2 = "F:\path\to\Qwen3.5-0.8B"
$PY = "F:\code\Independent Study\.venv313\Scripts\python.exe"
$RC = "F:\code\Independent Study\run_categorized_corpus.py"

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage init --words 200 --max_new_tokens 128 --batch_size 4 --chunk_size 768

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage generate

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_excerpts

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_responses

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage cka
```

You can stop between lines and resume days later; see **Resume and overwrite** below.

**Important:** After `init`, every later stage must use CLI flags that **match** `pipeline_config.json` exactly for: `--instruction`, `--batch_size`, `--chunk_size`, `--aggregation_level`, `--skip_generate`, `--skip_cka`, `--cka_chunk_rows`, and local-only vs `--allow_remote`. `--max_new_tokens` must match when running **`generate`**. If you need to change those, redo **`init`** with `--overwrite` (or a new `--out_dir`).

## Resume and overwrite

| Flag | Behavior |
|------|----------|
| `--resume` | **Default.** If an expected output file already exists (e.g. `responses/<slug>/responses.jsonl`, excerpt/response word `.npy`, `cka/cka_index.json`), that piece of work is skipped. |
| `--no_resume` | Do not skip based on existing files (still subject to normal errors if inputs are missing). |
| `--overwrite` | Same practical effect as forcing a redo for stage outputs: re-embed / regenerate / re-CKA as applicable. |

`pipeline_state.json` records which stages completed; skipped-by-config stages (e.g. CKA when `skip_cka` is true) are also marked complete so the run does not look “stuck.”

## Configuration files under `--out_dir`

- **`pipeline_config.json`** — Frozen settings from `init` (corpus path, word count, model paths, instruction template, batch/chunk sizes, `skip_generate`, `skip_cka`, etc.). Staged runs load this and **validate** your CLI against it.
- **`pipeline_state.json`** — List of completed stage names (`init`, `generate`, …).
- **`run_info.json`** — Timings, artifact paths, CKA summary records (merged across stages).

## Prompt template (`--instruction`)

Must contain at least one of `{title}` or `{excerpt}`. Only these placeholders are allowed.

Default: `Explain the following: {title}`

## Common CLI options

| Option | Notes |
|--------|--------|
| `--corpus` | Required for every invocation (path must stay consistent with config after `init`). |
| `--out_dir` | Required after `init`; optional on `init` (auto `outputs/categorized_*`). |
| `--models` | One or more **local** checkpoint directories; order and paths must match `pipeline_config.json` when resuming. |
| `--words` | First N words per doc used as excerpt (stored at init). |
| `--instruction` | Prompt template; see above. |
| `--max_new_tokens` | Generation length; must match config for `--stage generate`. |
| `--batch_size`, `--chunk_size` | Embedding batching (stored at init). |
| `--aggregation_level` | Aggregation output from token embeddings: `word` (default), `document`, or `both` (stored at init). CKA runs on the available aggregation output(s). |
| `--skip_generate` | Excerpt-only pipeline: no `responses/` generation; response embedding and response CKA pairs are omitted. |
| `--skip_cka` | Skip writing `cka/` (stored at init; staged `cka` stage reads this from config). |
| `--cka_chunk_rows` | Chunk size for chunked linear CKA (default 4096). |
| `--cka_filter` | **CKA stage only:** repeat `KEY=VALUE`; keep word rows whose metadata matches **all** filters (equality on string form, except `doc_id` compared as int). Typical key: `category` (same string as in corpus JSONL). |
| `--cka_doc_ids` | **CKA stage only:** comma-separated `doc_id` values; combined with filters by intersection. |
| `--cka_slice_label` | **CKA stage only:** short safe suffix for slice JSON files; if omitted when slicing, a hash of the filter spec is used. |
| `--allow_remote` | Allow Hugging Face Hub downloads; default is **local files only**. |

## CKA row slicing (by category or doc)

Aggregated `.npy` files store one row per embedding unit (word or document, based on aggregation) with metadata (`doc_id`, `category`, `title`, `segment`, …). Global CKA uses every row. To restrict analysis:

```powershell
& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage cka `
  --cka_filter "category=Category:Machine_learning"
```

Repeat `--cka_filter` for multiple **AND** conditions. Values are matched exactly (including spaces); use the first `=` only as the separator between key and value so values may contain `=`.

- **Unsliced** runs still write `excerpt__<a>__vs__<b>.json` (word) and/or `excerpt_document__<a>__vs__<b>.json` (document) and skip the CKA stage if `cka_index.json` already exists (`--resume`).
- **Sliced** runs write `excerpt__<a>__vs__<b>__<label_or_hash>.json`, do **not** hit that skip, and **merge** into `cka_index.json` (replacing prior entries with the same segment, model pair, and slice spec).

The standalone script supports the same filters:

```powershell
.\.venv313\Scripts\python.exe .\cka_word_embeddings.py `
  --a "...\word_embeddings_merged_agnostic.npy" `
  --b "...\word_embeddings_merged_agnostic.npy" `
  --cka_filter "category=Category:Statistics"
```

## Artifacts (typical layout)

```
<out_dir>/
  pipeline_config.json
  pipeline_state.json
  run_info.json
  corpus_excerpts_used.jsonl
  corpus_source.jsonl   (or similar copy of source)
  excerpts/<model_slug>/
    token_embeddings.npy
    word_embeddings_merged_agnostic.npy
    document_embeddings_merged_agnostic.npy   (if --aggregation_level document|both)
  responses/<model_slug>/
    responses.jsonl
    token_embeddings.npy
    word_embeddings_merged_agnostic.npy
    document_embeddings_merged_agnostic.npy   (if --aggregation_level document|both)
  cka/
    cka_index.json
    document_cka_by_category.json   (if --aggregation_level document|both)
    excerpt__<a>__vs__<b>.json
    excerpt_document__<a>__vs__<b>.json      (if --aggregation_level document|both)
    excerpt__<a>__vs__<b>__<slice_suffix>.json   (optional, when --cka_filter / --cka_doc_ids used)
    response__<a>__vs__<b>.json   (word, if generation ran and word response npys exist)
    response_document__<a>__vs__<b>.json   (document, if generation ran and document response npys exist)
```

Exact filenames follow `sanitize_model_slug()` (derived from the model directory name). Pairwise JSON records include `n_rows` (after slice), `n_rows_total`, and optional `row_slice` describing filters.

## Wikipedia helper script

`scripts/run_wiki_tree_cka_pipeline.py` builds `wiki_tree_corpus_for_cka`-style JSONL from `data/wiki_tree_random.jsonl` and then runs **`run_categorized_corpus.py` with `--stage all`** (full pipeline in one subprocess). It does **not** expose `--stage`; for multi-day staged runs, call `run_categorized_corpus.py` yourself with the `--corpus-out` path from that script (or your own JSONL).

## Ad-hoc CKA on existing `.npy` files

For manual pairwise checks outside the pipeline directory layout, see `cka_word_embeddings.py`.

## Troubleshooting

- **`--models must match pipeline_config.json`** — Use the same absolute paths (or the same paths as stored at init). Re-init with a new `--out_dir` if you switched checkpoints intentionally.
- **`--instruction must match`** — Edit only by re-running `init` with `--overwrite` or a new output directory.
- **CKA needs at least two models** — With a single model, the CKA stage prints a skip message and does not write pairwise matrices.
