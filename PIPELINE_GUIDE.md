# Categorized corpus pipeline guide

This document describes how to run the **excerpt → embeddings → (optional) generation → response embeddings → linear CKA** pipeline, including **pausing between stages** and **resuming** without redoing finished work.

**Exhaustive flag list (all scripts, defaults, optional flags):** [CLI_REFERENCE.md](CLI_REFERENCE.md)

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
| `generate` | For each `--models` entry: loads the causal LM once and runs every configured **generation task**. The primary task writes `responses/<slug>/responses.jsonl`; adding `--summarize` also writes `summaries/<slug>/responses.jsonl` (each line includes `segment`, `prompt`, and `response`). Generation length is capped by `--max_new_tokens` (new tokens) and, per task, a word cap (`--max_generated_words` for `response`, `--summarize_words` for `summary`, or each excerpt's word count when `--match_abstract_length` is set)—see **Generation limits** below. Skipped if `skip_generate` was set at init. |
| `embed_excerpts` | Token embeddings + aggregation (`--aggregation_level word|document|both`) for excerpt text; artifacts under `excerpts/<slug>/`. |
| `embed_responses` | Same for each generated task under its directory (`responses/<slug>/`, `summaries/<slug>/`, …), tagging rows with the task's segment label. **Own-embed only:** the model that wrote the text also embeds it. Requires generation outputs unless `skip_generate`. |
| `embed_cross_responses` | Cross-embed generations: each model embeds every *other* model's response/summary text. Writes under `<task>/<text_slug>/by_embedder/<embedder_slug>/`. Skippable on old runs as a standalone stage; included in `--stage all`. No-op when `skip_generate` or fewer than two models. |
| `cka` | Pairwise **linear CKA** on aligned embedding rows from the configured aggregation level (`word`, `document`, or both); writes `cka/cka_index.json` and per-pair JSON files. For document aggregation, also writes a category-stratified matrix at `cka/document_cka_by_category.json`. When `by_embedder/` artifacts exist, also records same-text cross-embedder and same-embedder cross-text comparisons (extra `comparison` fields in `cka_index.json`; `cross_embed` section in the document matrix). Honors `skip_cka` from `pipeline_config.json` on staged runs. |

**Default:** `--stage all` runs `init` through `cka` in one process (same order as above).

## Full run in one command

From the repo root (adjust paths and venv):

```powershell
cd F:\code\Independent Study
.\.venv313\Scripts\python.exe .\run_categorized_corpus.py --corpus "F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl" --out_dir "F:\code\Independent Study\outputs\my_run4" --models "F:\quantas\models\google\gemma-3-1b-it" "F:\quantas\models\Qwen\Qwen3.5-0.8B" --aggregation_level "document" --words 200 --max_new_tokens 800 --max_generated_words 200 --batch_size 4 --chunk_size 768 --stage init
```

Omit `--max_generated_words` if you only want the `--max_new_tokens` cap on generation.

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

# Same generation caps as init (required). Add `--max_generated_words N` here too if you set it at init.
& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage generate --max_new_tokens 128

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_excerpts

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_responses

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_cross_responses

& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage cka
```

You can stop between lines and resume days later; see **Resume and overwrite** below.

To add cross-embeds to an **old run** that already finished CKA:

```powershell
& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage embed_cross_responses
& $PY $RC --corpus $CORPUS --out_dir $OUT --models $M1 $M2 --stage cka --overwrite
```

**Important:** After `init`, every later stage must use CLI flags that **match** `pipeline_config.json` exactly for: `--instruction`, `--word_count_prompt`, `--match_abstract_length`, `--summarize` (with `--summarize_instruction` / `--summarize_words`), `--batch_size`, `--chunk_size`, `--aggregation_level`, `--skip_generate`, `--skip_cka`, `--cka_chunk_rows`, and local-only vs `--allow_remote`. For the **`generate`** stage only, `--max_new_tokens` and `--max_generated_words` must each match the effective saved values (see **Legacy `max_words` in config** under Troubleshooting). If you need to change those, redo **`init`** with `--overwrite` (or a new `--out_dir`).

## Resume and overwrite

| Flag | Behavior |
|------|----------|
| `--resume` | **Default.** If an expected output file already exists (e.g. `responses/<slug>/responses.jsonl`, excerpt/response word `.npy`, `cka/cka_index.json`), that piece of work is skipped. |
| `--no_resume` | Do not skip based on existing files (still subject to normal errors if inputs are missing). |
| `--overwrite` | Same practical effect as forcing a redo for stage outputs: re-embed / regenerate / re-CKA as applicable. |

`pipeline_state.json` records which stages completed; skipped-by-config stages (e.g. CKA when `skip_cka` is true) are also marked complete so the run does not look “stuck.”

## Configuration files under `--out_dir`

- **`pipeline_config.json`** — Frozen settings from `init` (corpus path, excerpt `--words`, model paths, instruction template, generation caps `max_new_tokens` / optional `max_generated_words` / `match_abstract_length`, batch/chunk sizes, `skip_generate`, `skip_cka`, etc.). Staged runs load this and **validate** your CLI against it.
- **`pipeline_state.json`** — List of completed stage names (`init`, `generate`, …).
- **`run_info.json`** — Timings, artifact paths, CKA summary records (merged across stages). Includes **`cli_command`** (Windows **cmd.exe** one-liner) and **`cli_argv`** (raw argv list) from the `init` invocation.

## Prompt template (`--instruction`)

Must contain at least one of `{title}` or `{excerpt}`. Allowed placeholders: `{title}`, `{excerpt}`, `{word_count}`.

Default: `Explain the following: {title}`

### Word-count preset (`--word_count_prompt`)

Use `--word_count_prompt` to switch to the preset instruction:

`In {word_count} words, explain the following: {title}`

This requires `--max_generated_words` or `--match_abstract_length` (the same value is injected into `{word_count}` in the prompt and used as the hard generation cap; with `--match_abstract_length`, that value is each excerpt's word count). Example:

```powershell
.\.venv313\Scripts\python.exe .\run_categorized_corpus.py `
  --corpus ".\data\wiki_tree_corpus_for_cka.jsonl" `
  --out_dir ".\outputs\my_run_wordcount" `
  --models "F:\path\to\gemma-3-1b-it" "F:\path\to\Qwen3.5-0.8B" `
  --aggregation_level "document" `
  --words 200 `
  --max_new_tokens 1500 `
  --max_generated_words 200 `
  --word_count_prompt `
  --batch_size 4 `
  --chunk_size 768 `
  --stage all
```

You can also put `{word_count}` in a custom `--instruction` (without the preset flag); `--max_generated_words` or `--match_abstract_length` is still required when `{word_count}` appears.

The `word_count_prompt` boolean is stored in `pipeline_config.json` and must match on resumed staged runs.

### Additive summarize task (`--summarize`)

Analysis has three forks: **model** (each `--models` entry), **segment** (`excerpt` = source text vs generated text), and now **generation task**. `--summarize` adds a second generation task *in the same run*, so a single pipeline produces both the explain-style `response` and a `summary` segment.

- Prompt (default): `Summarize the following in {word_count} words: {excerpt}` (override with `--summarize_instruction`).
- Word target: `--summarize_words N`; when omitted it **defaults to `--max_generated_words`** (the response task's cap). It fills `{word_count}` and hard-caps the generated summary. With `--match_abstract_length`, both tasks use each excerpt's word count instead (do not pass `--summarize_words`).
- Outputs: `summaries/<slug>/responses.jsonl` and embeddings, mirroring `responses/<slug>/`.
- CKA: the `summary` segment is embedded and compared like any other, so the document-by-category matrix spans **all `model × segment` pairs** (excerpt / response / summary) automatically.

The task list is frozen in `pipeline_config.json` as `generation_tasks`. Re-pass `--summarize` (and any `--summarize_instruction` / `--summarize_words`) on staged runs so the CLI matches config. Older configs without `generation_tasks` resume as a single `response` task.

```powershell
& $PY $RC `
  --corpus ".\data\wiki_tree_corpus_for_cka.jsonl" `
  --out_dir ".\outputs\my_run_summarize" `
  --models "F:\path\to\gemma-3-1b-it" "F:\path\to\Qwen3.5-0.8B" `
  --aggregation_level "document" `
  --words 200 `
  --max_new_tokens 1500 `
  --max_generated_words 200 `
  --word_count_prompt `
  --summarize `
  --summarize_words 50 `
  --batch_size 4 `
  --chunk_size 768 `
  --stage all
```

## Generation limits (`generate` stage only)

These flags affect **only** causal LM decoding when writing `responses/<slug>/responses.jsonl`. They do **not** change how many words are taken from the corpus for excerpts (`--words` is separate) and do **not** impose a word limit on embedding inputs beyond whatever text was generated and saved.

| Flag | Role |
|------|------|
| `--max_new_tokens` | Hard ceiling on **new** tokens from `model.generate` (per completion). May finish earlier on EOS. |
| `--max_generated_words` | Optional. Early stopping once the **decoded generated completion** (tokens after the prompt—the assistant reply) reaches **N** whitespace-separated words; hyphenated forms like `well-known` count as **one** word. Still limited by `--max_new_tokens` and EOS. After stop, the completion may be trimmed to at most **N** words so the cap is strict. Incompatible with `--match_abstract_length`. |
| `--match_abstract_length` | Optional alternative to fixed `--max_generated_words` / `--summarize_words`. For each document, the response and summary word target/cap equals that excerpt's whitespace word count (after `--words` truncation). Short docs therefore get shorter caps than `--words`. Fills `{word_count}` and hard-caps decoding the same way as a fixed N. |

These values are recorded at **`init`** and must be repeated exactly on **`--stage generate`** in staged runs.

## Common CLI options

See [CLI_REFERENCE.md](CLI_REFERENCE.md) for the complete table including `--stage`, `--resume`, `--overwrite`, and corpus-building scripts. Summary of the main pipeline flags:

| Option | Notes |
|--------|--------|
| `--corpus` | Required for every invocation (path must stay consistent with config after `init`). |
| `--out_dir` | Required after `init`; optional on `init` (auto `outputs/categorized_*`). |
| `--stage` | `all` (default), `init`, `generate`, `embed_excerpts`, `embed_responses`, `embed_cross_responses`, or `cka`. |
| `--models` | One or more **local** checkpoint directories; order and paths must match `pipeline_config.json` when resuming. |
| `--words` | First N words per doc used as excerpt (stored at init). |
| `--instruction` | Prompt template; see above. |
| `--word_count_prompt` | Use the word-count preset instruction; requires `--max_generated_words` or `--match_abstract_length`. Stored at init. |
| `--max_new_tokens` | **Generate** stage: max new tokens per completion; must match config for `--stage generate`. |
| `--max_generated_words` | **Generate** stage only (optional): max words in the decoded **generated reply**; see **Generation limits**. Must match config for `--stage generate`. Required with `--word_count_prompt` or `{word_count}` in instruction unless `--match_abstract_length`. |
| `--match_abstract_length` | **Generate** stage alternative: per-excerpt word caps for response/summary; see **Generation limits**. Stored at init; incompatible with fixed word-cap flags. |
| `--batch_size`, `--chunk_size` | Embedding batching (stored at init). |
| `--aggregation_level` | Aggregation output from token embeddings: `word` (default), `document`, or `both` (stored at init). CKA runs on the available aggregation output(s). |
| `--skip_generate` | Excerpt-only pipeline: no `responses/` generation; response embedding and response CKA pairs are omitted. |
| `--skip_cka` | Skip writing `cka/` (stored at init; staged `cka` stage reads this from config). |
| `--resume` / `--no_resume` / `--overwrite` | Control skipping vs redoing existing stage outputs. |
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
    by_embedder/<other_model_slug>/           (from embed_cross_responses)
      token_embeddings.npy
      word_embeddings_merged_agnostic.npy
      document_embeddings_merged_agnostic.npy
  summaries/<model_slug>/                     (only with --summarize)
    responses.jsonl
    token_embeddings.npy
    word_embeddings_merged_agnostic.npy
    document_embeddings_merged_agnostic.npy   (if --aggregation_level document|both)
    by_embedder/<other_model_slug>/           (from embed_cross_responses; same layout)
  cka/
    cka_index.json
    document_cka_by_category.json   (if --aggregation_level document|both; may include cross_embed)
    leave_k_out_pooled.json         (optional; scripts/cka_leave_k_out.py)
    leave_k_out_pooled.md           (optional; scripts/cka_leave_k_out.py)
    row_permutation_pooled.json     (optional; scripts/cka_row_permutation.py)
    row_permutation_pooled.md       (optional; scripts/cka_row_permutation.py)
    excerpt__<a>__vs__<b>.json
    excerpt_document__<a>__vs__<b>.json      (if --aggregation_level document|both)
    excerpt__<a>__vs__<b>__<slice_suffix>.json   (optional, when --cka_filter / --cka_doc_ids used)
    response__<a>__vs__<b>.json   (word, if generation ran and word response npys exist)
    response_document__<a>__vs__<b>.json   (document, if generation ran and document response npys exist)
    summary__<a>__vs__<b>.json   (word, only with --summarize)
    summary_document__<a>__vs__<b>.json   (document, only with --summarize)
    # Cross-embed enrichments (when by_embedder/ artifacts exist), e.g.:
    # response_document__same_text__<text>__emb_<a>__vs__<b>.json
    # excerpt_vs_response_document__emb_<e>__text_corpus__vs__<text>.json
```

Exact filenames follow `sanitize_model_slug()` (derived from the model directory name). Pairwise JSON records include `n_rows` (after slice), `n_rows_total`, and optional `row_slice` describing filters.

## Wikipedia helper script

`scripts/run_wiki_tree_cka_pipeline.py` builds `wiki_tree_corpus_for_cka`-style JSONL from `data/wiki_tree_random.jsonl` and then runs **`run_categorized_corpus.py` with `--stage all`** (full pipeline in one subprocess). It forwards `--max-new-tokens`, optional **`--max-generated-words`**, **`--word-count-prompt`**, and **`--instruction`**. It does **not** expose `--stage`, `--resume`, `--overwrite`, `--cka_filter`, or `--allow_remote`; for multi-day staged runs, call `run_categorized_corpus.py` yourself with the `--corpus-out` path from that script (or your own JSONL). See [CLI_REFERENCE.md](CLI_REFERENCE.md) for all wrapper flags and defaults.

## Ad-hoc CKA on existing `.npy` files

For manual pairwise checks outside the pipeline directory layout, see `cka_word_embeddings.py` (includes optional `--bootstrap`, `--leave_k_out` / `--drop_k`, and `--row_permutation`).

Optional post-hoc **leave-k-out** stability on a finished run’s focused pooled document pairs: `scripts/cka_leave_k_out.py --out_dir <run>` (defaults: drop 10 rows, 100 reps; writes `cka/leave_k_out_pooled.json` and `.md`).

Optional post-hoc **row-permutation** null (permute Y topic rows, mean/std over reps): `scripts/cka_row_permutation.py --out_dir <run>` (defaults: 100 reps, seed 42; writes `cka/row_permutation_pooled.json` and `.md`).

## Troubleshooting

- **`--models must match pipeline_config.json`** — Use the same absolute paths (or the same paths as stored at init). Re-init with a new `--out_dir` if you switched checkpoints intentionally.
- **`--instruction must match`** — Edit only by re-running `init` with `--overwrite` or a new output directory.
- **`--word_count_prompt must match`** — Same as instruction; re-init to switch between default and word-count preset prompts.
- **CKA needs at least two models** — With a single model, the CKA stage prints a skip message and does not write pairwise matrices.
