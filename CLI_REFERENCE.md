# CLI reference

Complete flag and option reference for the main scripts in this repo. For pipeline stages, resume behavior, and artifact layout, see [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).

Run any script with `--help` for the live argparse text:

```powershell
.\.venv313\Scripts\python.exe .\run_categorized_corpus.py --help
```

---

## Typical workflow

```text
wiki_category_library.py --refresh          # optional: refresh category metadata
wiki_tree_random_articles.py                # sample articles → data/wiki_tree_random.jsonl
scripts/run_wiki_tree_cka_pipeline.py         # OR build corpus + run full pipeline
  └─ run_categorized_corpus.py --stage all    # excerpt → generate → embed → CKA
scripts/render_document_cka_report.py       # optional: CKA matrix → markdown
```

Alternate corpus path: `wiki_fetch.py` → JSONL with the same `doc_id, category, title, text` schema.

---

## `run_categorized_corpus.py`

Main pipeline CLI. Alias: `pipeline_embed_explain_cka.py` (same `main()`).

**Required every invocation:** `--corpus`, `--models` (one or more local checkpoint directories).

**Default stage:** `--stage all` (runs `init` → `generate` → `embed_excerpts` → `embed_responses` → `cka`).

### All flags

| Flag | Default | Stored at `init`? | Used in stage(s) | Notes |
|------|---------|-------------------|------------------|-------|
| `--corpus` | *(required)* | yes | all | JSONL with `doc_id`, `category`, `title`, `text`. Path must match config when resuming. |
| `--out_dir` | auto on `init` only | — | all | Run directory. Required after `init`. If omitted on `init`, creates `outputs/categorized_<timestamp>/`. |
| `--models` | *(required)* | yes | generate, embed_*, cka | Local model checkpoint dirs. Order and paths must match `pipeline_config.json` when resuming. |
| `--stage` | `all` | — | — | `all`, `init`, `generate`, `embed_excerpts`, `embed_responses`, or `cka`. |
| `--words` | `200` | yes | init | First N whitespace-delimited words of each doc's `text` used as the excerpt. |
| `--instruction` | `Explain the following: {title}` | yes | generate | Prompt template. Placeholders: `{title}`, `{excerpt}`, `{word_count}`. Must contain at least one of `{title}` or `{excerpt}`. `{word_count}` requires `--max_generated_words` or `--match_abstract_length`. |
| `--word_count_prompt` | off | yes | generate | Preset: `In {word_count} words, explain the following: {title}`. Requires `--max_generated_words` or `--match_abstract_length`. Overrides `--instruction`. |
| `--summarize` | off | yes | generate, embed_responses, cka | **Additive** second generation task (segment `summary`). Runs *alongside* the response task from the same model load; outputs under `summaries/<slug>/`, embedded and CKA-compared as an extra segment (full model×segment matrix). Re-pass to match config on staged runs. |
| `--summarize_instruction` | `Summarize the following in {word_count} words: {excerpt}` | yes | generate | Prompt template for the `--summarize` task. Placeholders: `{title}`, `{excerpt}`, `{word_count}` (must contain at least one of `{title}`/`{excerpt}`). |
| `--summarize_words` | *(defaults to `--max_generated_words`)* | yes | generate | Word target/cap for the summary (fills `{word_count}` and hard-caps the generated summary). Omit to reuse the response task's word count. Incompatible with `--match_abstract_length`. |
| `--max_new_tokens` | `256` | yes | generate | Max **new** tokens per completion (`model.generate`). Must match config on `--stage generate`. Shared ceiling across all generation tasks. |
| `--max_generated_words` | *(none)* | yes | generate | Optional hard cap on decoded **assistant reply** words (whitespace-split; hyphenated = one word). Still bounded by `--max_new_tokens` and EOS. Required when `{word_count}` appears in instruction or when `--word_count_prompt` is set (unless `--match_abstract_length`). Must match config on `--stage generate`. Incompatible with `--match_abstract_length`. |
| `--match_abstract_length` | off | yes | generate | Per-document word target/cap: each response/summary uses that excerpt's word count (after `--words`) for `{word_count}` and the hard stop. Incompatible with `--max_generated_words` / `--summarize_words`. Enables `{word_count}` / `--word_count_prompt` without a fixed N. |
| `--batch_size` | `12` | yes | embed_* | Embedding batch size (grouped by sequence length). |
| `--chunk_size` | `1024` | yes | embed_* | Max tokens per forward pass when embedding long texts. |
| `--aggregation_level` | `word` | yes | embed_*, cka | `word`, `document`, or `both`. Controls which `.npy` aggregation outputs are written and which CKA pairs run. |
| `--skip_generate` | off | yes | generate, embed_responses, cka | Excerpt-only run: no causal LM responses; no response embeddings or response CKA. |
| `--skip_cka` | off | yes | cka | Skip writing `cka/` outputs. |
| `--cka_chunk_rows` | `4096` | yes | cka | Row chunk size for chunked linear CKA. |
| `--allow_remote` | off (local only) | yes | generate, embed_* | Allow Hugging Face Hub downloads. Default: local files only. Must match config when resuming. |
| `--resume` | on | — | all | Skip work when expected output files already exist. |
| `--no_resume` | — | — | all | Do not skip existing outputs. |
| `--overwrite` | off | — | all | Force redo of stage outputs (re-embed, regenerate, re-CKA as applicable). |
| `--cka_filter` | *(none)* | no | cka only | Repeatable `KEY=VALUE`. Keep embedding rows whose metadata matches **all** filters (AND). Example: `--cka_filter category=Category:Machine_learning`. |
| `--cka_doc_ids` | *(none)* | no | cka only | Comma-separated `doc_id` list; intersected with `--cka_filter`. |
| `--cka_slice_label` | auto hash | no | cka only | Filename suffix for sliced CKA JSON. Only relevant when filtering. |

### Frozen config (must match on resumed staged runs)

After `init`, these must match `pipeline_config.json` on every later stage:

`--corpus`, `--words`, `--models`, `--instruction`, `--word_count_prompt`, `--match_abstract_length`, `--summarize` (and its `--summarize_instruction` / `--summarize_words`), `--skip_generate`, `--skip_cka`, `--cka_chunk_rows`, `--batch_size`, `--chunk_size`, `--aggregation_level`, local-only vs `--allow_remote`.

Additionally on `--stage generate` only: `--max_new_tokens`, `--max_generated_words`.

To change frozen settings, re-run `init` with `--overwrite` or use a new `--out_dir`.

### Prompt examples

```powershell
# Default (title only)
--instruction "Explain the following: {title}"

# Include excerpt text
--instruction "Explain the following article titled {title}: {excerpt}"

# Word-count preset (prompt + hard cap both use 200)
--word_count_prompt --max_generated_words 200 --max_new_tokens 1500

# Custom word-count wording
--instruction "Summarize in {word_count} words: {title}" --max_generated_words 100

# Additive summarize task (segment "summary") alongside the response task.
# summary word target defaults to --max_generated_words (200) unless --summarize_words is set.
--word_count_prompt --max_generated_words 200 --summarize
--word_count_prompt --max_generated_words 200 --summarize --summarize_words 50

# Per-excerpt caps: each doc's response/summary targets that excerpt's word count
--word_count_prompt --match_abstract_length --summarize --words 200
```

### Generation tasks (segments)

Each run generates one or more **tasks**, recorded in `pipeline_config.json` as `generation_tasks`. The first is always the explain-style `response` task (segment `response`, dir `responses/`). Adding `--summarize` appends a `summary` task (segment `summary`, dir `summaries/`). Every task is embedded and enters CKA as its own segment, so the document-by-category matrix spans all `model × segment` pairs. `--max_new_tokens` is a single ceiling shared by all tasks; per-task word targets come from `--max_generated_words` (response) and `--summarize_words` (summary, defaulting to the former), or from each excerpt's length when `--match_abstract_length` is set.

### Full-run example

```powershell
$PY = ".\.venv313\Scripts\python.exe"
$RC = ".\run_categorized_corpus.py"

& $PY $RC `
  --corpus ".\data\wiki_tree_corpus_for_cka.jsonl" `
  --out_dir ".\outputs\my_run" `
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

---

## `scripts/run_wiki_tree_cka_pipeline.py`

Convenience wrapper: converts `wiki_tree_random.jsonl` → pipeline corpus schema, then subprocesses `run_categorized_corpus.py` with `--stage all`.

Does **not** expose `--stage`, `--resume`, `--overwrite`, `--cka_filter`, or `--allow_remote`. For staged multi-day runs, build the corpus with `--corpus-out` and call `run_categorized_corpus.py` directly.

| Flag | Default | Notes |
|------|---------|-------|
| `--wiki-jsonl` | `data/wiki_tree_random.jsonl` | Source JSONL from `wiki_tree_random_articles.py`. |
| `--corpus-out` | `data/wiki_tree_corpus_for_cka.jsonl` | Normalized corpus written for the pipeline. |
| `--out-dir` | auto `outputs/categorized_*` | Forwarded as `--out_dir`. |
| `--gemma` | env `GEMMA3_MODEL_PATH` or `F:\quantas\models\google\gemma-3-1b-it` | First model path. |
| `--qwen` | env `QWEN35_MODEL_PATH` or `F:\quantas\models\Qwen\Qwen3.5-0.8B` | Second model path. |
| `--words` | `256` | Excerpt length (wrapper default differs from pipeline default `200`). |
| `--batch-size` | `4` | Forwarded to `--batch_size`. |
| `--chunk-size` | `768` | Forwarded to `--chunk_size`. |
| `--aggregation-level` | `word` | `word`, `document`, or `both`. |
| `--max-new-tokens` | `128` | Forwarded to `--max_new_tokens`. |
| `--max-generated-words` | *(none)* | Forwarded to `--max_generated_words`. |
| `--match-abstract-length` | off | Forwarded to `--match_abstract_length`. |
| `--instruction` | `Explain the following: {title}` | Forwarded to `--instruction`. |
| `--word-count-prompt` | off | Forwarded to `--word_count_prompt`; requires `--max-generated-words` or `--match-abstract-length`. |
| `--skip-generate` | off | Excerpt embeddings only. |
| `--skip-cka` | off | Forwarded to `--skip_cka`. |

Example with word-count prompt:

```powershell
.\.venv313\Scripts\python.exe .\scripts\run_wiki_tree_cka_pipeline.py `
  --out-dir ".\outputs\my_run" `
  --words 200 `
  --max-new-tokens 1500 `
  --max-generated-words 200 `
  --word-count-prompt `
  --aggregation-level document
```

---

## `wiki_tree_random_articles.py`

Weighted random sampler using `wiki_category_library.json`. Writes JSONL with intro extracts.

| Flag | Default | Notes |
|------|---------|-------|
| `--library` | `wiki_category_library.json` | Category metadata (root + direct subcategories). |
| `--roots` | all entries | Limit to specific root titles, e.g. `"Science" "Culture"`. |
| `--per-group` | `10` | Articles sampled per library entry. |
| `--out` | `data/wiki_tree_random.jsonl` | Output JSONL path. |
| `--seed` | *(none)* | RNG seed for reproducibility. |
| `--delay_sec` | `0.15` | Pause between API calls. |
| `--api` | env `WIKIMEDIA_API_URL` or enwiki | MediaWiki API endpoint. |
| `--user_agent` | project default | `User-Agent` header. |
| `--api_key` | env `WIKIMEDIA_API_KEY` | Optional API key. |
| `--max-sample-attempts` | `500` | Resample limit per root (collisions / stale counts). |

```powershell
.\.venv313\Scripts\python.exe .\wiki_tree_random_articles.py --per-group 10 --seed 42
```

---

## `wiki_fetch.py`

Fetch Wikipedia intro extracts by explicit category list (simpler than tree sampling).

| Flag | Default | Notes |
|------|---------|-------|
| `--categories` | *(required)* | One or more categories, e.g. `Machine_learning` or `Category:Physics`. |
| `--per_category` | `10` | Articles picked per category. |
| `--member_pool` | same as `--per_category` | List up to N member titles from API, then sample `--per_category` locally. |
| `--seed` | *(none)* | RNG seed when `member_pool > per_category`. |
| `--out` | `data/corpus_wiki.jsonl` | Output JSONL (`doc_id`, `category`, `title`, `text`). |
| `--api` | env or enwiki | API endpoint. |
| `--user_agent` | project default | `User-Agent` header. |
| `--api_key` | env `WIKIMEDIA_API_KEY` | Optional API key. |
| `--delay_sec` | `0.5` | Pause between API calls. |

```powershell
.\.venv313\Scripts\python.exe .\wiki_fetch.py --categories Machine_learning Physics --per_category 20 --out data/corpus_wiki.jsonl
```

---

## `wiki_category_library.py`

Refresh or inspect the local category library used by tree sampling.

| Flag | Default | Notes |
|------|---------|-------|
| `--library` | `wiki_category_library.json` | Path to library JSON. |
| `--refresh` | off | Fetch `categoryinfo` and rewrite the library. Without this flag, prints help only. |
| `--no-subcategories` | off | Skip listing direct subcategories. |
| `--max-subcategories` | all | Cap subcategories fetched per entry. |
| `--api` | env or enwiki | API endpoint. |
| `--user_agent` | project default | `User-Agent` header. |
| `--api_key` | env `WIKIMEDIA_API_KEY` | Optional API key. |

```powershell
.\.venv313\Scripts\python.exe .\wiki_category_library.py --refresh
```

---

## `scripts/expand_wiki_tree_random.py`

Append more random docs per root to an existing `wiki_tree_random.jsonl` using the category library.

| Flag | Default | Notes |
|------|---------|-------|
| `--jsonl` | `data/wiki_tree_random.jsonl` | Existing JSONL to append to. |
| `--library` | `wiki_category_library.json` | Category metadata source. |
| `--add-per-root` | `90` | New articles per root category. |
| `--seed` | `20260423` | RNG seed. |
| `--delay-sec` | `0.10` | Pause between API calls. |
| `--title-pool-cache` | `data/wiki_category_title_pool_cache.json` | Cache for category→title pools. |
| `--category-index` | `data/wiki_snapshot_category_index.json` | Local snapshot index path. |
| `--use-category-index` | off | Prefer local index over API list calls. |
| `--index-only` | off | Require local index only (no API list fallbacks). |
| `--no-cache-read` | off | Ignore existing title pool cache. |
| `--no-cache-write` | off | Do not update cache after run. |
| `--max-categories-per-root` | `24` | Cap category list API calls per root. |
| `--oversample-ratio` | `1.5` | Candidate title multiplier before local sampling. |
| `--pool-multiplier` | `4.0` | Per-category member pool size factor. |
| `--pool-min` | `30` | Minimum member pool per category. |
| `--pool-max` | `500` | Maximum member pool per category. |
| `--api` | env or enwiki | API endpoint. |
| `--user-agent` | project default | `User-Agent` header. |
| `--api-key` | env `WIKIMEDIA_API_KEY` | Optional API key. |

---

## `cka_word_embeddings.py`

Standalone pairwise linear CKA on two aligned `.npy` embedding files (outside the pipeline layout).

| Flag | Default | Notes |
|------|---------|-------|
| `--a` | *(required)* | First `.npy` (e.g. `word_embeddings_merged_agnostic.npy`). |
| `--b` | *(required)* | Second `.npy`. |
| `--sample_size` | *(none)* | Random row subset for quick runs. |
| `--seed` | `0` | RNG seed for subsampling / bootstrap. |
| `--per_doc` | off | Also compute per-`doc_id` CKA. |
| `--min_doc_rows` | `32` | Skip docs with fewer rows for per-doc CKA. |
| `--bootstrap` | `0` | Bootstrap resample count (`0` = off). |
| `--json_out` | *(none)* | Write JSON report to this path. |
| `--chunk_rows` | `8192` | Chunked CKA block size (`0` = dense, higher RAM). |
| `--cka_filter` | *(none)* | Repeatable `KEY=VALUE` row filters (AND). |
| `--cka_doc_ids` | *(none)* | Comma-separated doc IDs (intersected with filters). |
| `--threads` | `4` | BLAS/OpenMP thread cap. |
| `--dtype` | `float32` | `float32` or `float64` when loading embeddings. |

---

## `scripts/render_document_cka_report.py`

Render `document_cka_by_category.json` as a markdown table.

| Flag | Default | Notes |
|------|---------|-------|
| `--input` | *(none)* | Path to `document_cka_by_category.json`. |
| `--out_dir` | *(none)* | Run directory; used when `--input` omitted (`<out_dir>/cka/document_cka_by_category.json`). |
| `--output` | alongside input `.md` | Output markdown path. |

```powershell
.\.venv313\Scripts\python.exe .\scripts\render_document_cka_report.py --out_dir ".\outputs\my_run5"
```

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `GEMMA3_MODEL_PATH` | `run_wiki_tree_cka_pipeline.py` | Default Gemma checkpoint directory. |
| `QWEN35_MODEL_PATH` | `run_wiki_tree_cka_pipeline.py` | Default Qwen checkpoint directory. |
| `WIKIMEDIA_API_URL` | wiki scripts | MediaWiki API endpoint override. |
| `WIKIMEDIA_API_KEY` | wiki scripts | Optional API key for authenticated requests. |

---

## Config artifacts (`<out_dir>/`)

| File | Contents |
|------|----------|
| `pipeline_config.json` | Frozen run settings from `init`. |
| `pipeline_state.json` | Completed stage names. |
| `run_info.json` | Timings, artifact paths, CKA summary. Includes `cli_command` (Windows **cmd.exe** one-liner) and `cli_argv` from the initiating `init` invocation. |
| `corpus_excerpts_used.jsonl` | Truncated excerpts actually embedded. |
| `corpus_source.jsonl` | Copy of input corpus. |
