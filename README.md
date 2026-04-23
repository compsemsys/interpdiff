# Independent Study

This repository is now centered on a staged local pipeline for:

1. Building excerpt corpora from categorized JSONL documents.
2. Generating model responses from excerpt/title prompts.
3. Embedding excerpts and responses.
4. Aggregating embeddings at word and/or document level.
5. Running linear CKA comparisons, including document CKA by category.

## Current Entry Points

- `run_categorized_corpus.py` - main pipeline CLI (`init`, `generate`, `embed_excerpts`, `embed_responses`, `cka`, or `all`).
- `pipeline_embed_explain_cka.py` - alias entry point to `run_categorized_corpus.py`.
- `scripts/run_wiki_tree_cka_pipeline.py` - convenience wrapper that builds a corpus from wiki-tree output and runs the full pipeline.
- `wiki_fetch.py` - category-based Wikipedia fetcher for corpus construction.
- `wiki_tree_random_articles.py` - weighted random wiki-tree sampler using `wiki_category_library.json`.
- `wiki_category_library.py` - refreshes and maintains local category metadata used by sampling.

## Environment Setup (Windows / PowerShell)

This project uses `.\.venv313`:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts with:

```powershell
.\.venv313\Scripts\python.exe .\run_categorized_corpus.py --help
```

## Pipeline Overview

The pipeline expects a JSONL corpus with at least:

- `doc_id` (int)
- `category` (str)
- `title` (str)
- `text` (str)

Main stages:

- `init` - writes excerpt corpus + pipeline config/state.
- `generate` - generates responses for each model.
- `embed_excerpts` - token embeddings + word/document aggregation for excerpts.
- `embed_responses` - same for generated responses.
- `cka` - pairwise CKA outputs and document-by-category matrix (when document aggregation exists).

See `PIPELINE_GUIDE.md` for full CLI details, resume/overwrite behavior, and artifact layout.

## Quick Start

```powershell
$PY = ".\.venv313\Scripts\python.exe"
$RC = ".\run_categorized_corpus.py"

& $PY $RC `
  --corpus ".\data\wiki_tree_corpus_for_cka.jsonl" `
  --out_dir ".\outputs\my_run2" `
  --models "F:\path\to\gemma-3-1b-it" "F:\path\to\Qwen3.5-0.8B" `
  --aggregation_level "document" `
  --words 200 `
  --max_new_tokens 128 `
  --batch_size 4 `
  --chunk_size 768 `
  --stage all
```

## `my_run` Findings (Document CKA by Category)

Source report: `outputs/my_run/cka/document_cka_by_category.md`

Run context:

- Models: `Qwen3.5-0.8B`, `gemma-3-1b-it`
- Segments: `excerpt`, `response`
- Categories: `Category:Culture`, `Category:Science`
- Comparisons: 12 total (10 aligned docs per comparison)

### Cross-model, same-segment CKA

- Culture: excerpt `0.920879`, response `0.866737`
- Science: excerpt `0.928342`, response `0.875575`

Observed pattern: cross-model agreement is high for excerpts and slightly lower for responses.

### Within-model, excerpt-vs-response CKA

- Qwen: Culture `0.950087`, Science `0.824079`
- Gemma: Culture `0.752757`, Science `0.759597`

Observed pattern: Qwen preserves stronger excerpt/response alignment than Gemma in this run.

### Cross-model, cross-segment CKA

- Culture: Qwen excerpt vs Gemma response `0.825404`; Qwen response vs Gemma excerpt `0.852045`
- Science: Qwen excerpt vs Gemma response `0.875776`; Qwen response vs Gemma excerpt `0.668196`

Observed pattern: category and segment pairing matter; the weakest pair in this report is Science with Qwen response vs Gemma excerpt (`0.668196`).

## Notes

- CKA values here are from a small sample (10 aligned docs per category comparison), so treat them as directional rather than final.
- For larger conclusions, rerun with more categories/docs and compare stability across seeds/checkpoints.
