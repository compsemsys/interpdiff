# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.593986 | 0.591522 | 0.623651 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.876160 | 0.851400 | 0.899293 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.620666 | 0.626902 | 0.645393 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.469692 | 0.453494 | 0.501867 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.734558 | 0.667312 | 0.789969 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.574064 | 0.525198 | 0.608846 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824920 | 0.840596 | 0.814234 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 0.651034 | 0.624732 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.766431 | 0.688544 | 0.804738 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**, **summary**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run13\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response, summary
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 30
- Pooled pair comparisons (all doc_id aligned rows): 15

## Run parameters

- **cli_command:** `.\run_categorized_corpus.py --corpus .\data\wiki_tree_corpus_for_cka.jsonl --out_dir .\outputs\my_run13 --models F:\quantas\models\google\gemma-3-1b-it F:\quantas\models\Qwen\Qwen3.5-0.8B --aggregation_level document --words 1000 --max_new_tokens 3000 --max_generated_words 200 --word_count_prompt --summarize --batch_size 4 --chunk_size 768 --stage all --summarize_words 200`
- **corpus:** `F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl`
- **words:** `1000`
- **num_docs:** `200`
- **models:** F:\quantas\models\google\gemma-3-1b-it, F:\quantas\models\Qwen\Qwen3.5-0.8B
- **aggregation_level:** `document`
- **max_new_tokens:** `3000`
- **max_generated_words:** `200`
- **match_abstract_length:** `False`
- **word_count_prompt:** `True`
- **generation_tasks:**
  - `response`: max_generated_words=200
    - instruction: `In {word_count} words, explain the following: {title}`
  - `summary`: max_generated_words=200
    - instruction: `Summarize the following in {word_count} words: {excerpt}`
- **batch_size:** `4`
- **chunk_size:** `768`
- **cka_chunk_rows:** `4096`
- **device:** `cuda`
- **local_only:** `True`
- **skip_generate:** `False`
- **skip_cka:** `False`


## Pooled (all documents)

One CKA per model/segment **pair** using the full set of document embeddings aligned by `doc_id` (not restricted to a single category).

### Cross model, same segment

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824920 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.766431 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.593986 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.876160 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.620666 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.469692 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.734558 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.574064 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.593986 | 0.876160 | 0.824920 | 0.537987 | 0.787879 |
| Qwen3.5-0.8B (response) | 0.593986 | 1.000000 | 0.620666 | 0.428372 | 0.619736 | 0.533847 |
| Qwen3.5-0.8B (summary) | 0.876160 | 0.620666 | 1.000000 | 0.658487 | 0.554078 | 0.766431 |
| gemma-3-1b-it (excerpt) | 0.824920 | 0.428372 | 0.658487 | 1.000000 | 0.469692 | 0.734558 |
| gemma-3-1b-it (response) | 0.537987 | 0.619736 | 0.554078 | 0.469692 | 1.000000 | 0.574064 |
| gemma-3-1b-it (summary) | 0.787879 | 0.533847 | 0.766431 | 0.734558 | 0.574064 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814234 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624732 | 100 |
| Category:Culture | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.804738 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.840596 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.651034 | 100 |
| Category:Science | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.688544 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.623651 | 100 |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.899293 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.645393 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.501867 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.789969 | 100 |
| Category:Culture | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.608846 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.591522 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.851400 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.626902 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.453494 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.667312 | 100 |
| Category:Science | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.525198 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.623651 | 0.899293 | 0.814234 | 0.550540 | 0.811364 |
| Qwen3.5-0.8B (response) | 0.623651 | 1.000000 | 0.645393 | 0.423512 | 0.624732 | 0.576660 |
| Qwen3.5-0.8B (summary) | 0.899293 | 0.645393 | 1.000000 | 0.654029 | 0.551707 | 0.804738 |
| gemma-3-1b-it (excerpt) | 0.814234 | 0.423512 | 0.654029 | 1.000000 | 0.501867 | 0.789969 |
| gemma-3-1b-it (response) | 0.550540 | 0.624732 | 0.551707 | 0.501867 | 1.000000 | 0.608846 |
| gemma-3-1b-it (summary) | 0.811364 | 0.576660 | 0.804738 | 0.789969 | 0.608846 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.591522 | 0.851400 | 0.840596 | 0.547018 | 0.740206 |
| Qwen3.5-0.8B (response) | 0.591522 | 1.000000 | 0.626902 | 0.447520 | 0.651034 | 0.483484 |
| Qwen3.5-0.8B (summary) | 0.851400 | 0.626902 | 1.000000 | 0.658289 | 0.565559 | 0.688544 |
| gemma-3-1b-it (excerpt) | 0.840596 | 0.447520 | 0.658289 | 1.000000 | 0.453494 | 0.667312 |
| gemma-3-1b-it (response) | 0.547018 | 0.651034 | 0.565559 | 0.453494 | 1.000000 | 0.525198 |
| gemma-3-1b-it (summary) | 0.740206 | 0.483484 | 0.688544 | 0.667312 | 0.525198 | 1.000000 |

