# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 0.590778 | 0.629827 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.877439 | 0.851973 | 0.901715 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.619842 | 0.625624 | 0.647979 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 0.453394 | 0.502379 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.738360 | 0.668553 | 0.790936 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.570563 | 0.522099 | 0.597581 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 0.651034 | 0.624732 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.765228 | 0.691200 | 0.799948 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**, **summary**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run8\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response, summary
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 30
- Pooled pair comparisons (all doc_id aligned rows): 15

## Run parameters

- **cli_command:** `.\run_categorized_corpus.py --corpus .\data\wiki_tree_corpus_for_cka.jsonl --out_dir .\outputs\my_run8 --models F:\quantas\models\google\gemma-3-1b-it F:\quantas\models\Qwen\Qwen3.5-0.8B --aggregation_level document --words 200 --max_new_tokens 1500 --max_generated_words 200 --word_count_prompt --summarize --batch_size 4 --chunk_size 768 --stage all --summarize_words 200`
- **corpus:** `F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl`
- **words:** `200`
- **num_docs:** `200`
- **models:** F:\quantas\models\google\gemma-3-1b-it, F:\quantas\models\Qwen\Qwen3.5-0.8B
- **aggregation_level:** `document`
- **max_new_tokens:** `1500`
- **max_generated_words:** `200`
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
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.765228 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.877439 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.619842 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.738360 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.570563 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.595318 | 0.877439 | 0.825885 | 0.537848 | 0.787474 |
| Qwen3.5-0.8B (response) | 0.595318 | 1.000000 | 0.619842 | 0.431099 | 0.619736 | 0.531162 |
| Qwen3.5-0.8B (summary) | 0.877439 | 0.619842 | 1.000000 | 0.663728 | 0.555651 | 0.765228 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.431099 | 0.663728 | 1.000000 | 0.471680 | 0.738360 |
| gemma-3-1b-it (response) | 0.537848 | 0.619736 | 0.555651 | 0.471680 | 1.000000 | 0.570563 |
| gemma-3-1b-it (summary) | 0.787474 | 0.531162 | 0.765228 | 0.738360 | 0.570563 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624732 | 100 |
| Category:Culture | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.799948 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.651034 | 100 |
| Category:Science | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.691200 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.629827 | 100 |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.901715 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.647979 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.502379 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.790936 | 100 |
| Category:Culture | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.597581 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.590778 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.851973 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.625624 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.453394 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.668553 | 100 |
| Category:Science | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.522099 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.629827 | 0.901715 | 0.814855 | 0.550358 | 0.804873 |
| Qwen3.5-0.8B (response) | 0.629827 | 1.000000 | 0.647979 | 0.430369 | 0.624732 | 0.567421 |
| Qwen3.5-0.8B (summary) | 0.901715 | 0.647979 | 1.000000 | 0.663540 | 0.556461 | 0.799948 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.430369 | 0.663540 | 1.000000 | 0.502379 | 0.790936 |
| gemma-3-1b-it (response) | 0.550358 | 0.624732 | 0.556461 | 0.502379 | 1.000000 | 0.597581 |
| gemma-3-1b-it (summary) | 0.804873 | 0.567421 | 0.799948 | 0.790936 | 0.597581 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.590778 | 0.851973 | 0.841013 | 0.545733 | 0.741338 |
| Qwen3.5-0.8B (response) | 0.590778 | 1.000000 | 0.625624 | 0.446967 | 0.651034 | 0.482575 |
| Qwen3.5-0.8B (summary) | 0.851973 | 0.625624 | 1.000000 | 0.659898 | 0.566883 | 0.691200 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.446967 | 0.659898 | 1.000000 | 0.453394 | 0.668553 |
| gemma-3-1b-it (response) | 0.545733 | 0.651034 | 0.566883 | 0.453394 | 1.000000 | 0.522099 |
| gemma-3-1b-it (summary) | 0.741338 | 0.482575 | 0.691200 | 0.668553 | 0.522099 | 1.000000 |

