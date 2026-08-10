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

## Cross-embed summary

| Comparison | A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- | --- |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.878439 | 0.898536 | 0.857830 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.872853 | 0.887644 | 0.855975 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.867581 | 0.888678 | 0.834955 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.858011 | 0.875234 | 0.843815 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.470437 | 0.473636 | 0.459733 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.587980 | 0.582151 | 0.598899 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.769986 | 0.713306 | 0.818358 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.826770 | 0.777579 | 0.867621 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.661025 | 0.692877 | 0.671418 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.697558 | 0.729985 | 0.699542 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.827479 | 0.710589 | 0.901424 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.844073 | 0.757155 | 0.912781 |
_Cross-embed pairs: same generated text under two embedders, or same embedder on excerpt/own text vs foreign-model text. Labels use `emb=` (who embedded) and `text=` (whose string)._

## Cross-embed (pooled)

### Same text, different embedders

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.878439 | 200 |
| emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.872853 | 200 |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.867581 | 200 |
| emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.858011 | 200 |

### Same embedder: excerpt vs foreign text

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.470437 | 200 |
| emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.587980 | 200 |
| emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.769986 | 200 |
| emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.826770 | 200 |

### Same embedder: own vs foreign text

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.661025 | 200 |
| emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.697558 | 200 |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.827479 | 200 |
| emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.844073 | 200 |

## Cross-embed (per-category)

### Same text, different embedders

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.857830 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.898536 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.855975 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.887644 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.834955 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.888678 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.843815 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.875234 | 100 |

### Same embedder: excerpt vs foreign text

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.459733 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.473636 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.598899 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.582151 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.818358 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.713306 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.867621 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.777579 | 100 |

### Same embedder: own vs foreign text

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.671418 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.692877 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.699542 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.729985 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.901424 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.710589 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.912781 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.757155 | 100 |

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

