# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.596347 | 0.616331 | 0.603927 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946685 | 0.937181 | 0.958357 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.616859 | 0.633807 | 0.622528 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.488759 | 0.501158 | 0.499186 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.795062 | 0.743365 | 0.837280 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.587390 | 0.581135 | 0.610672 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824920 | 0.840596 | 0.814234 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.659424 | 0.683672 | 0.675811 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.802326 | 0.805442 | 0.806923 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**, **summary**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run12\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response, summary
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 30
- Pooled pair comparisons (all doc_id aligned rows): 15

## Run parameters

- **cli_command:** `.\run_categorized_corpus.py --corpus .\data\wiki_tree_corpus_for_cka.jsonl --out_dir .\outputs\my_run12 --models F:\quantas\models\google\gemma-3-1b-it F:\quantas\models\Qwen\Qwen3.5-0.8B --aggregation_level document --words 1000 --max_new_tokens 3000 --match_abstract_length --word_count_prompt --summarize --batch_size 4 --chunk_size 768 --stage all`
- **corpus:** `F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl`
- **words:** `1000`
- **num_docs:** `200`
- **models:** F:\quantas\models\google\gemma-3-1b-it, F:\quantas\models\Qwen\Qwen3.5-0.8B
- **aggregation_level:** `document`
- **max_new_tokens:** `3000`
- **max_generated_words:** `None`
- **match_abstract_length:** `True`
- **word_count_prompt:** `True`
- **generation_tasks:**
  - `response`: max_generated_words=None
    - instruction: `In {word_count} words, explain the following: {title}`
  - `summary`: max_generated_words=None
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
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.659424 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.802326 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.596347 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946685 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.616859 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.488759 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.795062 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.587390 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.596347 | 0.946685 | 0.824920 | 0.556890 | 0.798027 |
| Qwen3.5-0.8B (response) | 0.596347 | 1.000000 | 0.616859 | 0.432939 | 0.659424 | 0.532946 |
| Qwen3.5-0.8B (summary) | 0.946685 | 0.616859 | 1.000000 | 0.769000 | 0.565853 | 0.802326 |
| gemma-3-1b-it (excerpt) | 0.824920 | 0.432939 | 0.769000 | 1.000000 | 0.488759 | 0.795062 |
| gemma-3-1b-it (response) | 0.556890 | 0.659424 | 0.565853 | 0.488759 | 1.000000 | 0.587390 |
| gemma-3-1b-it (summary) | 0.798027 | 0.532946 | 0.802326 | 0.795062 | 0.587390 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814234 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.675811 | 100 |
| Category:Culture | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.806923 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.840596 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.683672 | 100 |
| Category:Science | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.805442 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603927 | 100 |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.958357 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.622528 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.499186 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.837280 | 100 |
| Category:Culture | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.610672 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.616331 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.937181 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.633807 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.501158 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.743365 | 100 |
| Category:Science | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.581135 | 100 |

## Cross-embed summary

| Comparison | A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- | --- |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.856098 | 0.869957 | 0.852011 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.815218 | 0.823271 | 0.816622 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.860678 | 0.881180 | 0.844348 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.827754 | 0.830696 | 0.834573 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.486322 | 0.511263 | 0.472532 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.619317 | 0.632580 | 0.621340 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.899108 | 0.847747 | 0.936798 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.870389 | 0.855674 | 0.885782 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.731636 | 0.743594 | 0.744217 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.748748 | 0.771899 | 0.758788 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.837018 | 0.791175 | 0.879711 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.882068 | 0.871634 | 0.895530 |
_Cross-embed pairs: same generated text under two embedders, or same embedder on excerpt/own text vs foreign-model text. Labels use `emb=` (who embedded) and `text=` (whose string)._

## Cross-embed (pooled)

### Same text, different embedders

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.856098 | 200 |
| emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.815218 | 200 |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.860678 | 200 |
| emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.827754 | 200 |

### Same embedder: excerpt vs foreign text

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.486322 | 200 |
| emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.619317 | 200 |
| emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.899108 | 200 |
| emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.870389 | 200 |

### Same embedder: own vs foreign text

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.731636 | 200 |
| emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.748748 | 200 |
| emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.837018 | 200 |
| emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.882068 | 200 |

## Cross-embed (per-category)

### Same text, different embedders

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.852011 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.869957 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.816622 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.823271 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.844348 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.881180 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.834573 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.830696 | 100 |

### Same embedder: excerpt vs foreign text

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.472532 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.511263 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.621340 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.632580 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.936798 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.847747 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.885782 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.855674 | 100 |

### Same embedder: own vs foreign text

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.744217 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.743594 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.758788 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.771899 | 100 |
| Category:Culture | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.879711 | 100 |
| Category:Science | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.791175 | 100 |
| Category:Culture | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.895530 | 100 |
| Category:Science | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.871634 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.603927 | 0.958357 | 0.814234 | 0.566659 | 0.804650 |
| Qwen3.5-0.8B (response) | 0.603927 | 1.000000 | 0.622528 | 0.420056 | 0.675811 | 0.532250 |
| Qwen3.5-0.8B (summary) | 0.958357 | 0.622528 | 1.000000 | 0.772015 | 0.584824 | 0.806923 |
| gemma-3-1b-it (excerpt) | 0.814234 | 0.420056 | 0.772015 | 1.000000 | 0.499186 | 0.837280 |
| gemma-3-1b-it (response) | 0.566659 | 0.675811 | 0.584824 | 0.499186 | 1.000000 | 0.610672 |
| gemma-3-1b-it (summary) | 0.804650 | 0.532250 | 0.806923 | 0.837280 | 0.610672 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.616331 | 0.937181 | 0.840596 | 0.581236 | 0.794589 |
| Qwen3.5-0.8B (response) | 0.616331 | 1.000000 | 0.633807 | 0.474869 | 0.683672 | 0.564543 |
| Qwen3.5-0.8B (summary) | 0.937181 | 0.633807 | 1.000000 | 0.776234 | 0.586520 | 0.805442 |
| gemma-3-1b-it (excerpt) | 0.840596 | 0.474869 | 0.776234 | 1.000000 | 0.501158 | 0.743365 |
| gemma-3-1b-it (response) | 0.581236 | 0.683672 | 0.586520 | 0.501158 | 1.000000 | 0.581135 |
| gemma-3-1b-it (summary) | 0.794589 | 0.564543 | 0.805442 | 0.743365 | 0.581135 | 1.000000 |

