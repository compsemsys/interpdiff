# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 0.590778 | 0.629827 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.918091 | 0.897290 | 0.940956 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.612772 | 0.615505 | 0.644064 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 0.453394 | 0.502379 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.764360 | 0.715676 | 0.809412 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.553238 | 0.543544 | 0.586638 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 0.651034 | 0.624732 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.784325 | 0.773501 | 0.801154 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**, **summary**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run9\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response, summary
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 30
- Pooled pair comparisons (all doc_id aligned rows): 15


## Pooled (all documents)

One CKA per model/segment **pair** using the full set of document embeddings aligned by `doc_id` (not restricted to a single category).

### Cross model, same segment

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.784325 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.918091 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.612772 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.764360 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.553238 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.595318 | 0.918091 | 0.825885 | 0.537848 | 0.771768 |
| Qwen3.5-0.8B (response) | 0.595318 | 1.000000 | 0.612772 | 0.431099 | 0.619736 | 0.511397 |
| Qwen3.5-0.8B (summary) | 0.918091 | 0.612772 | 1.000000 | 0.746070 | 0.573018 | 0.784325 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.431099 | 0.746070 | 1.000000 | 0.471680 | 0.764360 |
| gemma-3-1b-it (response) | 0.537848 | 0.619736 | 0.573018 | 0.471680 | 1.000000 | 0.553238 |
| gemma-3-1b-it (summary) | 0.771768 | 0.511397 | 0.784325 | 0.764360 | 0.553238 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624732 | 100 |
| Category:Culture | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.801154 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.651034 | 100 |
| Category:Science | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.773501 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.629827 | 100 |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.940956 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.644064 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.502379 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.809412 | 100 |
| Category:Culture | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.586638 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.590778 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.897290 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.615505 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.453394 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.715676 | 100 |
| Category:Science | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.543544 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.629827 | 0.940956 | 0.814855 | 0.550358 | 0.791508 |
| Qwen3.5-0.8B (response) | 0.629827 | 1.000000 | 0.644064 | 0.430369 | 0.624732 | 0.545297 |
| Qwen3.5-0.8B (summary) | 0.940956 | 0.644064 | 1.000000 | 0.754447 | 0.584346 | 0.801154 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.430369 | 0.754447 | 1.000000 | 0.502379 | 0.809412 |
| gemma-3-1b-it (response) | 0.550358 | 0.624732 | 0.584346 | 0.502379 | 1.000000 | 0.586638 |
| gemma-3-1b-it (summary) | 0.791508 | 0.545297 | 0.801154 | 0.809412 | 0.586638 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.590778 | 0.897290 | 0.841013 | 0.545733 | 0.763052 |
| Qwen3.5-0.8B (response) | 0.590778 | 1.000000 | 0.615505 | 0.446967 | 0.651034 | 0.520499 |
| Qwen3.5-0.8B (summary) | 0.897290 | 0.615505 | 1.000000 | 0.737076 | 0.593978 | 0.773501 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.446967 | 0.737076 | 1.000000 | 0.453394 | 0.715676 |
| gemma-3-1b-it (response) | 0.545733 | 0.651034 | 0.593978 | 0.453394 | 1.000000 | 0.543544 |
| gemma-3-1b-it (summary) | 0.763052 | 0.520499 | 0.773501 | 0.715676 | 0.543544 | 1.000000 |

