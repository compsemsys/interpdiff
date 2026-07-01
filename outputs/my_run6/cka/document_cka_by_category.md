# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 0.590778 | 0.629827 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 0.453394 | 0.502379 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 0.651034 | 0.624732 |
_A / B: model and **excerpt** vs **response** for each side of the comparison. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run6\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 12
- Pooled pair comparisons (all doc_id aligned rows): 6


## Pooled (all documents)

One CKA per model/segment **pair** using the full set of document embeddings aligned by `doc_id` (not restricted to a single category).

### Cross model, same segment

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619736 | 200 |

### Within model, excerpt vs response

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471680 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.595318 | 0.825885 | 0.537848 |
| Qwen3.5-0.8B (response) | 0.595318 | 1.000000 | 0.431099 | 0.619736 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.431099 | 1.000000 | 0.471680 |
| gemma-3-1b-it (response) | 0.537848 | 0.619736 | 0.471680 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624732 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.651034 | 100 |

### Within model, excerpt vs response

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.629827 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.502379 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.590778 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.453394 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.629827 | 0.814855 | 0.550358 |
| Qwen3.5-0.8B (response) | 0.629827 | 1.000000 | 0.430369 | 0.624732 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.430369 | 1.000000 | 0.502379 |
| gemma-3-1b-it (response) | 0.550358 | 0.624732 | 0.502379 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.590778 | 0.841013 | 0.545733 |
| Qwen3.5-0.8B (response) | 0.590778 | 1.000000 | 0.446967 | 0.651034 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.446967 | 1.000000 | 0.453394 |
| gemma-3-1b-it (response) | 0.545733 | 0.651034 | 0.453394 | 1.000000 |

