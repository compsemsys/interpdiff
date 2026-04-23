# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.628038 | 0.624467 | 0.656407 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.494630 | 0.474753 | 0.527266 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624348 | 0.662507 | 0.647680 |

- Source: `F:\code\Independent Study\outputs\my_run3\cka\document_cka_by_category.json`
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
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.624348 | 200 |

### Within model, excerpt vs response

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.628038 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.494630 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.628038 | 0.825885 | 0.525811 |
| Qwen3.5-0.8B (response) | 0.628038 | 1.000000 | 0.470765 | 0.624348 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.470765 | 1.000000 | 0.494630 |
| gemma-3-1b-it (response) | 0.525811 | 0.624348 | 0.494630 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.647680 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.662507 | 100 |

### Within model, excerpt vs response

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.656407 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.527266 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.624467 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.474753 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.656407 | 0.814855 | 0.544833 |
| Qwen3.5-0.8B (response) | 0.656407 | 1.000000 | 0.470223 | 0.647680 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.470223 | 1.000000 | 0.527266 |
| gemma-3-1b-it (response) | 0.544833 | 0.647680 | 0.527266 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.624467 | 0.841013 | 0.550563 |
| Qwen3.5-0.8B (response) | 0.624467 | 1.000000 | 0.484162 | 0.662507 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.484162 | 1.000000 | 0.474753 |
| gemma-3-1b-it (response) | 0.550563 | 0.662507 | 0.474753 | 1.000000 |

