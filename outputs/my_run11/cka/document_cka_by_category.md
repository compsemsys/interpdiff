# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603685 | 0.613576 | 0.627741 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946356 | 0.937102 | 0.958054 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.621047 | 0.631697 | 0.638227 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.491971 | 0.495308 | 0.504158 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.794580 | 0.745708 | 0.833205 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.584264 | 0.577990 | 0.603372 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.643304 | 0.666024 | 0.665416 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.803468 | 0.807126 | 0.806770 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**, **summary**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run11\cka\document_cka_by_category.json`
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
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.643304 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.803468 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603685 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946356 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.621047 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.491971 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.794580 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.584264 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.603685 | 0.946356 | 0.825885 | 0.555111 | 0.796248 |
| Qwen3.5-0.8B (response) | 0.603685 | 1.000000 | 0.621047 | 0.443632 | 0.643304 | 0.537215 |
| Qwen3.5-0.8B (summary) | 0.946356 | 0.621047 | 1.000000 | 0.770615 | 0.565092 | 0.803468 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.443632 | 0.770615 | 1.000000 | 0.491971 | 0.794580 |
| gemma-3-1b-it (response) | 0.555111 | 0.643304 | 0.565092 | 0.491971 | 1.000000 | 0.584264 |
| gemma-3-1b-it (summary) | 0.796248 | 0.537215 | 0.803468 | 0.794580 | 0.584264 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.665416 | 100 |
| Category:Culture | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.806770 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.666024 | 100 |
| Category:Science | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.807126 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.627741 | 100 |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.958054 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.638227 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.504158 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.833205 | 100 |
| Category:Culture | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.603372 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.613576 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.937102 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.631697 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.495308 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.745708 | 100 |
| Category:Science | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.577990 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.627741 | 0.958054 | 0.814855 | 0.571533 | 0.798573 |
| Qwen3.5-0.8B (response) | 0.627741 | 1.000000 | 0.638227 | 0.438795 | 0.665416 | 0.541974 |
| Qwen3.5-0.8B (summary) | 0.958054 | 0.638227 | 1.000000 | 0.773948 | 0.591457 | 0.806770 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.438795 | 0.773948 | 1.000000 | 0.504158 | 0.833205 |
| gemma-3-1b-it (response) | 0.571533 | 0.665416 | 0.591457 | 0.504158 | 1.000000 | 0.603372 |
| gemma-3-1b-it (summary) | 0.798573 | 0.541974 | 0.806770 | 0.833205 | 0.603372 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | gemma-3-1b-it (summary) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.613576 | 0.937102 | 0.841013 | 0.570157 | 0.795924 |
| Qwen3.5-0.8B (response) | 0.613576 | 1.000000 | 0.631697 | 0.476598 | 0.666024 | 0.561179 |
| Qwen3.5-0.8B (summary) | 0.937102 | 0.631697 | 1.000000 | 0.776459 | 0.578463 | 0.807126 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.476598 | 0.776459 | 1.000000 | 0.495308 | 0.745708 |
| gemma-3-1b-it (response) | 0.570157 | 0.666024 | 0.578463 | 0.495308 | 1.000000 | 0.577990 |
| gemma-3-1b-it (summary) | 0.795924 | 0.561179 | 0.807126 | 0.745708 | 0.577990 | 1.000000 |

