# Document CKA (by category and pooled)

## Summary

| A | B | Pooled CKA (200) | Science CKA (100) | Culture CKA (100) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603337 | 0.591571 | 0.628776 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.456869 | 0.460326 | 0.446968 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825885 | 0.841013 | 0.814855 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.633531 | 0.668976 | 0.641238 |
_A / B: model and segment for each side (within-model different segments, then cross-model same segment). Segments: **excerpt**, **response**. Science column: `Category:Science`; Culture column: `Category:Culture` (matched from category strings in the JSON)._

- Source: `F:\code\Independent Study\outputs\my_run5\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response
- Categories: Category:Culture, Category:Science
- Per-category pair comparisons: 12
- Pooled pair comparisons (all doc_id aligned rows): 6

## Run parameters

- **corpus:** `F:\code\Independent Study\data\wiki_tree_corpus_for_cka.jsonl`
- **words:** `200`
- **num_docs:** `200`
- **models:** F:\quantas\models\google\gemma-3-1b-it, F:\quantas\models\Qwen\Qwen3.5-0.8B
- **aggregation_level:** `document`
- **max_new_tokens:** `1500`
- **max_generated_words:** `200`
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
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.633531 | 200 |

### Within model, different segments

| A | B | CKA | Rows |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603337 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.456869 | 200 |

### Full matrix (pooled)

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.603337 | 0.825885 | 0.525305 |
| Qwen3.5-0.8B (response) | 0.603337 | 1.000000 | 0.433491 | 0.633531 |
| gemma-3-1b-it (excerpt) | 0.825885 | 0.433491 | 1.000000 | 0.456869 |
| gemma-3-1b-it (response) | 0.525305 | 0.633531 | 0.456869 | 1.000000 |

## Per-category: highlighted comparisons

### Cross model, same segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.814855 | 100 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.641238 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.841013 | 100 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.668976 | 100 |

### Within model, different segments

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.628776 | 100 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.446968 | 100 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.591571 | 100 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.460326 | 100 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.628776 | 0.814855 | 0.519374 |
| Qwen3.5-0.8B (response) | 0.628776 | 1.000000 | 0.422424 | 0.641238 |
| gemma-3-1b-it (excerpt) | 0.814855 | 0.422424 | 1.000000 | 0.446968 |
| gemma-3-1b-it (response) | 0.519374 | 0.641238 | 0.446968 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.591571 | 0.841013 | 0.547332 |
| Qwen3.5-0.8B (response) | 0.591571 | 1.000000 | 0.453624 | 0.668976 |
| gemma-3-1b-it (excerpt) | 0.841013 | 0.453624 | 1.000000 | 0.460326 |
| gemma-3-1b-it (response) | 0.547332 | 0.668976 | 0.460326 | 1.000000 |

