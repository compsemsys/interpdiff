# Document CKA By Category

- Source: `F:\code\Independent Study\outputs\my_run\cka\document_cka_by_category.json`
- Models: Qwen3.5-0.8B, gemma-3-1b-it
- Segments: excerpt, response
- Categories: Category:Culture, Category:Science
- Comparisons: 12

## Cross Model Same Segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.920879 | 10 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.866737 | 10 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.928342 | 10 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.875575 | 10 |

## Within Model Excerpt Vs Response

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.950087 | 10 |
| Category:Culture | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.752757 | 10 |
| Category:Science | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.824079 | 10 |
| Category:Science | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.759597 | 10 |

## Cross Model Cross Segment

| Category | A | B | CKA | Rows |
| --- | --- | --- | --- | --- |
| Category:Culture | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (response) | 0.825404 | 10 |
| Category:Culture | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | 0.852045 | 10 |
| Category:Science | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (response) | 0.875776 | 10 |
| Category:Science | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | 0.668196 | 10 |

## Per-Category Matrices

### Category:Culture

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.950087 | 0.920879 | 0.825404 |
| Qwen3.5-0.8B (response) | 0.950087 | 1.000000 | 0.852045 | 0.866737 |
| gemma-3-1b-it (excerpt) | 0.920879 | 0.852045 | 1.000000 | 0.752757 |
| gemma-3-1b-it (response) | 0.825404 | 0.866737 | 0.752757 | 1.000000 |

### Category:Science

| A \ B | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | 1.000000 | 0.824079 | 0.928342 | 0.875776 |
| Qwen3.5-0.8B (response) | 0.824079 | 1.000000 | 0.668196 | 0.875575 |
| gemma-3-1b-it (excerpt) | 0.928342 | 0.668196 | 1.000000 | 0.759597 |
| gemma-3-1b-it (response) | 0.875776 | 0.875575 | 0.759597 | 1.000000 |

