# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run12\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 9
- Elapsed: 19.28s

| A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.596346 | 0.598685 | 0.004926 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946685 | 0.946966 | 0.001145 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.616860 | 0.621252 | 0.004967 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.488762 | 0.491375 | 0.008129 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.795063 | 0.794482 | 0.005429 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.587391 | 0.590194 | 0.007408 | 200 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.825112 | 0.003220 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.659425 | 0.661914 | 0.004356 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.802325 | 0.803808 | 0.003563 | 200 |
