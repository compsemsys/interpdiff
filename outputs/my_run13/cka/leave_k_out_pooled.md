# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run13\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 9
- Elapsed: 15.71s

| A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.593985 | 0.596628 | 0.005257 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.876160 | 0.876329 | 0.003048 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.620666 | 0.624559 | 0.005453 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.469695 | 0.471944 | 0.007310 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.734561 | 0.734407 | 0.007943 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.574064 | 0.576978 | 0.009229 | 200 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.825112 | 0.003220 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619734 | 0.622443 | 0.005398 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.766430 | 0.769608 | 0.007605 | 200 |
