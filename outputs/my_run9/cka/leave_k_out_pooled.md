# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run9\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 9
- Elapsed: 17.24s

| A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.595318 | 0.597910 | 0.005263 | 200 |
| Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.918090 | 0.918622 | 0.002084 | 200 |
| Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.612772 | 0.617222 | 0.005320 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.471683 | 0.474008 | 0.007285 | 200 |
| gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.764364 | 0.763867 | 0.006306 | 200 |
| gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.553239 | 0.556685 | 0.007415 | 200 |
| Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825886 | 0.826076 | 0.003188 | 200 |
| Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619734 | 0.622443 | 0.005398 | 200 |
| Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.784323 | 0.786228 | 0.002930 | 200 |
