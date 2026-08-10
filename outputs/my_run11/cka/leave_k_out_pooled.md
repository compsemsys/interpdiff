# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run11\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 9
- Elapsed: 19.90s

| Comparison | A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.603685 | 0.606094 | 0.004842 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946356 | 0.946643 | 0.001096 | 200 |
| own_embed | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.621047 | 0.625500 | 0.004950 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.491974 | 0.494722 | 0.008328 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.794581 | 0.793998 | 0.005402 | 200 |
| own_embed | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.584264 | 0.587212 | 0.006999 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.825886 | 0.826076 | 0.003188 | 200 |
| own_embed | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.643304 | 0.645929 | 0.004823 | 200 |
| own_embed | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.803466 | 0.804970 | 0.003624 | 200 |
