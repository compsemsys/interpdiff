# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run12\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 21
- Elapsed: 39.97s

| Comparison | A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.596346 | 0.598685 | 0.004926 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946685 | 0.946966 | 0.001145 | 200 |
| own_embed | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.616860 | 0.621252 | 0.004967 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.488762 | 0.491375 | 0.008129 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.795063 | 0.794482 | 0.005429 | 200 |
| own_embed | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.587391 | 0.590194 | 0.007408 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.825112 | 0.003220 | 200 |
| own_embed | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.659425 | 0.661914 | 0.004356 | 200 |
| own_embed | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.802325 | 0.803808 | 0.003563 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.856102 | 0.856739 | 0.002054 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.815225 | 0.816301 | 0.002470 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.860683 | 0.861316 | 0.001717 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.827760 | 0.828502 | 0.003617 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.486326 | 0.488027 | 0.008676 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.619317 | 0.623120 | 0.006354 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.899111 | 0.899495 | 0.005299 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.870389 | 0.871351 | 0.002629 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.731637 | 0.732953 | 0.006480 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.748748 | 0.750994 | 0.004578 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.837018 | 0.838394 | 0.006683 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.882068 | 0.882689 | 0.002146 | 200 |
