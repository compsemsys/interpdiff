# Leave-k-out CKA (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run13\cka\document_cka_by_category.json`
- drop_k=10, n_reps=100, seed=0
- Pairs: 21
- Elapsed: 40.29s

| Comparison | A | B | Full CKA | Leave-k-out mean | Leave-k-out std | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.593985 | 0.596628 | 0.005257 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.876160 | 0.876329 | 0.003048 | 200 |
| own_embed | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.620666 | 0.624559 | 0.005453 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.469695 | 0.471944 | 0.007310 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.734561 | 0.734407 | 0.007943 | 200 |
| own_embed | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.574064 | 0.576978 | 0.009229 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.825112 | 0.003220 | 200 |
| own_embed | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619734 | 0.622443 | 0.005398 | 200 |
| own_embed | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.766430 | 0.769608 | 0.007605 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.878443 | 0.878906 | 0.001671 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.872858 | 0.873287 | 0.001497 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.867585 | 0.868264 | 0.002440 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.858015 | 0.858680 | 0.001638 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.470440 | 0.473008 | 0.009324 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.587980 | 0.591297 | 0.006425 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.769987 | 0.771342 | 0.008278 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.826770 | 0.828025 | 0.005094 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.661026 | 0.662032 | 0.007930 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.697558 | 0.699637 | 0.007045 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.827477 | 0.828126 | 0.009401 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.844072 | 0.844185 | 0.007565 | 200 |
