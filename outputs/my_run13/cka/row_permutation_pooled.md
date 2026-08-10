# Row-permutation CKA null (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run13\cka\document_cka_by_category.json`
- permute=Y, n_reps=100, seed=42
- Pairs: 21
- Elapsed: 38.47s

| Comparison | A | B | Full CKA | Row-perm mean | Row-perm std | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.593985 | 0.146003 | 0.006673 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.876160 | 0.139571 | 0.005868 | 200 |
| own_embed | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.620666 | 0.151393 | 0.005769 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.469695 | 0.085734 | 0.005376 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.734561 | 0.078149 | 0.006060 | 200 |
| own_embed | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.574064 | 0.099378 | 0.007038 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.095525 | 0.005831 | 200 |
| own_embed | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.619734 | 0.131244 | 0.006232 | 200 |
| own_embed | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.766430 | 0.114607 | 0.006369 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.878443 | 0.125225 | 0.006473 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.872858 | 0.118806 | 0.006708 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.867585 | 0.109607 | 0.005828 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.858015 | 0.108304 | 0.006517 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.470440 | 0.076815 | 0.006056 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.587980 | 0.138606 | 0.006275 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.769987 | 0.073031 | 0.006194 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.826770 | 0.133207 | 0.005401 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.661026 | 0.098202 | 0.006062 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.697558 | 0.148404 | 0.005987 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.827477 | 0.085780 | 0.007094 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.844072 | 0.137271 | 0.005931 | 200 |
