# Row-permutation CKA null (focused pooled)

- Source: `F:\code\Independent Study\outputs\my_run12\cka\document_cka_by_category.json`
- permute=Y, n_reps=100, seed=42
- Pairs: 21
- Elapsed: 39.18s

| Comparison | A | B | Full CKA | Row-perm mean | Row-perm std | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (response) | 0.596346 | 0.144338 | 0.006641 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | Qwen3.5-0.8B (summary) | 0.946685 | 0.140001 | 0.006003 | 200 |
| own_embed | Qwen3.5-0.8B (response) | Qwen3.5-0.8B (summary) | 0.616860 | 0.150161 | 0.006008 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (response) | 0.488762 | 0.089186 | 0.005863 | 200 |
| own_embed | gemma-3-1b-it (excerpt) | gemma-3-1b-it (summary) | 0.795063 | 0.078839 | 0.006256 | 200 |
| own_embed | gemma-3-1b-it (response) | gemma-3-1b-it (summary) | 0.587391 | 0.103887 | 0.006290 | 200 |
| own_embed | Qwen3.5-0.8B (excerpt) | gemma-3-1b-it (excerpt) | 0.824921 | 0.095525 | 0.005831 | 200 |
| own_embed | Qwen3.5-0.8B (response) | gemma-3-1b-it (response) | 0.659425 | 0.135672 | 0.005712 | 200 |
| own_embed | Qwen3.5-0.8B (summary) | gemma-3-1b-it (summary) | 0.802325 | 0.115637 | 0.005905 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.856102 | 0.136417 | 0.006296 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | 0.815225 | 0.123041 | 0.006458 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.860683 | 0.114524 | 0.005796 | 200 |
| same_text_cross_embedder | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | 0.827760 | 0.106368 | 0.006655 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.486326 | 0.080828 | 0.006440 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.619317 | 0.146050 | 0.006237 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=gemma-3-1b-it / text=corpus (excerpt) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.899111 | 0.072054 | 0.006617 | 200 |
| same_embedder_excerpt_vs_foreign_text | emb=Qwen3.5-0.8B / text=corpus (excerpt) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.870389 | 0.138351 | 0.005113 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (response) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (response) | 0.731637 | 0.107140 | 0.006338 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (response) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (response) | 0.748748 | 0.155583 | 0.005990 | 200 |
| same_embedder_own_vs_foreign_text | emb=gemma-3-1b-it / text=gemma-3-1b-it (summary) | emb=gemma-3-1b-it / text=Qwen3.5-0.8B (summary) | 0.837018 | 0.086135 | 0.007657 | 200 |
| same_embedder_own_vs_foreign_text | emb=Qwen3.5-0.8B / text=Qwen3.5-0.8B (summary) | emb=Qwen3.5-0.8B / text=gemma-3-1b-it (summary) | 0.882068 | 0.142736 | 0.005881 | 200 |
