"""
Entry point identical to ``run_categorized_corpus.py`` (same CLI).

Use this when you want a memorable name for the full pipeline:
excerpt (first N words) → token + word embeddings → title/excerpt-prompted generation
→ response embeddings → pairwise linear CKA, with every artifact under ``--out_dir``.

Example:
  .\\.venv313\\Scripts\\python.exe pipeline_embed_explain_cka.py \\
    --corpus data/wiki_tree_corpus_for_cka.jsonl --out_dir outputs/wiki_run1 \\
    --models F:\\\\quantas\\\\models\\\\google\\\\gemma-3-1b-it F:\\\\quantas\\\\models\\\\Qwen\\\\Qwen3.5-0.8B \\
    --words 200 --max_new_tokens 128
"""

from run_categorized_corpus import main

if __name__ == "__main__":
    # Intentional passthrough: keep one canonical implementation in run_categorized_corpus.
    main()
