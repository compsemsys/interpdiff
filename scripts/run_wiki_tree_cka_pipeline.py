"""
Build a categorized corpus from ``data/wiki_tree_random.jsonl``, then run
``run_categorized_corpus.py`` / ``pipeline_embed_explain_cka.py`` (full pipeline by
default: excerpt embeddings, generation from ``--instruction``, response
embeddings, CKA). All artifacts live under the run ``--out_dir`` or
``outputs/categorized_<timestamp>/``.

Defaults: Gemma 3 1B IT and Qwen 3.5 0.8B (local dirs):

  F:\\quantas\\models\\google\\gemma-3-1b-it
  F:\\quantas\\models\\Qwen\\Qwen3.5-0.8B

Override with ``--gemma`` / ``--qwen`` or env ``GEMMA3_MODEL_PATH`` / ``QWEN35_MODEL_PATH``.
Local checkpoints only (no ``--allow_remote``).

Key functions:
- ``build_corpus``: convert wiki-tree rows to run_categorized_corpus input schema.
- ``main``: validate model dirs, build corpus JSONL, and launch pipeline subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_dir(*candidates: str | None) -> str | None:
    for c in candidates:
        if c and Path(c).is_dir():
            return c
    return None


def _default_gemma() -> str:
    return _first_dir(
        os.environ.get("GEMMA3_MODEL_PATH"),
        r"F:\quantas\models\google\gemma-3-1b-it",
    ) or r"F:\quantas\models\google\gemma-3-1b-it"


def _default_qwen() -> str:
    return _first_dir(
        os.environ.get("QWEN35_MODEL_PATH"),
        r"F:\quantas\models\Qwen\Qwen3.5-0.8B",
    ) or r"F:\quantas\models\Qwen\Qwen3.5-0.8B"


def build_corpus(src: Path, dst: Path) -> None:
    # Normalize wiki-tree rows into the schema expected by run_categorized_corpus.
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append(
            {
                "doc_id": int(o["doc_id"]),
                "category": o.get("root_category", ""),
                "title": o.get("title", ""),
                "text": o["text"],
            }
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wiki-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "wiki_tree_random.jsonl",
        help="Source JSONL (wiki_tree_random output)",
    )
    p.add_argument(
        "--corpus-out",
        type=Path,
        default=REPO_ROOT / "data" / "wiki_tree_corpus_for_cka.jsonl",
        help="JSONL written for run_categorized_corpus",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        dest="out_dir",
        help="Passed through as --out_dir (default: timestamped outputs/categorized_*)",
    )
    p.add_argument(
        "--gemma",
        default=_default_gemma(),
        metavar="DIR",
        help="Local Gemma 3 checkpoint directory",
    )
    p.add_argument(
        "--qwen",
        default=_default_qwen(),
        metavar="DIR",
        help="Local Qwen 3.5 checkpoint directory",
    )
    p.add_argument("--words", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=768)
    p.add_argument(
        "--aggregation-level",
        choices=("word", "document", "both"),
        default="word",
        dest="aggregation_level",
        help="Forwarded to run_categorized_corpus --aggregation_level",
    )
    p.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    p.add_argument(
        "--instruction",
        default="Explain the following: {title}",
        help="Forwarded to run_categorized_corpus (supports {title}, {excerpt})",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        dest="skip_generate",
        help="Only excerpt embeddings (no LM generation or response CKA)",
    )
    p.add_argument(
        "--skip-cka",
        action="store_true",
        dest="skip_cka",
        help="Forwarded to run_categorized_corpus",
    )
    args = p.parse_args()

    gemma, qwen = args.gemma, args.qwen
    for label, path in (("gemma", gemma), ("qwen", qwen)):
        if not Path(path).is_dir():
            p.error(
                f"{label}: not a local directory ({path!r}). "
                f"Set GEMMA3_MODEL_PATH / QWEN35_MODEL_PATH or --gemma / --qwen."
            )

    build_corpus(args.wiki_jsonl, args.corpus_out)
    print(f"Wrote corpus ({args.corpus_out})")

    py = sys.executable
    rc = REPO_ROOT / "run_categorized_corpus.py"
    # Invoke the same interpreter/environment that launched this wrapper script.
    ts_cmd = [
        py,
        str(rc),
        "--corpus",
        str(args.corpus_out),
        "--models",
        gemma,
        qwen,
        "--words",
        str(args.words),
        "--batch_size",
        str(args.batch_size),
        "--chunk_size",
        str(args.chunk_size),
        "--aggregation_level",
        args.aggregation_level,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--instruction",
        args.instruction,
    ]
    if args.out_dir:
        ts_cmd += ["--out_dir", str(args.out_dir)]
    if args.skip_generate:
        ts_cmd.append("--skip_generate")
    if args.skip_cka:
        ts_cmd.append("--skip_cka")

    print("Running:", " ".join(ts_cmd))
    subprocess.run(ts_cmd, cwd=str(REPO_ROOT), check=True)

    out_dir = args.out_dir
    if out_dir is None:
        outs = sorted(
            (REPO_ROOT / "outputs").glob("categorized_*"),
            key=lambda x: x.stat().st_mtime,
        )
        if not outs:
            raise SystemExit("No outputs/categorized_* directory found after run")
        out_dir = outs[-1]
    print(f"Done. Run directory: {out_dir.resolve()}")
    print("CKA JSON (if run):", out_dir.resolve() / "cka")


if __name__ == "__main__":
    main()
