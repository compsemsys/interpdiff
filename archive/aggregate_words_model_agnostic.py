import argparse

import numpy as np

from doc_merge_agnostic import merge_token_embeddings_to_docs
from word_merge_agnostic import merge_token_embeddings_to_words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/embeddings.npy",
        help="Path to token-level embeddings.npy",
    )
    parser.add_argument(
        "--aggregation_level",
        choices=("word", "document"),
        default="word",
        help="Output aggregation level from token embeddings.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    data = np.load(input_path, allow_pickle=True).item()
    embeddings = data["embeddings"]
    token_meta = data["token_meta"]

    if args.aggregation_level == "word":
        arr, meta = merge_token_embeddings_to_words(embeddings, token_meta)
        output_name = "word_embeddings_merged_agnostic.npy"
    else:
        arr, meta = merge_token_embeddings_to_docs(embeddings, token_meta)
        output_name = "document_embeddings_merged_agnostic.npy"

    for i in range(min(5, len(meta))):
        m = meta[i]
        if args.aggregation_level == "word":
            print(f"word {i}: {m.get('word')!r} doc_id={m.get('doc_id')}")
        else:
            print(f"doc {i}: doc_id={m.get('doc_id')} token_count={m.get('token_count')}")

    output_path = input_path.replace("embeddings.npy", output_name)
    np.save(output_path, {"embeddings": arr, "meta": meta}, allow_pickle=True)
    print(
        f"Saved merged {args.aggregation_level}-level embeddings and metadata to {output_path}"
    )


if __name__ == "__main__":
    main()
