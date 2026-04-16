import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/embeddings.npy",
    help="Path to token-level embeddings.npy",
)
args = parser.parse_args()
input_path = args.input_path
data = np.load(input_path, allow_pickle=True).item()
embeddings = data["embeddings"]
token_meta = data["token_meta"]


def flatten_token_meta(meta):
    if isinstance(meta[0], list):
        return [item for chunk in meta for item in chunk]
    return meta


def token_starts_new_word(curr, prev):
    if prev is None:
        return True
    if curr["doc_id"] != prev["doc_id"]:
        return True

    curr_start, _ = curr["span"]
    _, prev_end = prev["span"]

    curr_ts = curr.get("token_str", "")
    prev_ts = prev.get("token_str", "")
    prev_word = prev.get("word", "")

    # BERT-style continuation marker: always continue the current word.
    if curr_ts.startswith("##"):
        return False

    # GPT/Roberta-style space/newline markers indicate a new lexical item.
    if curr_ts.startswith("Ġ") or curr_ts.startswith("Ċ"):
        return True

    # Common sentencepiece boundary marker.
    if curr_ts.startswith("▁"):
        return True

    # A visible char gap means whitespace or punctuation boundary in source text.
    if curr_start > prev_end:
        return True

    # If previous token text contains explicit whitespace/newline, the next token
    # should begin a new lexical item even when spans are contiguous.
    if prev_word and prev_word[-1].isspace():
        return True

    # If previous token already encoded an explicit boundary marker, split.
    if prev_ts.startswith("Ċ"):
        return True

    return False


def group_tokens(meta):
    meta_sorted = sorted(meta, key=lambda m: (m["doc_id"], m["span"][0], m["span"][1]))
    groups = []
    current = []
    prev = None

    for m in meta_sorted:
        if token_starts_new_word(m, prev):
            if current:
                groups.append(current)
            current = [m]
        else:
            current.append(m)
        prev = m

    if current:
        groups.append(current)
    return groups


def canonical_word_and_span(group):
    # Build absolute-position chars from token-level `word` strings and spans.
    # This avoids tokenizer marker assumptions and keeps spans aligned.
    pos_to_char = {}
    for m in group:
        start, end = m["span"]
        token_word = m.get("word", "")
        span_len = max(0, end - start)
        usable = token_word[:span_len]
        for i, ch in enumerate(usable):
            abs_pos = start + i
            if abs_pos not in pos_to_char:
                pos_to_char[abs_pos] = ch

    if not pos_to_char:
        return "", None

    allowed = []
    for pos in sorted(pos_to_char):
        ch = pos_to_char[pos]
        if ch.isalnum() or ch in ["-", "'"]:
            allowed.append((pos, ch))

    if not allowed:
        return "", None

    # Normalize output word consistently across models.
    word = "".join(ch for _, ch in allowed).lower()
    word = re.sub(r"'+$", "", word)
    if not re.search(r"[a-zA-Z0-9]", word):
        return "", None

    span = (allowed[0][0], allowed[-1][0] + 1)
    return word, span


def safe_text(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


token_meta = flatten_token_meta(token_meta)
meta_id_to_idx = {id(m): i for i, m in enumerate(token_meta)}
groups = group_tokens(token_meta)

word_embeddings = []
word_meta = []

for group in groups:
    word, span = canonical_word_and_span(group)
    if not word:
        continue

    indices = [meta_id_to_idx[id(m)] for m in group]
    pooled = embeddings[indices].mean(axis=0)
    word_embeddings.append(pooled)
    word_meta.append(
        {
            "doc_id": group[0]["doc_id"],
            "span": span,
            "word": word,
            "token_indices": indices,
            "subwords": [m.get("word", "") for m in group],
            "token_strs": [m.get("token_str", "") for m in group],
        }
    )

    if len(word_meta) <= 5:
        print(f"Set {len(word_meta)}:")
        print("  Tokens:")
        for m in group:
            print(
                f"    token_id={m.get('token_id', '?')}, token_str={safe_text(m.get('token_str', ''))}, "
                f"word={safe_text(m.get('word', ''))}, span={m['span']}"
            )
        print(f"  Merged word: {word}, span={span}")

output_path = input_path.replace("embeddings.npy", "word_embeddings_merged_agnostic.npy")
np.save(output_path, {"embeddings": np.stack(word_embeddings, axis=0), "meta": word_meta}, allow_pickle=True)
print(f"Saved merged word-level embeddings and metadata to {output_path}")
