import numpy as np

# Load embeddings and token metadata
input_path = 'F:/quantas/outputs/output_bert-base-uncased_20231101.en_100docs_20260318_075941/embeddings.npy'
#input_path = 'F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/embeddings.npy'
data = np.load(input_path, allow_pickle=True).item()
embeddings = data['embeddings']
token_meta = data['token_meta']


# Flatten token_meta if needed
if isinstance(token_meta[0], list):
    token_meta = [item for chunk in token_meta for item in chunk]

# --- Meta inspection ---
from collections import defaultdict, Counter

# Count unique (doc_id, span) pairs
span_counter = Counter((m['doc_id'], m['span']) for m in token_meta)
unique_spans = list(span_counter.keys())
print(f"Unique (doc_id, span) pairs: {len(unique_spans)}")
multi_token_spans = [k for k, v in span_counter.items() if v > 1]
print(f"Pairs with multiple tokens: {len(multi_token_spans)}")
if multi_token_spans:
    print("Sample multi-token (doc_id, span) pairs and their words (max 5 shown):")
    for doc_id, span in multi_token_spans[:5]:
        words = [m['word'] for m in token_meta if m['doc_id'] == doc_id and m['span'] == span]
        print(f"doc_id={doc_id}, span={span}, words={words}")



# --- Group tokens by (doc_id, overlapping span) ---
def merge_overlapping_spans(meta_list):
    from operator import itemgetter
    merged = []
    meta_list = sorted(meta_list, key=lambda m: (m['doc_id'], m['span'][0], m['span'][1]))
    current_group = []
    current_doc = None
    current_span = None
    for m in meta_list:
        doc_id = m['doc_id']
        span = m['span']
        if not current_group:
            current_group = [m]
            current_doc = doc_id
            current_span = list(span)
        elif doc_id == current_doc and span[0] <= current_span[1]:
            # Overlapping or contiguous
            current_group.append(m)
            current_span[1] = max(current_span[1], span[1])
        else:
            merged.append((current_doc, tuple(current_span), list(current_group)))
            current_group = [m]
            current_doc = doc_id
            current_span = list(span)
    if current_group:
        merged.append((current_doc, tuple(current_span), list(current_group)))
    return merged


if 'token_str' in token_meta[0] and any(m['token_str'].startswith('Ġ') for m in token_meta):
    # GPT-like: split token_meta directly at 'Ġ' boundaries
    print("Detected GPT-like model. Splitting tokens at 'Ġ' boundaries.")
    word_groups = []
    current = []
    for m in token_meta:
        ts = m['token_str']
        if ts.startswith('Ġ') and current:
            word_groups.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        word_groups.append(current)
    print(f"Word groups (GPT-like): {len(word_groups)}")
    print("Sample word groups (max 5 shown):")
    for i in range(min(5, len(word_groups))):
        group = word_groups[i]
        doc_id = group[0]['doc_id']
        span = (group[0]['span'][0], group[-1]['span'][1])
        words = [m['word'] for m in group]
        words_str = str(words)[:100] + ('...' if len(str(words)) > 100 else '')
        print(f"doc_id={doc_id}, span={span}, words={words_str}")
    merged_groups = [(group[0]['doc_id'], (group[0]['span'][0], group[-1]['span'][1]), group) for group in word_groups]
else:
    merged_groups = merge_overlapping_spans(token_meta)
    print(f"Merged groups (by overlapping span): {len(merged_groups)}")
    print("Sample merged groups (max 5 shown):")
    for i in range(min(5, len(merged_groups))):
        doc_id, span, group = merged_groups[i]
        words = [m['word'] for m in group]
        words_str = str(words)[:100] + ('...' if len(str(words)) > 100 else '')
        print(f"doc_id={doc_id}, span={span}, words={words_str}")


# Optimize: build id(m) to index mapping for fast lookup
meta_id_to_idx = {id(m): i for i, m in enumerate(token_meta)}



# Auto-detect model type
model_type = None
if 'token_str' in token_meta[0]:
    if any(m['token_str'].startswith('Ġ') for m in token_meta):
        model_type = 'gpt_like'  # Qwen, GPT, RoBERTa
    elif any(m['token_str'].startswith('##') for m in token_meta):
        model_type = 'bert_like'
    else:
        model_type = 'other'

word_embeddings = []
word_meta = []
total_groups = len(merged_groups)

if model_type == 'gpt_like':
    # For GPT-like models, split tokens at 'Ġ', 'Ċ', and 'ĊĊ' boundaries
    for i, (doc_id, span, group) in enumerate(merged_groups):
        if i % 50000 == 0:
            print(f"Aggregating group {i+1}/{total_groups}...")
        import string
        segments = []
        current = []
        for m in group:
            ts = m['token_str']
            # Treat 'Ċ' and punctuation (except apostrophe and hyphen) as boundaries
            is_boundary = False
            if 'Ċ' in ts:
                is_boundary = True
            else:
                for c in ts:
                    if c in string.punctuation and c not in "'-":
                        is_boundary = True
                        break
            if is_boundary:
                if current:
                    segments.append(current)
                    current = []
                # Do not include this token in any segment
            else:
                current.append(m)
        if current:
            segments.append(current)
        for seg in segments:
            indices = [meta_id_to_idx[id(m)] for m in seg]
            import re
            # Harmonized span calculation: increment start for every leading special char stripped
            raw_word = ''
            start = seg[0]['span'][0]
            for m in seg:
                ts = m['token_str']
                i = 0
                while i < len(ts) and (ts[i] == 'Ġ' or ts[i] == 'Ċ' or ts[i] in string.punctuation and ts[i] not in "'-"):
                    start += 1
                    i += 1
                raw_word += ts[i:]
            raw_word = raw_word.lower()
            word = re.sub(r"[^\w'-]", " ", raw_word)
            word = ''.join(word.split())
            seg_span = (start, start + len(word))
            pooled = embeddings[indices].mean(axis=0)
            if word:
                word_embeddings.append(pooled)
                word_meta.append({
                    'doc_id': doc_id,
                    'span': seg_span,
                    'word': word,
                    'token_indices': indices,
                    'subwords': [m['word'] for m in seg],
                    'token_strs': [m['token_str'] for m in seg]
                })
elif model_type == 'bert_like':
    # For BERT-like models, merge subwords using '##' logic and normalize punctuation
    import string
    import re
    for i, (doc_id, span, group) in enumerate(merged_groups):
        if i % 50000 == 0:
            print(f"Aggregating group {i+1}/{total_groups}...")
        indices = [meta_id_to_idx[id(m)] for m in group]
        raw_word = ''
        start = group[0]['span'][0]
        for m in group:
            ts = m['token_str']
            i = 0
            # Only increment for punctuation at the start, not for '##'
            while i < len(ts) and (ts[i] in string.punctuation and ts[i] not in "'-"):
                start += 1
                i += 1
            # Strip '##' but do not advance span
            if ts[i:i+2] == '##':
                i += 2
            raw_word += ts[i:]
        raw_word = raw_word.lower()
        word = re.sub(r"[^\w'-]", " ", raw_word)
        word = ''.join(word.split())
        new_span = (start, start + len(word))
        pooled = embeddings[indices].mean(axis=0)
        if word:
            word_embeddings.append(pooled)
            word_meta.append({
                'doc_id': doc_id,
                'span': new_span,
                'word': word,
                'token_indices': indices,
                'subwords': [m['word'] for m in group],
                'token_strs': [m['token_str'] for m in group]
            })
else:
    # Fallback: treat each merged group as a word
    for i, (doc_id, span, group) in enumerate(merged_groups):
        if i % 50000 == 0:
            print(f"Aggregating group {i+1}/{total_groups}...")
        indices = [meta_id_to_idx[id(m)] for m in group]
        pooled = embeddings[indices].mean(axis=0)
        word = ''.join([m['word'] for m in group])
        word_embeddings.append(pooled)
        word_meta.append({
            'doc_id': doc_id,
            'span': span,
            'word': word,
            'token_indices': indices,
            'subwords': [m['word'] for m in group],
            'token_strs': [m['token_str'] for m in group] if 'token_str' in group[0] else None
        })

# Inspect output meta
print(f"Output word_meta count: {len(word_meta)}")
print("Sample output word_meta (max 15 shown):")
for i in range(min(15, len(word_meta))):
    print(word_meta[i])

# Save new arrays
output_path = input_path.replace('embeddings.npy', 'word_embeddings_merged.npy')
np.save(output_path, {'embeddings': np.stack(word_embeddings, axis=0), 'meta': word_meta}, allow_pickle=True)
print(f"Saved merged word-level embeddings and metadata to {output_path}")



# Inspect input token_meta (print full meta for first 5 tokens)
print(f"Input token_meta count: {len(token_meta)}")
print("Sample input token_meta (full):")
for i in range(min(5, len(token_meta))):
    print(token_meta[i])
