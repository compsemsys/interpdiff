import numpy as np
import sys

# Usage: python compare_word_embeddings.py <file1.npy> <file2.npy>

def load_meta(path):
    data = np.load(path, allow_pickle=True).item()
    return data['meta']

def get_word_span_set(meta):
    return set((m['doc_id'], m['span'], m['word']) for m in meta)


# Hardcoded file paths
file1 = 'F:/quantas/outputs/output_bert-base-uncased_20231101.en_100docs_20260318_075941/word_embeddings_merged_agnostic.npy'
#file2 = 'F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/word_embeddings_merged_agnostic.npy'
file2 = 'F:/quantas/outputs/output_gemma-3-1b-pt_20231101.en_100docs_20260407_234039/word_embeddings_merged_agnostic.npy'

meta1 = load_meta(file1)
meta2 = load_meta(file2)
set1 = get_word_span_set(meta1)
set2 = get_word_span_set(meta2)

# Print first five (doc_id, span, word) tuples from each file
print("\nFirst five word/span tuples from file 1:")
for m in meta1[:5]:
    print((m['doc_id'], m['span'], m['word']))

print("\nFirst five word/span tuples from file 2:")
for m in meta2[:5]:
    print((m['doc_id'], m['span'], m['word']))

common = set1 & set2
only1 = set1 - set2
only2 = set2 - set1
print(f"File 1: {file1}")
print(f"  Unique words/spans: {len(set1)}")
print(f"File 2: {file2}")
print(f"  Unique words/spans: {len(set2)}")
print(f"Common words/spans: {len(common)}")
print(f"Words/spans only in file 1: {len(only1)}")
print(f"Words/spans only in file 2: {len(only2)}")

# Print first five mismatches in order for file 1
if only1:
    print("First five mismatches in file 1 (in order):")
    ordered1 = [t for t in meta1 if (t['doc_id'], t['span'], t['word']) in only1][:8]
    for m in ordered1:
        print((m['doc_id'], m['span'], m['word']))

# Print first five mismatches in order for file 2
if only2:
    print("First five mismatches in file 2 (in order):")
    ordered2 = [t for t in meta2 if (t['doc_id'], t['span'], t['word']) in only2][:8]
    for m in ordered2:
        print((m['doc_id'], m['span'], m['word']))
