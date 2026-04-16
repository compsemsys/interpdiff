import numpy as np

#data = np.load('F:/quantas/outputs/output_bert-base-uncased_20231101.en_100docs_20260317_233805/embeddings.npy', allow_pickle=True).item()
data = np.load('F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/embeddings.npy', allow_pickle=True).item()
embeddings = data['embeddings']
token_meta = data['token_meta']

# Inspect first 5 rows
for i in range(10):
    print(embeddings[0].shape)
    print(f"Embedding[{i}.shape]: {embeddings[i].shape}")
    print(f"Metadata[{i}]: {token_meta[i]}")