import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import argparse

def center_gram(K):
    """Center a Gram matrix."""
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def linear_cka(X, Y):
    """Compute linear CKA between two matrices X and Y."""
    K = linear_kernel(X)
    L = linear_kernel(Y)
    Kc = center_gram(K)
    Lc = center_gram(L)
    cka = np.trace(Kc @ Lc) / (np.linalg.norm(Kc, 'fro') * np.linalg.norm(Lc, 'fro'))
    return cka

def main():

    # Hardcoded paths to embedding files
    embeddings1 = "F:/quantas/outputs/output_bert-base-uncased_20231101.en_100docs_20260318_075941/word_embeddings.npy"
    embeddings2 = "F:/quantas/outputs/output_Qwen2.5-0.5B_20231101.en_100docs_20260318_001538/word_embeddings.npy"

    X = np.load(embeddings1, allow_pickle=True)
    Y = np.load(embeddings2, allow_pickle=True)



    # Extract embeddings arrays
    if isinstance(X, np.ndarray) and X.shape == () and isinstance(X.item(), dict) and 'embeddings' in X.item():
        X_emb = X.item()['embeddings']
    else:
        print("Could not extract embeddings from X.")
        return
    if isinstance(Y, np.ndarray) and Y.shape == () and isinstance(Y.item(), dict) and 'embeddings' in Y.item():
        Y_emb = Y.item()['embeddings']
    else:
        print("Could not extract embeddings from Y.")
        return

    if X_emb.shape != Y_emb.shape:
        print(f"Shape mismatch: {X_emb.shape} vs {Y_emb.shape}")
        return

    cka_score = linear_cka(X_emb, Y_emb)
    print(f"Linear CKA score: {cka_score:.6f}")

    if X.shape != Y.shape:
        print(f"Shape mismatch: {X.shape} vs {Y.shape}")
        return

    # Commented out CKA calculation until extraction is fixed
    # cka_score = linear_cka(X, Y)
    # print(f"Linear CKA score: {cka_score:.6f}")

if __name__ == '__main__':
    main()
