# Independent Study — Embedding experiments

Small workspace for experimenting with contextual token embeddings using Hugging Face Transformers and a local embedding model.

## Overview

This project demonstrates how to obtain contextual token embeddings (per-token/subword vectors) from a Transformer model and how to pool them into sentence or per-word embeddings.

Files:
- `mainlad.py`: minimal example that loads a model and prints `last_hidden_state` (contextual token embeddings).
- `llama33.py`, `cache_model.py`, `requirements.txt`: supporting files and dependencies.

## Requirements

- Python 3.8+
- PyTorch
- transformers

Virtual environments are recommended to keep dependencies isolated.

Create and activate a `venv` (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create and activate a `venv` (Windows cmd):

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Create and activate a `venv` (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer manual install without a virtualenv:

```bash
pip install torch transformers
```

## Configuration

Set `model_path` inside `mainlad.py` to the path of a local or remote HF model that provides hidden states. Example in the repository uses:

```py
model_path = "F:/quantas/models/Qwen/Qwen3-Embedding-0.6B"
```

Adjust for your environment (local path or Hugging Face model identifier).


## Output schema

The output of `mainlad.py` is a `.npy` file containing a dictionary with two keys:

- `embeddings`: a numpy array of shape `(num_tokens, hidden_size)` containing all token embeddings.
- `token_meta`: a list of lists, where each inner list contains dictionaries for each token in a chunk. Each dictionary has:
	- `token_id`: integer token id
	- `token_str`: string representation of the token
	- `word`: original word or text span
	- `span`: (start, end) character offsets in the original text
	- `word_idx`: index of the token in the document

Example usage:

```python
import numpy as np
data = np.load('embeddings.npy', allow_pickle=True).item()
embeddings = data['embeddings']
token_meta = data['token_meta']
print(token_meta[0][0])  # Metadata for first token in first chunk
```

## What the code returns

In `mainlad.py`, the script loads the model and returns contextual token embeddings and token metadata. See Output schema above.

To obtain a sentence vector, pool across tokens (mean, sum, or a model-provided pooled output when available).

Example mean-pooling (handles attention mask):

```py
import torch
# word_embeddings: outputs.last_hidden_state (batch, seq_len, hidden_size)
# mask: inputs['attention_mask'] (batch, seq_len)
mask = inputs['attention_mask'].unsqueeze(-1)
masked_embeddings = word_embeddings * mask
sentence_embeddings = masked_embeddings.sum(dim=1) / mask.sum(dim=1)
```

To inspect tokens and see subword pieces:

```py
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(tokens)
```

To get per-word embeddings (merge subword tokens), group contiguous subword pieces belonging to the same word and average their token vectors.

## Usage

Run the example script:

```bash
python mainlad.py
```

Expected output includes the `inputs` dict, the `word_embeddings` tensor and its shape (e.g., `(1, seq_len, hidden_size)`).

## Next steps / Suggestions

- Add an example function in `mainlad.py` that returns a pooled sentence embedding (mean pooling) and a utility to merge subword tokens into word-level vectors.
- Optionally, add GPU support by moving model and tensors to `cuda` when available.

If you want, I can update `mainlad.py` with the pooling and token-merge utilities and a short demo run.
