# --- Batch processing for chunk embeddings ---
BATCH_SIZE = 8  # You can tune this for your GPU/CPU
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import torch
from hf_stream_local import choose_source

# --- CONFIGURATION ---
# Output base directory for all runs
OUTPUT_BASE_DIR = "F:/quantas/outputs"
# Set these to switch between streaming/local dataset
USE_LOCAL_DATASET = True  # Set False to stream from HuggingFace
LOCAL_DATASET_PATH = "F:/quantas/datasets/wikimedia/wikipedia/20231101.en"
REMOTE_DATASET = "wikimedia/wikipedia"
REMOTE_CONFIG = "20231101.en"
SPLIT = "train"

MAX_DOCS = 100
TEXT_KEY = "text"
CHUNK_SIZE = 1024  # Number of tokens per chunk (will be capped by model max)
STREAMING = not USE_LOCAL_DATASET


#MODEL_PATH = "F:/quantas/models/bert-base-uncased"
#MODEL_PATH = "F:/quantas/models/Qwen/Qwen2.5-0.5B"
MODEL_PATH = "F:/quantas/models/google/gemma-3-1b-pt"


def _pick_device():
    """
    Prefer CUDA if it actually runs kernels. Some PyTorch wheels only ship sm_75+
    (Turing+); Pascal (e.g. GTX 1060, sm_61) then reports cuda available but fails at runtime.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda", dtype=torch.float32)
        x = x + 1
        torch.cuda.synchronize()
        return "cuda"
    except Exception as e:
        print(
            "CUDA is visible but this PyTorch build cannot run on this GPU "
            f"(e.g. Pascal sm_61 vs wheel built for sm_75+). Falling back to CPU. ({e})"
        )
        return "cpu"


DEVICE = _pick_device()
print(f"Using device: {DEVICE}")


# Load tokenizer and model (use_fast=True when available: full HF tokenizer API)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)


def _special_prefix_suffix_ids(tokenizer):
    """
    Return (prefix_ids, suffix_ids) so that prefix + content_ids + suffix matches
    tokenizer(..., add_special_tokens=True) for typical models (Gemma, Llama, BERT, etc.).

    Used when TokenizersBackend / GemmaTokenizer does not implement
    build_inputs_with_special_tokens; we infer affixes from encode-with vs encode-without
    on a short probe string.
    """
    for probe in ("a", "the", " hello", "x", "test"):
        inner = tokenizer(probe, add_special_tokens=False)["input_ids"]
        outer = tokenizer(probe, add_special_tokens=True)["input_ids"]
        if not inner:
            continue
        n = len(inner)
        for i in range(len(outer) - n + 1):
            if outer[i : i + n] == inner:
                return outer[:i], outer[i + n :]

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    add_bos = bool(getattr(tokenizer, "add_bos_token", True))
    add_eos = bool(getattr(tokenizer, "add_eos_token", False))
    pre = [bos_id] if (add_bos and bos_id is not None) else []
    suf = [eos_id] if (add_eos and eos_id is not None) else []
    return pre, suf


def _num_special_tokens(tokenizer):
    """How many special tokens wrap one content segment (prefix + suffix)."""
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return len(tokenizer.build_inputs_with_special_tokens([]))
    pre, suf = _special_prefix_suffix_ids(tokenizer)
    return len(pre) + len(suf)


def _wrap_token_ids_with_special_tokens(tokenizer, token_ids):
    """Prepend/append special token ids."""
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return tokenizer.build_inputs_with_special_tokens(token_ids)
    pre, suf = _special_prefix_suffix_ids(tokenizer)
    return pre + list(token_ids) + suf

# Detect model's max sequence length
try:
    model_max_length = getattr(model.config, 'max_position_embeddings', None)
    if model_max_length is None:
        model_max_length = getattr(tokenizer, 'model_max_length', 512)
    print(f"Model max sequence length: {model_max_length}")
except Exception:
    model_max_length = 512
    print("Could not detect model max sequence length, defaulting to 512.")

# Use the smaller of CHUNK_SIZE and model_max_length
CHUNK_SIZE = min(CHUNK_SIZE, model_max_length)
print(f"Using chunk size: {CHUNK_SIZE}")

# Use choose_source to get a generator for the dataset
if USE_LOCAL_DATASET:
    data_iter = choose_source(
        mode="dataset",
        local_dataset_path=LOCAL_DATASET_PATH,
        split=SPLIT,
        max_docs=MAX_DOCS,
        text_key=TEXT_KEY,
        streaming=False
    )
else:
    data_iter = choose_source(
        mode="dataset",
        dataset=REMOTE_DATASET,
        config_name=REMOTE_CONFIG,
        split=SPLIT,
        max_docs=MAX_DOCS,
        text_key=TEXT_KEY,
        streaming=True
    )

# Collect the texts from the generator, printing status for each
import time
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import itertools
texts = []
program_start = time.time()
start_load = time.time()
for idx, item in enumerate(data_iter, 1):
    texts.append(item)
    if idx >= MAX_DOCS:
        break
end_load = time.time()
print(f"Loaded {len(texts)} documents in {end_load - start_load:.2f} seconds.")



# Tokenize and keep word/token mapping
def chunk_token_ids_with_meta(text, tokenizer, chunk_size, doc_id=None):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    num_special = _num_special_tokens(tokenizer)
    max_chunk = chunk_size - num_special
    for i in range(0, len(token_ids), max_chunk):
        chunk_ids = token_ids[i:i+max_chunk]
        chunk_offsets = offsets[i:i+max_chunk]
        chunk_tokens = tokens[i:i+max_chunk]
        chunk_ids_with_special = _wrap_token_ids_with_special_tokens(tokenizer, chunk_ids)
        chunk_tensor = torch.tensor(chunk_ids_with_special, dtype=torch.long)
        # Build metadata for each token
        chunk_meta = []
        for j, (tok, off) in enumerate(zip(chunk_tokens, chunk_offsets)):
            word = text[off[0]:off[1]] if off[0] < off[1] else ''
            chunk_meta.append({
                'token_id': chunk_ids[j],
                'token_str': tok,
                'word': word,
                'span': off,
                'word_idx': j + i,
                'doc_id': doc_id
            })
        yield chunk_tensor, chunk_meta

# Parallel tokenization and chunking for each document
def tokenize_and_chunk_document(doc_idx_doc):
    doc_idx, doc = doc_idx_doc
    chunk_tensors = []
    for chunk_idx, (chunk_tensor, chunk_meta) in enumerate(chunk_token_ids_with_meta(doc, tokenizer, CHUNK_SIZE, doc_id=doc_idx), 1):
        input_len = chunk_tensor.shape[0]
        if input_len > model_max_length:
            print(f"ERROR: input_ids length {input_len} exceeds model max length {model_max_length}!")
            print(f"input_ids: {chunk_tensor}")
            raise ValueError(f"Input chunk too long: {input_len} > {model_max_length}")
        chunk_tensors.append((chunk_tensor, chunk_meta, (doc_idx, chunk_idx)))
    return chunk_tensors

# --- Efficient batch processing: group all chunks by length ---
BATCH_SIZE = 12  # You can tune this for your GPU/CPU
all_embeddings = []
total_chunk_time = 0.0
chunk_groups = {}  # length -> list of (chunk_tensor, (doc_idx, chunk_idx))
num_chunks = 0


# Parallelize tokenization and chunking
with ThreadPoolExecutor() as executor:
    results = list(executor.map(tokenize_and_chunk_document, enumerate(texts, 1)))


# Flatten results and group by chunk length
for doc_chunks in results:
    for chunk_tensor, chunk_meta, (doc_idx, chunk_idx) in doc_chunks:
        input_len = chunk_tensor.shape[0]
        chunk_groups.setdefault(input_len, []).append((chunk_tensor, chunk_meta, (doc_idx, chunk_idx)))
        num_chunks += 1



# Now process each group in batches, with progress and timing
total_batches = 0
total_chunks_processed = 0
total_chunks = sum(len(lst) for lst in chunk_groups.values())
embedding_start = time.time()
all_token_meta = []
for chunk_len, chunk_list in chunk_groups.items():
    num_batches = (len(chunk_list) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num in range(num_batches):
        batch_items = chunk_list[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE]
        batch_tensors = [item[0] for item in batch_items]
        batch_metas = [item[1] for item in batch_items]
        batch_tensor = torch.stack(batch_tensors, dim=0).to(DEVICE)
        start_batch = time.time()
        with torch.no_grad():
            outputs = model(input_ids=batch_tensor)
        end_batch = time.time()
        total_chunk_time += (end_batch - start_batch)
        emb = outputs.last_hidden_state  # [batch, chunk_len, emb_dim]
        # Unbatch: append each token embedding and meta separately
        for single_emb, single_meta in zip(emb.cpu(), batch_metas):
            for token_emb, token_meta in zip(single_emb, single_meta):
                # BF16/FP16 tensors: NumPy has no native bfloat16; store as float32
                all_embeddings.append(token_emb.detach().float().cpu().numpy())
                all_token_meta.append(token_meta)
        total_batches += 1
        total_chunks_processed += len(batch_tensors)
        if total_batches % 10 == 0 or batch_num == num_batches - 1:
            print(f"Processed {total_chunks_processed}/{total_chunks} chunks in {total_batches} batches (last batch: {end_batch - start_batch:.2f}s, chunk_len={chunk_len})")
embedding_end = time.time()
program_end = time.time()
print(f"Total embedding time for all chunks: {total_chunk_time:.2f} seconds (wall time: {embedding_end - embedding_start:.2f}s, batches: {total_batches}, chunks: {total_chunks_processed})")

# Save all embeddings and run info in a dedicated output folder
if all_embeddings:
    arr = np.stack(all_embeddings, axis=0)
    # Build dynamic folder and filenames
    model_name = os.path.basename(MODEL_PATH).replace("/", "_").replace("\\", "_")
    if USE_LOCAL_DATASET:
        dataset_name = os.path.basename(LOCAL_DATASET_PATH).replace("/", "_").replace("\\", "_")
    else:
        dataset_name = REMOTE_DATASET.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"output_{model_name}_{dataset_name}_{len(texts)}docs_{timestamp}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "embeddings.npy")
    # Save as a dict: {'embeddings': arr, 'token_meta': all_token_meta}
    np.save(fname, {'embeddings': arr, 'token_meta': all_token_meta}, allow_pickle=True)
    # Gather run info
    def human_readable_size(num_bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    file_size_bytes = os.path.getsize(fname)
    num_chunks = len(all_embeddings)
    run_info = {
        "model": MODEL_PATH,
        "dataset": LOCAL_DATASET_PATH if USE_LOCAL_DATASET else REMOTE_DATASET,
        "config": REMOTE_CONFIG if not USE_LOCAL_DATASET else None,
        "num_documents": len(texts),
        "num_tokens": arr.shape[0],
        "num_chunks": num_chunks,
        "embedding_shape": list(arr.shape),
        "chunk_size": CHUNK_SIZE,
        "max_model_length": model_max_length,
        "device": DEVICE,
        "total_embedding_time_sec": total_chunk_time,
        "output_file": fname,
        "output_file_size_bytes": file_size_bytes,
        "output_file_size_hr": human_readable_size(file_size_bytes),
        "timestamp": timestamp,
        "embedding_wall_time_sec": embedding_end - embedding_start,
        "total_program_runtime_sec": program_end - program_start,
        "output_schema": "{'embeddings': np.ndarray, 'token_meta': List[List[Dict]]}"
    }
    info_path = os.path.join(output_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        for k, v in run_info.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved embeddings and token metadata to {fname} with shape {arr.shape}")
    print(f"Saved run info to {info_path}")
else:
    print("No embeddings to save.")