"""hf_stream_local.py

Utilities to either stream a Hugging Face dataset (without downloading everything)
or load a local/Hugging Face model. The streaming generator uses the datasets
streaming API and `itertools.islice` to stop after `max_docs` items so only the
necessary shards/files are requested.

Example usage (stream first 1000 docs):
    python hf_stream_local.py --mode dataset --dataset wikitext --split train --max_docs 1000 --text_key text

Example usage (load local model):
    python hf_stream_local.py --mode model --model_path /path/to/local/model --local_only
"""
from typing import Generator, Optional, Any, Dict, Tuple
from itertools import islice
import logging


def stream_hf_dataset(dataset_name: Optional[str] = None,
                      config_name: Optional[str] = None,
                      split: str = "train",
                      streaming: bool = True,
                      max_docs: Optional[int] = None,
                      text_key: Optional[str] = None,
                      local_dataset_path: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
    """Stream examples from a Hugging Face dataset.

    - Uses `datasets.load_dataset(..., streaming=True)` so shards/files are
      fetched as needed.
    - If `max_docs` is set, only that many examples are yielded; this causes
      the iterator to stop early and avoids downloading remaining shards.
    - If `text_key` is provided, yields the value of that key for each example
      (useful for datasets where the text field is named `text`, `content`, etc.).

    Returns a generator of raw examples (or the selected text field when
    `text_key` is provided).
    """
    try:
        from datasets import load_dataset, load_from_disk
    except Exception as exc:  # pragma: no cover - import/runtime guard
        raise ImportError("Please install 'datasets' (pip install datasets)") from exc

    if local_dataset_path:
        # Load from disk (local dataset previously saved with save_to_disk)
        ds = load_from_disk(local_dataset_path)
        # If split exists, select it
        if split in ds:
            ds_iter = ds[split]
        else:
            ds_iter = ds
        # Not streaming, so just use islice if max_docs
        if max_docs is not None:
            ds_iter = islice(ds_iter, max_docs)
    else:
        # Always use the plain split (e.g. "train") and use islice to limit docs.
        if config_name:
            ds_iter = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
        else:
            ds_iter = load_dataset(dataset_name, split=split, streaming=streaming)
        if max_docs is not None:
            ds_iter = islice(ds_iter, max_docs)

    for example in ds_iter:
        if text_key is not None:
            yield example.get(text_key, example)
        else:
            yield example


def load_hf_model(model_name_or_path: str,
                  local_only: bool = False,
                  device: str = "cpu") -> Tuple[Any, Any]:
    """Load tokenizer and model from Hugging Face or a local path.

    - If `local_only=True`, transformers will only load from local files.
    - Returns `(tokenizer, model)`; the model is moved to `device`.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - import/runtime guard
        raise ImportError("Please install 'transformers' (pip install transformers)") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, local_files_only=local_only)
    try:
        model.to(device)
    except Exception:
        logging.warning("Could not move model to device '%s'; proceeding on default device.", device)
    return tokenizer, model


def choose_source(mode: str = "dataset", **kwargs):
    """Convenience wrapper to either return a dataset generator or load a model.

    - mode == 'dataset' -> returns a generator from `stream_hf_dataset`
    - mode == 'model' -> returns `(tokenizer, model)` from `load_hf_model`
    """
    if mode == "dataset":
        return stream_hf_dataset(dataset_name=kwargs.get("dataset"),
                                 config_name=kwargs.get("config_name"),
                                 split=kwargs.get("split", "train"),
                                 streaming=kwargs.get("streaming", True),
                                 max_docs=kwargs.get("max_docs"),
                                 text_key=kwargs.get("text_key"),
                                 local_dataset_path=kwargs.get("local_dataset_path"))
    elif mode == "model":
        return load_hf_model(kwargs.get("model_path"),
                             local_only=kwargs.get("local_only", False),
                             device=kwargs.get("device", "cpu"))
    else:
        raise ValueError("mode must be 'dataset' or 'model'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream HF datasets or load local/HF models")
    parser.add_argument("--mode", choices=["dataset", "model"], required=True)
    parser.add_argument("--dataset", help="Hugging Face dataset id (e.g. wikitext)")
    parser.add_argument("--config", default=None, help="Dataset config name (e.g. 20231101.en for wikimedia/wikipedia)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--text_key", default=None, help="Field name for text in dataset examples")
    parser.add_argument("--model_path", help="Model name or local path")
    parser.add_argument("--local_dataset_path", default=None, help="Path to local dataset saved with save_to_disk")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false",
                        help="Disable streaming (downloads full split)")
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    if args.mode == "dataset":
        if not args.dataset and not args.local_dataset_path:
            parser.error("--dataset or --local_dataset_path is required when --mode dataset")
        gen = stream_hf_dataset(dataset_name=args.dataset, config_name=args.config, split=args.split,
                                streaming=args.streaming, max_docs=args.max_docs, text_key=args.text_key,
                                local_dataset_path=args.local_dataset_path)
        count = 0
        for item in gen:
            count += 1
            if count <= 5:
                print(f"Example {count}:", item)
        print(f"Streamed {count} examples (stopped at max_docs={args.max_docs})")
    else:
        if not args.model_path:
            parser.error("--model_path is required when --mode model")
        tok, mod = load_hf_model(args.model_path, local_only=args.local_only, device=args.device)
        print("Loaded tokenizer:", type(tok))
        print("Loaded model:", type(mod))
