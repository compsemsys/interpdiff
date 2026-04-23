"""
Compare Hugging Face Gemma 4 checkpoint key names to Transformers model state_dict keys.

Use this to validate whether checkpoint tensors are stored under a top-level ``model.`` prefix
while ``Gemma4Model`` / ``Gemma4ForConditionalGeneration`` expect unprefixed submodule names
(``language_model``, ``vision_tower``, ...), which produces UNEXPECTED + MISSING load reports.

Does not load full weight tensors - only key names (and optional shard listing via safetensors).

Usage:
  python scripts/check_gemma_checkpoint_keys.py
  python scripts/check_gemma_checkpoint_keys.py --model-id google/gemma-4-E2B-it --architecture conditional
  python scripts/check_gemma_checkpoint_keys.py --checkpoint F:/path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def _instantiate_for_keys(model_factory: Callable, config):
    """Build model on meta / empty weights so state_dict keys match real load without full RAM."""
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            return model_factory(config)
    except Exception:
        pass
    import torch

    try:
        with torch.device("meta"):
            return model_factory(config)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not build a zero-weight model for key comparison. "
            "Install `accelerate` (recommended) or use a PyTorch build that supports "
            "`torch.device('meta')` for this model."
        ) from e


def _load_checkpoint_keys(checkpoint_path: str) -> set[str]:
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        return set(f.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E2B-it",
        help="Hub model id (used for config and default checkpoint download).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model.safetensors. If omitted, downloads/resolves via huggingface_hub.",
    )
    parser.add_argument(
        "--architecture",
        choices=("base", "conditional"),
        default="base",
        help="base = AutoModel (Gemma4Model). conditional = AutoModelForImageTextToText (Gemma4ForConditionalGeneration).",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(args.model_id)

    if args.architecture == "base":
        model_factory = AutoModel.from_config
        label = "AutoModel / Gemma4Model"
    else:
        model_factory = AutoModelForImageTextToText.from_config
        label = "AutoModelForImageTextToText / Gemma4ForConditionalGeneration"

    model = _instantiate_for_keys(model_factory, config)
    model_keys = set(model.state_dict().keys())

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = hf_hub_download(args.model_id, "model.safetensors")

    ckpt_keys = _load_checkpoint_keys(ckpt_path)
    stripped = {k[6:] if k.startswith("model.") else k for k in ckpt_keys}
    prefixed_from_model = {"model." + k for k in model_keys}

    inter_raw = ckpt_keys & model_keys
    inter_stripped = stripped & model_keys
    inter_prefixed = ckpt_keys & prefixed_from_model

    only_prefixed = sum(1 for k in ckpt_keys if k.startswith("model."))
    non_prefixed = sorted(k for k in ckpt_keys if not k.startswith("model."))

    missing_model_key_in_ckpt = [k for k in model_keys if ("model." + k) not in ckpt_keys]

    full_match_as_is = ckpt_keys == model_keys
    full_match_if_strip = stripped == model_keys

    print(f"Model id:        {args.model_id}")
    print(f"Architecture:    {label}")
    print(f"Checkpoint file: {ckpt_path}")
    print()
    print(f"Checkpoint tensors: {len(ckpt_keys)}")
    print(f"Model state_dict keys: {len(model_keys)}")
    print()
    print("Prefix statistics (checkpoint keys):")
    print(f"  start with 'model.': {only_prefixed} / {len(ckpt_keys)}")
    print(f"  do NOT start with 'model.': {len(non_prefixed)}")
    if non_prefixed[:15]:
        print(f"  examples (non-prefixed): {non_prefixed[:15]}")
    print()
    print("Overlap (how many names match exactly):")
    print(f"  checkpoint ∩ model (as-is):              {len(inter_raw)}")
    print(f"  strip 'model.' from checkpoint ∩ model:   {len(inter_stripped)}")
    print(f"  checkpoint ∩ {{'model.' + k for k in model}}: {len(inter_prefixed)}")
    print()
    print(
        "Hypothesis (Hub-style): every tensor is stored as 'model.' + <module state_dict key> "
        "(Google export), while Transformers modules use unprefixed names."
    )
    print(f"  model keys with no 'model.<key>' in checkpoint: {len(missing_model_key_in_ckpt)}")
    if missing_model_key_in_ckpt[:12] and only_prefixed > 0:
        print(f"  examples: {missing_model_key_in_ckpt[:12]}")
    print(f"  all checkpoint keys use 'model.' prefix: {only_prefixed == len(ckpt_keys)}")
    print()
    if full_match_as_is:
        print(
            "Result: checkpoint key names match the model state_dict exactly (no rename needed).\n"
            "  Typical of: output from model.save_pretrained(...), or any file saved with the same\n"
            "  names as nn.Module.state_dict(). Hub raw weights often add a top-level 'model.' segment\n"
            "  instead; compare with the copy under the Hugging Face cache if you see UNEXPECTED/MISSING."
        )
    elif full_match_if_strip and not full_match_as_is:
        if only_prefixed == len(ckpt_keys):
            print(
                "Result: every checkpoint key is 'model.' + <state_dict key>. "
                "Strip that prefix to match this model (or load via a loader that remaps names)."
            )
        else:
            print(
                "Result: after stripping a leading 'model.' where present, names match the full state_dict."
            )
    else:
        leftover_model = model_keys - stripped
        leftover_ckpt = stripped - model_keys
        print("Result: names still differ after treating 'model.' as optional; check architecture or format.")
        print(f"  model keys not in strip(checkpoint): {len(leftover_model)}")
        print(f"  strip(checkpoint) keys not in model: {len(leftover_ckpt)}")
        if list(leftover_model)[:8]:
            print(f"  sample missing from ckpt (after strip): {list(leftover_model)[:8]}")
        if list(leftover_ckpt)[:8]:
            print(f"  sample extra after strip: {list(leftover_ckpt)[:8]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
