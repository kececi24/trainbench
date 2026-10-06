"""Checks for adapter weights lost or misnamed during FSDP export."""

from pathlib import Path

import torch
from accelerate.utils import extract_model_from_parallel
from peft import get_peft_model_state_dict
from safetensors import safe_open


def check_adapter_tensors(model, checkpoint: Path | str) -> int:
    """Reject missing adapter tensors rather than evaluating fresh initialization."""
    weights_path = Path(checkpoint) / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    loaded = get_peft_model_state_dict(model)
    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        saved_keys = set(weights.keys())
        loaded_keys = set(loaded)
        if not saved_keys:
            raise RuntimeError(
                f"Empty adapter checkpoint: {weights_path}. No learned tensors were saved; "
                "renaming cannot recover them. Re-export from the trained model or retrain."
            )
        if saved_keys != loaded_keys:
            missing = sorted(loaded_keys - saved_keys)
            unexpected = sorted(saved_keys - loaded_keys)
            raise RuntimeError(
                f"Adapter key mismatch: file has {len(saved_keys)} keys, model expects "
                f"{len(loaded_keys)}; missing={len(missing)} {missing[:5]}, "
                f"unexpected={len(unexpected)} {unexpected[:5]}. "
                "If unexpected=0, renaming keys cannot restore the missing tensors."
            )
        for key in saved_keys:
            actual = loaded[key].detach().cpu()
            expected = weights.get_tensor(key).to(dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"Loaded adapter tensor differs from checkpoint: {key}")
    return len(saved_keys)


def expected_lora_keys(model) -> set[str]:
    """Capture adapter tensor names before FSDP changes the module tree."""
    expected = set()
    for name, module in model.named_modules():
        lora_a = module._modules.get("lora_A")
        lora_b = module._modules.get("lora_B")
        if lora_a is None or lora_b is None:
            continue
        if set(lora_a) != set(lora_b):
            raise RuntimeError(f"LoRA A/B adapters differ in module {name}")
        for adapter_name in lora_a:
            if adapter_name != "default":
                raise RuntimeError(f"Unexpected adapter name {adapter_name!r} in {name}")
            expected.add(f"{name}.lora_A.weight")
            expected.add(f"{name}.lora_B.weight")
    if not expected:
        raise RuntimeError("PEFT model contains no LoRA A/B modules")
    return expected


def normalize_fsdp_wrapped_keys(state_dict: dict) -> dict:
    """Remove exact FSDP wrapper segments without silently merging keys."""
    normalized = {}
    for key, value in state_dict.items():
        clean_key = ".".join(part for part in key.split(".")
                             if part != "_fsdp_wrapped_module")
        if clean_key in normalized:
            raise RuntimeError(f"FSDP key collision after normalization: {clean_key}")
        normalized[clean_key] = value
    return normalized


def prepare_fsdp_adapter_export(model, full_state_dict: dict, method: str,
                                expected_keys: set[str]):
    """Align the module tree with gathered keys for final PEFT export.

    All ranks must finish gathering full_state_dict before this call. Recursive
    unwrapping mutates the tree; do not run further FSDP forwards afterward.
    """
    unwrapped = extract_model_from_parallel(model, recursive=True)
    state_dict = prepare_fsdp_adapter_state_dict(
        unwrapped, full_state_dict, method, expected_keys
    )
    return unwrapped, state_dict


def prepare_fsdp_adapter_state_dict(model, full_state_dict: dict, method: str,
                                    expected_keys: set[str]) -> dict:
    """Return a PEFT-compatible full state dict, repairing FSDP wrapper segments.

    The model must already be recursively unwrapped after gathering its full
    state, so PEFT's structural filtering sees canonical module names.
    This repairs *names* only. Missing or zero learned tensors cannot be
    reconstructed from a checkpoint and must stop the export.
    """
    normalized = normalize_fsdp_wrapped_keys(full_state_dict)

    adapter_state = get_peft_model_state_dict(model, state_dict=normalized)
    if any("_fsdp_wrapped_module" in key for key in adapter_state):
        raise RuntimeError("FSDP wrapper names remain in extracted adapter keys")

    missing = sorted(expected_keys - set(adapter_state))
    if missing:
        raise RuntimeError(
            f"FSDP export is missing {len(missing)} expected LoRA tensors: {missing[:5]}"
        )

    b_weights = [value for key, value in adapter_state.items()
                 if ".lora_B." in key]
    expected_b = sum(".lora_B." in key for key in expected_keys)
    if len(b_weights) != expected_b:
        raise RuntimeError(
            f"FSDP export has {len(b_weights)} LoRA B weights; expected {expected_b}"
        )
    if method == "lora" and not any(torch.count_nonzero(value).item()
                                    for value in b_weights):
        raise RuntimeError("Exported LoRA B weights are all zero")

    if method == "dora":
        magnitudes = [key for key in adapter_state if "lora_magnitude_vector" in key]
        if len(magnitudes) != expected_b:
            raise RuntimeError(
                f"FSDP export has {len(magnitudes)} DoRA magnitudes; expected {expected_b}"
            )

    return normalized
