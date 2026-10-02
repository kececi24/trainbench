"""Checks for adapter weights lost or misnamed during FSDP export."""

import torch
from peft import get_peft_model_state_dict


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


def prepare_fsdp_adapter_state_dict(model, full_state_dict: dict, method: str) -> dict:
    """Return a PEFT-compatible full state dict, repairing FSDP wrapper segments.

    This repairs *names* only. Missing or zero learned tensors cannot be
    reconstructed from a checkpoint and must stop the export.
    """
    normalized = normalize_fsdp_wrapped_keys(full_state_dict)

    adapter_state = get_peft_model_state_dict(model, state_dict=normalized)
    if any("_fsdp_wrapped_module" in key for key in adapter_state):
        raise RuntimeError("FSDP wrapper names remain in extracted adapter keys")

    expected_b = sum(len(module._modules["lora_B"]) for module in model.modules()
                     if "lora_B" in module._modules)
    b_weights = [value for key, value in adapter_state.items()
                 if ".lora_B." in key]
    if expected_b == 0 or len(b_weights) != expected_b:
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
