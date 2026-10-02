"""Copy an adapter checkpoint while removing erroneous FSDP key segments.

The source is never modified. This only repairs tensor names, not missing or
untrained weights; verify the output with verify_checkpoint.py afterward.
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from checkpoint_integrity import normalize_fsdp_wrapped_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("--output must be outside the source checkpoint")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    weights_path = source / "adapter_model.safetensors"
    if not (source / "adapter_config.json").is_file() or not weights_path.is_file():
        parser.error("--source must contain adapter_config.json and adapter_model.safetensors")

    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        original = {key: weights.get_tensor(key) for key in weights.keys()}
        metadata = weights.metadata()
    if not any("_fsdp_wrapped_module" in key.split(".") for key in original):
        parser.error("No FSDP wrapper key segments found; nothing to repair")

    repaired = normalize_fsdp_wrapped_keys(original)
    b_weights = [value for key, value in repaired.items() if ".lora_B." in key]
    if not b_weights:
        raise RuntimeError("Checkpoint has no LoRA B weights; renaming cannot repair it")
    if not any(torch.count_nonzero(value).item() for value in b_weights):
        print("Warning: every LoRA B tensor is zero; renaming may not restore a learned adapter")

    shutil.copytree(source, output)
    save_file({key: value.contiguous() for key, value in repaired.items()},
              str(output / "adapter_model.safetensors"), metadata=metadata)
    print(f"Wrote repaired copy to {output}; verify its reload and validation loss next")


if __name__ == "__main__":
    main()
