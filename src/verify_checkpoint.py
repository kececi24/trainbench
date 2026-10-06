"""Verify a saved LoRA/DoRA adapter in a fresh process.

Run from the repository root after training has exited. This uses the same
validation split, tokenizer, batch size, masking, and dtype as train.py.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_accounting import InstructionDataset, collate_fn, instruction_split_specs
from evaluate import load_finetuned_model
from checkpoint_integrity import check_adapter_tensors


def last_validation_record(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for record in reversed(records):
        if record.get("val_loss") is not None and math.isfinite(record["val_loss"]):
            return record
    raise ValueError(f"No finite validation loss in {path}")


def validation_loss(model, loader, device: str) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            active_tokens = int((batch["labels"] != -100).sum().item())
            if active_tokens:
                total_loss += model(**batch).loss.item() * active_tokens
                total_tokens += active_tokens
    if not total_tokens:
        raise ValueError("Validation split contains no response tokens")
    return total_loss / total_tokens, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--atol", type=float, default=0.02,
                        help="Maximum absolute validation-loss difference (default: 0.02)")
    args = parser.parse_args()
    if args.atol < 0:
        parser.error("--atol must be nonnegative")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["method"]["name"]
    if method not in {"lora", "dora"}:
        parser.error("This verifier is for unquantized LoRA and DoRA checkpoints")

    name = config["experiment"]["name"]
    tracking = config.get("tracking", {})
    checkpoint = args.checkpoint or Path(tracking.get("checkpoint_dir", f"checkpoints/{name}"))
    log_path = args.log or Path(tracking.get("log_dir", "logs")) / f"{name}.jsonl"
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config in {checkpoint}")
    record = last_validation_record(log_path)
    if record["step"] != config["training"]["max_steps"]:
        raise ValueError("Log does not have final-step validation; cannot compare final model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported() and config["model"].get("dtype", "bfloat16") == "bfloat16":
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = config["data"]
    _, val_split = instruction_split_specs(data.get("train_samples", 500),
                                           data.get("val_samples", 50))
    raw_val = load_dataset(data.get("dataset_name", data.get("name", "yahma/alpaca-cleaned")),
                           split=val_split)
    dataset = InstructionDataset(raw_val, tokenizer, max_length=data["max_length"])
    if not dataset:
        raise ValueError("Validation split contains no usable responses")
    loader = DataLoader(dataset, batch_size=data["batch_size"], shuffle=False,
                        collate_fn=lambda rows: collate_fn(rows, tokenizer.pad_token_id))

    model = load_finetuned_model(config["model"]["name"], str(checkpoint), dtype, device)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tensor_count = check_adapter_tensors(model, checkpoint)
    loss, tokens = validation_loss(model, loader, device)
    expected_loss = float(record["val_loss"])
    expected_tokens = int(record["eval_tokens"])
    difference = abs(loss - expected_loss)
    print(f"Adapter tensors verified: {tensor_count}")
    print(f"Validation tokens: {tokens} (training log: {expected_tokens})")
    print(f"Reloaded loss: {loss:.6f}; training loss: {expected_loss:.6f}; difference: {difference:.6f}")
    if tokens != expected_tokens or difference > args.atol:
        raise RuntimeError("Checkpoint round trip failed")
    print("Checkpoint round trip passed")


if __name__ == "__main__":
    main()
