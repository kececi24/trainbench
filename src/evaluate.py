"""
Module 3: Evaluation & Catastrophic Forgetting Analysis.

Roadmap Sections 32 & 33:
- Level 1: Cross-entropy loss and Perplexity (PPL = exp(loss)) on held-out target data.
- Level 3: General Capability Retention & Catastrophic Forgetting:
    Forgetting = Loss_after - Loss_before (or PPL_after - PPL_before)
    AdaptationGain = Q_target_after - Q_target_before
"""

import math
import os
import gc
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from data_accounting import InstructionDataset, collate_fn, instruction_split_specs


def load_finetuned_model(base_model_name: str, checkpoint_path: str | None,
                         dtype, device: str):
    """Load either a PEFT adapter, an FFT checkpoint, or the base model."""
    if checkpoint_path is None:
        return AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device)
        return PeftModel.from_pretrained(base_model, checkpoint_path)
    return AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype, device_map=device)


def evaluate_dataset(model, dataloader, device: str = "cuda") -> dict:
    """
    Evaluates cross-entropy loss and perplexity over a DataLoader.
    Disables Autograd and dropout for deterministic and memory-efficient inference.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # Count non-masked response tokens for exact weighted token loss
            active_tokens = (batch["labels"] != -100).sum().item()
            if active_tokens > 0:
                total_loss += loss.item() * active_tokens
                total_tokens += active_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "val_loss": round(avg_loss, 4),
        "val_ppl": round(perplexity, 2),
        "eval_tokens": total_tokens
    }


def evaluate_general_language(model, tokenizer, device: str = "cuda", num_samples: int = 40, max_length: int = 256) -> dict:
    """
    Evaluates general next-token language modeling ability (e.g. using wikitext or standard text)
    to measure baseline language capability retention.
    """
    print("Loading general corpus samples for forgetting assessment...")
    raw_general = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{num_samples}]")
    text_column = "text"

    # Filter out empty texts
    texts = [item[text_column] for item in raw_general if item.get(text_column, "").strip()]

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            num_tokens = input_ids.shape[1] - 1

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "general_loss": round(avg_loss, 4),
        "general_ppl": round(ppl, 2),
        "total_tokens": total_tokens
    }


def compute_forgetting_metrics(base_model_name: str, adapter_checkpoint_path: str = None) -> dict:
    """
    Compares baseline model vs. fine-tuned model on general language capability
    to compute catastrophic forgetting metrics (Section 32 & 33).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    print(f"\n[Forgetting Evaluation] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device)

    print("Evaluating Base Model general capability...")
    base_eval = evaluate_general_language(base_model, tokenizer, device=device)
    print(f"Base Model -> General Loss: {base_eval['general_loss']} | General PPL: {base_eval['general_ppl']}")

    if adapter_checkpoint_path is not None:
        print(f"Loading Fine-Tuned Checkpoint from: {adapter_checkpoint_path}")
        if os.path.isfile(os.path.join(adapter_checkpoint_path, "adapter_config.json")):
            ft_model = PeftModel.from_pretrained(base_model, adapter_checkpoint_path)
        else:
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ft_model = load_finetuned_model(base_model_name, adapter_checkpoint_path, dtype, device)
    else:
        ft_model = base_model

    print("Evaluating Fine-Tuned Model general capability...")
    ft_eval = evaluate_general_language(ft_model, tokenizer, device=device)
    print(f"Fine-Tuned -> General Loss: {ft_eval['general_loss']} | General PPL: {ft_eval['general_ppl']}")

    delta_loss = round(ft_eval["general_loss"] - base_eval["general_loss"], 4)
    delta_ppl = round(ft_eval["general_ppl"] - base_eval["general_ppl"], 2)

    results = {
        "base_loss": base_eval["general_loss"],
        "base_ppl": base_eval["general_ppl"],
        "ft_loss": ft_eval["general_loss"],
        "ft_ppl": ft_eval["general_ppl"],
        "delta_loss_forgetting": delta_loss,
        "delta_ppl_forgetting": delta_ppl
    }

    print("\n" + "=" * 55)
    print("Catastrophic Forgetting Summary (Sections 32-33):")
    print(f"  Delta General Loss: {delta_loss:+.4f} (Positive = Performance Degraded)")
    print(f"  Delta General PPL:  {delta_ppl:+.2f}")
    print("=" * 55)

    return results


def run_evaluation(base_model_name: str, adapter_checkpoint_path: str = None,
                   dataset_name: str = "yahma/alpaca-cleaned", output_json: str = None,
                   train_samples: int = 1000, val_samples: int = 50,
                   max_length: int = 256) -> dict:
    """
    Executes full evaluation:
    1. Held-out validation loss & perplexity on target instruction task.
    2. Catastrophic forgetting assessment on general language capability.
    Saves results to output_json if provided.
    """
    import json
    if adapter_checkpoint_path is not None and not os.path.isdir(adapter_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {adapter_checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    print(f"\n[Evaluation] Loading model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Target Validation Performance (Held-Out Split)
    print(f"[Evaluation] Evaluating held-out validation set from {dataset_name}...")
    model = None
    try:
        _, val_split = instruction_split_specs(train_samples, val_samples)
        raw_val = load_dataset(dataset_name, split=val_split)
        val_dataset = InstructionDataset(raw_val, tokenizer, max_length=max_length)
        if not val_dataset:
            raise ValueError("Validation split contains no usable responses")
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

        model = load_finetuned_model(base_model_name, adapter_checkpoint_path, dtype, device)

        target_metrics = evaluate_dataset(model, val_loader, device=device)
    except Exception as e:
        print(f"[Evaluation Warning] Could not evaluate target validation set: {e}")
        target_metrics = {"val_loss": None, "val_ppl": None, "eval_tokens": 0}
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 2. Catastrophic Forgetting
    print(f"[Evaluation] Measuring catastrophic forgetting...")
    try:
        forgetting_metrics = compute_forgetting_metrics(base_model_name, adapter_checkpoint_path)
    except Exception as e:
        print(f"[Evaluation Warning] Could not evaluate forgetting: {e}")
        forgetting_metrics = {"delta_loss_forgetting": None, "delta_ppl_forgetting": None}

    combined = {
        "model": base_model_name,
        "adapter": adapter_checkpoint_path,
        "target_val_loss": target_metrics.get("val_loss"),
        "target_val_ppl": target_metrics.get("val_ppl"),
        "eval_tokens": target_metrics.get("eval_tokens"),
        **forgetting_metrics
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"[Evaluation] Results successfully saved to: {output_json}")

    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate target performance and catastrophic forgetting")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="Path to fine-tuned checkpoint / adapter")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="Target evaluation dataset")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training rows before validation starts")
    parser.add_argument("--val-samples", type=int, default=50, help="Number of held-out validation rows")
    parser.add_argument("--max-length", type=int, default=256, help="Validation sequence length")

    args = parser.parse_args()
    run_evaluation(
        base_model_name=args.model,
        adapter_checkpoint_path=args.adapter,
        dataset_name=args.dataset,
        output_json=args.output_json,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        max_length=args.max_length,
    )
