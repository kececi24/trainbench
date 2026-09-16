"""
Module 3: Evaluation & Catastrophic Forgetting Analysis.

Roadmap Sections 32 & 33:
- Level 1: Cross-entropy loss and Perplexity (PPL = exp(loss)) on held-out target data.
- Level 3: General Capability Retention & Catastrophic Forgetting:
    Forgetting = Loss_after - Loss_before (or PPL_after - PPL_before)
    AdaptationGain = Q_target_after - Q_target_before
"""

import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from data_accounting import InstructionDataset, collate_fn


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
    try:
        raw_general = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{num_samples}]")
        text_column = "text"
    except Exception:
        # Fallback to a general subset if network/dataset has issues
        raw_general = load_dataset("yahma/alpaca-cleaned", split=f"train[1000:1000+{num_samples}]")
        text_column = "instruction"

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
            num_tokens = input_ids.shape[1]

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
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"\n[Forgetting Evaluation] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=device)

    print("Evaluating Base Model general capability...")
    base_eval = evaluate_general_language(base_model, tokenizer, device=device)
    print(f"Base Model -> General Loss: {base_eval['general_loss']} | General PPL: {base_eval['general_ppl']}")

    if adapter_checkpoint_path is not None:
        print(f"Loading Fine-Tuned Adapter from: {adapter_checkpoint_path}")
        ft_model = PeftModel.from_pretrained(base_model, adapter_checkpoint_path)
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


if __name__ == "__main__":
    # Test evaluation module on base model
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print("--- Running Test Evaluation on Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_val = load_dataset("yahma/alpaca-cleaned", split="train[100:120]")
    val_dataset = InstructionDataset(raw_val, tokenizer, max_length=256)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
    val_results = evaluate_dataset(model, val_loader, device=device)
    print(f"Validation Target Loss: {val_results['val_loss']} | PPL: {val_results['val_ppl']} across {val_results['eval_tokens']} tokens")
