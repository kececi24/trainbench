"""
Module 4: Weight-Space & Representation SVD Analysis.

Roadmap Section 26:
- For FFT:  Delta W = W_FT - W_0
- For LoRA/DoRA: Delta W = W_merged - W_0 (includes DoRA magnitude)
- Computes:
    - Frobenius Norm: ||Delta W||_F
    - Spectral Norm:  sigma_max = S_0
    - Effective Rank: exp(-sum(p_i * ln(p_i))) where p_i = S_i / sum(S_i)
    - Relative Update Magnitude: ||Delta W||_F / ||W_0||_F
    - Cosine Similarity of updates across methods
"""

import math
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from checkpoint_integrity import check_adapter_tensors


def compute_matrix_svd_metrics(delta_w: torch.Tensor, w0: torch.Tensor = None) -> dict:
    """
    Computes spectral and rank metrics for a single 2D weight update matrix Delta W.
    delta_w: 2D PyTorch Tensor [d_out, d_in]
    w0: Optional base weight matrix [d_out, d_in]
    """
    delta_w = delta_w.float().cpu()

    # Frobenius norm
    frob_norm = torch.norm(delta_w, p="fro").item()

    # Singular Value Decomposition: delta_w = U * S * Vh
    # S contains singular values in descending order: S_0 >= S_1 >= ... >= S_{min(m, n)-1}
    singular_values = torch.linalg.svdvals(delta_w)

    # Spectral norm is the largest singular value
    spectral_norm = singular_values[0].item() if len(singular_values) > 0 else 0.0

    # Effective Rank: exp(Shannon entropy of normalized singular values)
    # Measures the effective dimensionality occupied by the update
    s_sum = singular_values.sum().item()
    if s_sum > 1e-12:
        p = (singular_values / s_sum).clamp(min=1e-12)
        entropy = -torch.sum(p * torch.log(p)).item()
        effective_rank = math.exp(entropy)
    else:
        effective_rank = 0.0

    metrics = {
        "frobenius_norm": round(frob_norm, 4),
        "spectral_norm": round(spectral_norm, 4),
        "effective_rank": round(effective_rank, 2),
        "top_5_singular_values": [round(s.item(), 4) for s in singular_values[:5]],
        "total_singular_values": len(singular_values)
    }

    # Relative update magnitude: ||Delta W|| / ||W_0||
    if w0 is not None:
        w0_frob = torch.norm(w0.float().cpu(), p="fro").item()
        rel_update = frob_norm / w0_frob if w0_frob > 0 else 0.0
        metrics["relative_update_magnitude"] = round(rel_update, 6)

    return metrics


def compute_update_cosine_similarity(delta_w1: torch.Tensor, delta_w2: torch.Tensor) -> float:
    """
    Calculates the Frobenius inner product cosine similarity between two update matrices:
    cos_sim = <Delta W_1, Delta W_2>_F / (||Delta W_1||_F * ||Delta W_2||_F)
    Answers Section 26: Is LoRA learning essentially the same update direction as FFT?
    """
    vec1 = delta_w1.flatten().float().cpu()
    vec2 = delta_w2.flatten().float().cpu()

    norm1 = torch.norm(vec1, p=2)
    norm2 = torch.norm(vec2, p=2)

    if norm1 > 1e-9 and norm2 > 1e-9:
        cos_sim = (torch.dot(vec1, vec2) / (norm1 * norm2)).item()
    else:
        cos_sim = 0.0

    return round(cos_sim, 4)


def extract_adapter_update(peft_model, target_module_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the effective update through PEFT's merge, including DoRA magnitude."""
    for name, module in peft_model.named_modules():
        if name == target_module_name and hasattr(module, "lora_A"):
            if module.merged:
                raise ValueError(f"Adapter module is already merged: {name}")
            base_layer = module.get_base_layer()
            w0 = base_layer.weight.detach().float().cpu().clone()
            try:
                module.merge()
                delta_w = base_layer.weight.detach().float().cpu() - w0
            finally:
                if module.merged:
                    module.unmerge()
            return delta_w, w0
    raise ValueError(f"Could not find adapter weights for target module: {target_module_name}")


def extract_lora_delta_w(peft_model, target_module_name: str = "layers.0.self_attn.q_proj") -> torch.Tensor:
    """Compatibility wrapper returning the effective LoRA or DoRA update."""
    delta_w, _ = extract_adapter_update(peft_model, target_module_name)
    return delta_w


def analyze_model_updates(base_model_name: str, adapter_path: str = None, output_json: str = None) -> dict:
    """
    Performs comprehensive SVD and rank analysis on adapted linear layers.
    Dynamically discovers adapted LoRA modules across any model architecture.
    """
    import os
    import json

    if adapter_path is not None and not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint directory not found: {adapter_path}")

    print(f"\n[Weight Analysis] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32, device_map="cpu")

    if adapter_path is not None:
        print(f"Loading adapter: {adapter_path}")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        check_adapter_tensors(peft_model, adapter_path)
    else:
        # If testing without an external checkpoint, create an in-memory test adapter
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(base_model, config)

    # Dynamically find adapted modules
    adapted_modules = []
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            adapted_modules.append(name)

    if not adapted_modules:
        print("[Weight Analysis Warning] No LoRA adapter modules found to analyze.")
        return {}

    # Sample key adapted modules (e.g. first layer and last layer modules to keep output clean)
    target_modules = list(dict.fromkeys(adapted_modules[:2] + adapted_modules[-2:]))

    report = {}
    print("\n" + "=" * 85)
    print(f"{'Module Name':<40} {'Rank':<8} {'Eff. Rank':<12} {'Frob Norm':<12} {'Spectral':<10}")
    print("-" * 85)

    for target in target_modules:
        try:
            delta_w, w0 = extract_adapter_update(peft_model, target)

            metrics = compute_matrix_svd_metrics(delta_w, w0)
            report[target] = metrics

            module = dict(peft_model.named_modules())[target]
            nominal_rank = module.r["default"]
            print(f"{target[-38:]:<40} {nominal_rank:<8} {metrics['effective_rank']:<12} {metrics['frobenius_norm']:<12} {metrics['spectral_norm']:<10}")
        except Exception as e:
            print(f"[Weight Analysis Warning] Could not analyze {target}: {e}")

    print("=" * 85)

    if output_json:
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Weight Analysis] Results successfully saved to: {output_json}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weight SVD and Rank Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None, help="Path to adapter checkpoint")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save SVD results JSON")

    args = parser.parse_args()
    analyze_model_updates(
        base_model_name=args.model,
        adapter_path=args.adapter,
        output_json=args.output_json
    )
