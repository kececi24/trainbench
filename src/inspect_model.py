"""
Step 1: Inspecting a Pretrained Decoder LLM in PyTorch.

Theory Concepts Covered:
1. Model Architecture & Layer hierarchy (Embedding -> Transformer Blocks -> LM Head).
2. Weight Tensors, Shapes, and dtypes (FP32 vs BF16).
3. Parameter Counting: Total parameters vs. Trainable parameters (requires_grad).
4. Forward pass & Causal Language Modeling loss calculation:
   CrossEntropy(logits[t], labels[t+1]).
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def inspect_model(model_name: str = "Qwen/Qwen2.5-0.5B"):
    print(f"=" * 60)
    print(f"Loading model and tokenizer: {model_name}")
    print(f"=" * 60)

    # 1. Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # 2. Tokenizer: Maps human text strings <-> discrete integers (tokens)
    print("\n--- 1. Tokenizer Inspection ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Vocabulary size (V): {tokenizer.vocab_size:,}")
    print(f"BOS (Beginning of sequence) token ID: {tokenizer.bos_token_id}")
    print(f"EOS (End of sequence) token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    if tokenizer.pad_token is None:
        # Many decoder models don't have a dedicated pad token; we use eos_token
        tokenizer.pad_token = tokenizer.eos_token
        print(f"-> Assigned pad_token = eos_token ({tokenizer.pad_token})")

    # 3. Model: Load the weights
    # We use torch_dtype=torch.bfloat16 to save memory and compute (standard for modern LLMs)
    print("\n--- 2. Loading Weights (bfloat16) ---")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    print(f"Model loaded successfully with dtype: {dtype}")

    if device == "cuda":
        print(f"VRAM after loading model: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # 4. Parameter Inspection: Shapes and requires_grad
    print("\n--- 3. Parameter Breakdown ---")
    total_params = 0
    trainable_params = 0

    print(f"{'Layer / Module Name':<50} {'Shape':<25} {'Dtype':<15} {'Requires Grad'}")
    print("-" * 105)

    sample_printed = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

        # Print the first few layers and the last layer to see the structure
        if sample_printed < 8 or "lm_head" in name:
            shape_str = str(list(param.shape))
            print(f"{name:<50} {shape_str:<25} {str(param.dtype):<15} {param.requires_grad}")
            if sample_printed == 7:
                print(f"{'... (intermediate transformer layers omitted for brevity) ...':<50}")
            sample_printed += 1

    print("-" * 105)
    print(f"Total Parameters (P_total):     {total_params:,}")
    print(f"Trainable Parameters (P_train): {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 5. A Simple Forward Pass & Loss Computation
    print("\n--- 4. Forward Pass & Causal LM Loss Demo ---")
    prompt_text = "Benchmarking fine-tuning efficiency is important because"
    
    # Encode text into input_ids tensor: shape [Batch, Sequence_Length]
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Input text: \"{prompt_text}\"")
    print(f"Token IDs tensor shape: {input_ids.shape} (Batch size = {input_ids.shape[0]}, Sequence length = {input_ids.shape[1]})")
    print(f"Token IDs: {input_ids.tolist()[0]}")

    # In Causal Language Modeling, labels are the same as input_ids.
    # The Hugging Face model internally shifts logits and labels by 1 position:
    # logits[:, :-1, :] vs labels[:, 1:]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )

    logits = outputs.logits
    loss = outputs.loss

    print(f"\nLogits shape [B, L, V]: {list(logits.shape)}")
    print(f"Theoretical interpretation: For each of the {input_ids.shape[1]} tokens, a probability distribution over {tokenizer.vocab_size:,} vocabulary items.")
    print(f"Cross Entropy Loss: {loss.item():.4f}")
    print(f"Perplexity (exp(Loss)): {torch.exp(loss).item():.2f}")

    print("\n" + "=" * 60)
    print("Step 1 inspection complete!")
    print("=" * 60)

if __name__ == "__main__":
    inspect_model()
