import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = TaskType.CAUSAL_LM,
    use_dora = False
)

def inspect_peft_adapters(model_name: str = "Qwen/Qwen2.5-0.5B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        device_map = device
    )

    peft_model = get_peft_model(model, lora_config)

    # LoRA parameter count
    print("\n--- LoRA (use_dora = False) ---")
    peft_model.print_trainable_parameters()

    # Shape of A B matrices
    print("\n--- Trainable Adapter Tensors in Layer 0 ---")
    for name, param in peft_model.named_parameters():
        if param.requires_grad and "layers.0" in name:
            print(f"{name:<55} Shape: {list(param.shape)}")

if __name__ == "__main__":
    inspect_peft_adapters()