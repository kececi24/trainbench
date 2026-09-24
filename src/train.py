import argparse
import os
import yaml
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from accelerate import Accelerator

from data_accounting import InstructionDataset, collate_fn, count_tokens
from metrics import EfficiencyTracker


def train(config, enable_profiling: bool = False):
    # 1. Initialize Hugging Face Accelerator (Supports Single-GPU, Multi-GPU DDP, and FSDP)
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    # Telemetry tracker only runs on main process
    tracker = None
    if is_main:
        tracker = EfficiencyTracker(run_name=f"{config.experiment.name}")

    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and getattr(config.model, "dtype", "bfloat16") == "bfloat16":
        dtype = torch.bfloat16

    # 2. QLoRA 4-bit Quantization Setup
    quant_config = None
    if config.method.name == "qlora" or getattr(config.method, "quantization", None) == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            if is_main:
                print("[Warning] bitsandbytes not installed; running standard LoRA without 4-bit quantization.")

    # 3. Model Loading: In multi-GPU / DDP, avoid hardcoded device_map="cuda"
    device_map = None
    if quant_config is not None:
        device_map = {"": accelerator.local_process_index}
    elif accelerator.num_processes == 1:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device_map
    )

    # 4. PEFT Wrapping (LoRA / DoRA / QLoRA)
    if config.method.name != "fft":
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=config.method.r,
            lora_alpha=config.method.lora_alpha,
            lora_dropout=config.method.lora_dropout,
            target_modules=config.method.target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_dora=getattr(config.method, "use_dora", False),
        )
        model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing if configured (reduces activation memory overhead)
    if getattr(config.training, "gradient_checkpointing", False) or getattr(config.model, "gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Move model to device if not mapped (skip for FSDP so parameters shard directly without VRAM spike)
    is_fsdp = str(accelerator.distributed_type).upper().endswith("FSDP")
    if device_map is None and quant_config is None and not is_fsdp:
        model.to(device)

    # 5. Tokenizer & Data Pipeline
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset_name = getattr(config.data, "dataset_name", getattr(config.data, "name", "yahma/alpaca-cleaned"))
    train_samples = getattr(config.data, "train_samples", 500)
    raw_data = load_dataset(dataset_name, split=f"train[:{train_samples}]")
    dataset = InstructionDataset(raw_data, tokenizer, max_length=config.data.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    warmup = getattr(config.training, "warmup_steps", getattr(config.training, "warming_steps", 5))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=config.training.max_steps)

    # 7. Accelerator Preparation (Prepares model, optimizer, dataloader for multi-GPU DDP / FSDP)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    model.train()
    if is_main and tracker is not None:
        tracker.start()

    # Configure Profiler on main process if requested
    prof = None
    if enable_profiling and is_main and torch.cuda.is_available():
        print(f"\n[Profiler] CUPTI Profiler ENABLED for run: {config.experiment.name}")
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_flops=True
        )
        prof.start()

    global_step = 0
    use_nvtx = torch.cuda.is_available()

    for batch_id, batch in enumerate(loader):
        if global_step >= config.training.max_steps:
            break

        if use_nvtx and is_main: torch.cuda.nvtx.range_push("Forward_Pass")
        outputs = model(**batch)
        loss = outputs.loss
        if use_nvtx and is_main: torch.cuda.nvtx.range_pop()

        if use_nvtx and is_main: torch.cuda.nvtx.range_push("Backward_Pass")
        # Accelerator handles distributed gradient AllReduce across GPUs
        accelerator.backward(loss)
        if use_nvtx and is_main: torch.cuda.nvtx.range_pop()

        if use_nvtx and is_main: torch.cuda.nvtx.range_push("Optimizer_Step")
        accelerator.clip_grad_norm_(model.parameters(), max_norm=getattr(config.training, "gradient_clip", 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if use_nvtx and is_main: torch.cuda.nvtx.range_pop()

        # Multi-GPU Token Accounting: gather real non-padding tokens across all ranks
        local_stats = count_tokens(batch)
        local_tokens = local_stats["non_pad_tokens"]
        if accelerator.num_processes > 1:
            tokens_tensor = torch.tensor([local_tokens], device=device)
            total_step_tokens = accelerator.gather(tokens_tensor).sum().item()
        else:
            total_step_tokens = local_tokens

        if is_main and tracker is not None:
            current_lr = float(scheduler.get_last_lr()[0])
            effective_batch = config.data.batch_size * accelerator.num_processes
            tracker.log_step(
                step=global_step,
                loss=loss.item(),
                lr=current_lr,
                batch_non_pad_tokens=total_step_tokens,
                batch_size=effective_batch
            )

        if prof is not None and is_main:
            prof.step()

        global_step += 1

    if prof is not None and is_main:
        prof.stop()
        trace_path = f"{config.tracking.log_dir}/{config.experiment.name}_trace.json.gz"
        prof.export_chrome_trace(trace_path)
        print(f"\n[Profiler] Trace successfully saved to: {trace_path}")
        print(f"[Profiler] Open https://ui.perfetto.dev to view the interactive CUDA kernel timeline.")
        print("\n--- Top 15 CUDA Kernels by Time (CUPTI) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Synchronize processes before saving
    accelerator.wait_for_everyone()

    # Save Model Checkpoint & Tokenizer
    checkpoint_dir = getattr(config.tracking, "checkpoint_dir", f"checkpoints/{config.experiment.name}")
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\n[Checkpoint] Saving weights to: {checkpoint_dir}...")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if str(accelerator.distributed_type).upper().endswith("FSDP"):
        state_dict = accelerator.get_state_dict(model)
        if is_main:
            unwrapped_model.save_pretrained(checkpoint_dir, state_dict=state_dict)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[Checkpoint] Weights and tokenizer successfully saved to: {checkpoint_dir}")
    else:
        if is_main:
            unwrapped_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[Checkpoint] Weights and tokenizer successfully saved to: {checkpoint_dir}")


def dict_to_namespace(d):
    """Recursively converts a dictionary into SimpleNamespace objects for dot-notation access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pipeline")
    parser.add_argument("--config", type=str, default="configs/lora/qwen2.5_0.5b.yaml", help="Location of the config file")
    parser.add_argument("--profile", action="store_true", help="Enable CUPTI/PyTorch profiler and export Chrome trace")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = dict_to_namespace(config_dict)
    train(config, enable_profiling=args.profile)