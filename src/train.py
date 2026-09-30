import argparse
import os
import time
import yaml
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_accounting import InstructionDataset, collate_fn, count_tokens, instruction_split_specs
from compute_accounting import FlopEstimator
from metrics import EfficiencyTracker


def evaluate_validation(model, loader, accelerator):
    """Return response-token-weighted validation loss across all ranks."""
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            active_tokens = (batch["labels"] != -100).sum()
            if active_tokens.item() == 0:
                continue
            loss = model(**batch).loss
            totals[0] += loss.detach().double() * active_tokens
            totals[1] += active_tokens
    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    if totals[1].item() == 0:
        raise ValueError("Validation split contains no response tokens")
    # Every rank evaluates the same held-out rows, avoiding padded duplicates
    # from a distributed evaluation sampler.
    eval_tokens = int(totals[1].item() / accelerator.num_processes)
    return (totals[0] / totals[1]).item(), eval_tokens


def train(config, enable_profiling: bool = False):
    # 1. Initialize Hugging Face Accelerator (Supports Single-GPU, Multi-GPU DDP, and FSDP)
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(config.experiment.seed, device_specific=True)

    dtype = torch.float32 if device.type == "cpu" else torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported() and getattr(config.model, "dtype", "bfloat16") == "bfloat16":
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
        except ImportError as exc:
            raise RuntimeError("QLoRA requires bitsandbytes and PEFT 4-bit support") from exc

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

    # Ensure uniform parameter dtype for FSDP sharding (casts PEFT float32 adapter params to model dtype)
    if quant_config is None:
        model.to(dtype)

    # Enable gradient checkpointing if configured (reduces activation memory overhead)
    gradient_checkpointing = (getattr(config.training, "gradient_checkpointing", False)
                              or getattr(config.model, "gradient_checkpointing", False))
    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Move model to device if not mapped (skip for FSDP so parameters shard directly without VRAM spike)
    is_fsdp = str(accelerator.distributed_type).upper().endswith("FSDP")
    if device_map is None and quant_config is None and not is_fsdp:
        model.to(device)

    flop_estimator = FlopEstimator.from_model(model, gradient_checkpointing)
    tracker = None
    if is_main:
        tracker = EfficiencyTracker(
            run_name=config.experiment.name,
            log_dir=getattr(config.tracking, "log_dir", "logs"),
            flops_estimate_method=flop_estimator.method,
        )

    # 5. Tokenizer & Data Pipeline
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset_name = getattr(config.data, "dataset_name", getattr(config.data, "name", "yahma/alpaca-cleaned"))
    train_samples = getattr(config.data, "train_samples", 500)
    val_samples = getattr(config.data, "val_samples", 50)
    train_split, val_split = instruction_split_specs(train_samples, val_samples)
    raw_data = load_dataset(dataset_name, split=train_split)
    raw_val = load_dataset(dataset_name, split=val_split)
    dataset = InstructionDataset(raw_data, tokenizer, max_length=config.data.max_length)
    val_dataset = InstructionDataset(raw_val, tokenizer, max_length=config.data.max_length)
    if not dataset or not val_dataset:
        raise ValueError("Training and validation splits must both contain usable responses")
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    warmup = getattr(config.training, "warmup_steps", getattr(config.training, "warming_steps", 5))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=config.training.max_steps)

    # 7. Accelerator Preparation (Prepares model, optimizer, dataloader for multi-GPU DDP / FSDP)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    if len(loader) == 0:
        raise ValueError("Prepared training loader is empty")

    model.train()
    if is_main and tracker is not None:
        tracker.start()

    if is_main and device.type == "cuda":
        torch.cuda.synchronize(device)
    baseline_started = time.perf_counter() if is_main else 0.0
    baseline_loss, baseline_tokens = evaluate_validation(model, val_loader, accelerator)
    if is_main and tracker is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tracker.exclude_time(time.perf_counter() - baseline_started)
        tracker.log_step(
            step=0, loss=None, lr=float(scheduler.get_last_lr()[0]),
            batch_non_pad_tokens=0, batch_size=0,
            val_loss=baseline_loss, eval_tokens=baseline_tokens,
            batch_estimated_flops=0,
        )

    # Configure Profiler on main process if requested
    prof = None
    if enable_profiling and is_main and device.type == "cuda":
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
    use_nvtx = device.type == "cuda"
    max_steps = config.training.max_steps
    eval_every = getattr(config.training, "eval_every_steps", max(1, max_steps // 10))
    if eval_every < 1:
        raise ValueError("eval_every_steps must be positive")

    while global_step < max_steps:
        for batch in loader:
            if global_step >= max_steps:
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

            # Count actual examples and response/non-padding tokens across all ranks.
            local_stats = count_tokens(batch)
            step_stats = torch.tensor([
                loss.item() * local_stats["trained_tokens"],
                local_stats["trained_tokens"],
                local_stats["non_pad_tokens"],
                batch["input_ids"].shape[0],
                local_stats["total_elements"],
                batch["input_ids"].shape[0] * batch["input_ids"].shape[1] ** 2,
            ], device=device, dtype=torch.float64)
            step_stats = accelerator.reduce(step_stats, reduction="sum")
            train_loss = (step_stats[0] / step_stats[1]).item()
            estimated_step_flops = flop_estimator.estimate_step(
                padded_token_slots=int(step_stats[4].item()),
                attention_positions=int(step_stats[5].item()),
            )

            val_loss = None
            eval_tokens = 0
            if (global_step + 1) % eval_every == 0 or global_step + 1 == max_steps:
                if is_main and device.type == "cuda":
                    torch.cuda.synchronize(device)
                eval_started = time.perf_counter() if is_main else 0.0
                val_loss, eval_tokens = evaluate_validation(model, val_loader, accelerator)
                if is_main and tracker is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tracker.exclude_time(time.perf_counter() - eval_started)

            if is_main and tracker is not None:
                current_lr = float(scheduler.get_last_lr()[0])
                tracker.log_step(
                    step=global_step + 1,
                    loss=train_loss,
                    lr=current_lr,
                    batch_non_pad_tokens=int(step_stats[2].item()),
                    batch_trained_tokens=int(step_stats[1].item()),
                    batch_size=int(step_stats[3].item()),
                    val_loss=val_loss,
                    eval_tokens=eval_tokens,
                    batch_estimated_flops=estimated_step_flops,
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
