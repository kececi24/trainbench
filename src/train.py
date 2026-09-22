import argparse
import yaml
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

from data_accounting import InstructionDataset, collate_fn, count_tokens
from metrics import EfficiencyTracker


def train(config, enable_profiling: bool = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = EfficiencyTracker(run_name=f"{config.experiment.name}")

    dtype = torch.float16
    if torch.cuda.is_bf16_supported() and config.model.dtype == "bfloat16":
        dtype = torch.bfloat16

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
            print("[Warning] bitsandbytes not installed; running standard LoRA without 4-bit quantization.")

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device
    )

    if config.method.name != "fft":
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=config.method.r,
            lora_alpha=config.method.lora_alpha,
            lora_dropout=config.method.lora_dropout,
            target_modules=config.method.target_modules,
            task_type=TaskType.CAUSAL_LM,
            use_dora=getattr(config.method, "use_dora", False),
        ))

    if quant_config is None:
        model.to(device)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    warmup = getattr(config.training, "warmup_steps", getattr(config.training, "warming_steps", 5))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=config.training.max_steps)

    model.train()
    tracker.start()

    # Configure CUPTI Profiler if requested
    prof = None
    if enable_profiling and torch.cuda.is_available():
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

        if use_nvtx: torch.cuda.nvtx.range_push("Data_Transfer")
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_nvtx: torch.cuda.nvtx.range_pop()

        if use_nvtx: torch.cuda.nvtx.range_push("Forward_Pass")
        outputs = model(**batch)
        loss = outputs.loss
        if use_nvtx: torch.cuda.nvtx.range_pop()

        if use_nvtx: torch.cuda.nvtx.range_push("Backward_Pass")
        loss.backward()
        if use_nvtx: torch.cuda.nvtx.range_pop()

        if use_nvtx: torch.cuda.nvtx.range_push("Optimizer_Step")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(config.training, "gradient_clip", 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if use_nvtx: torch.cuda.nvtx.range_pop()

        current_lr = float(scheduler.get_last_lr()[0])
        stats = count_tokens(batch)
        tracker.log_step(step=global_step, loss=loss.item(), lr=current_lr, batch_non_pad_tokens=stats["non_pad_tokens"], batch_size=config.data.batch_size)

        if prof is not None:
            prof.step()

        global_step += 1

    if prof is not None:
        prof.stop()
        trace_path = f"{config.tracking.log_dir}/{config.experiment.name}_trace.json.gz"
        prof.export_chrome_trace(trace_path)
        print(f"\n[Profiler] Trace successfully saved to: {trace_path}")
        print(f"[Profiler] Open https://ui.perfetto.dev to view the interactive CUDA kernel timeline.")
        print("\n--- Top 15 CUDA Kernels by Time (CUPTI) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def dict_to_namespace(d):
    """Recursively converts a dictionary into SimpleNamespace objects for dot-notation access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pipeline")
    parser.add_argument("--config", type=str, default="configs/lora/base.yaml", help="Location of the config file")
    parser.add_argument("--profile", action="store_true", help="Enable CUPTI/PyTorch profiler and export Chrome trace")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = dict_to_namespace(config_dict)
    train(config, enable_profiling=args.profile)


     