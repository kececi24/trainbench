"""Two-GPU FSDP adapter export regression and fresh-process reload check.

Uses a random, tiny Qwen model; no Hub access or existing checkpoints needed.
Train under torchrun with ACCELERATE_USE_FSDP=true, then run --verify in a
separate process against the same --output directory. Repeat for --method dora.
"""

import argparse
from pathlib import Path
import sys

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors import safe_open
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint_integrity import (check_adapter_tensors, expected_lora_keys,
                                  prepare_fsdp_adapter_export)


def train_probe(args):
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=1, auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen2DecoderLayer"],
        use_orig_params=True, state_dict_type="FULL_STATE_DICT",
    )
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision="no")
    assert str(accelerator.distributed_type).endswith("FSDP")
    set_seed(42)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, attention_dropout=0.0,
                         use_cache=False)
    config._attn_implementation = "eager"
    base = Qwen2ForCausalLM(config).to(torch.bfloat16)
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=False)
        base.save_pretrained(args.output / "base")
    accelerator.wait_for_everyone()
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", use_dora=args.method == "dora",
    )).to(torch.bfloat16)
    keys = expected_lora_keys(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=accelerator.device)
    model.train()
    accelerator.backward(model(input_ids=inputs, labels=inputs).loss)
    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        reference = model(input_ids=inputs, labels=inputs)
        logits = reference.logits.detach().cpu()
        loss = reference.loss.detach().cpu()
    # This collective MUST precede removing any FSDP wrappers.
    full_state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        shallow = accelerator.unwrap_model(model)
        broken = get_peft_model_state_dict(shallow, state_dict=full_state)
        print(f"Before recursive unwrap: {len(broken)} adapter tensors", flush=True)
        print("Wrapped example:", next(n for n, _ in shallow.named_modules()
                                       if n.endswith("lora_A")), flush=True)
        print("Gathered example:", next(k for k in full_state if ".lora_A." in k), flush=True)
        # Final export only: this changes the module tree in place.
        clean, checked = prepare_fsdp_adapter_export(model, full_state, args.method, keys)
        clean.save_pretrained(args.output / "adapter", state_dict=checked)
        with safe_open(str(args.output / "adapter" / "adapter_model.safetensors"),
                       framework="pt", device="cpu") as saved:
            count = len(list(saved.keys()))
        torch.save({"inputs": inputs.cpu(), "logits": logits, "loss": loss},
                   args.output / "reference.pt")
        print(f"Exported {args.method}: {count} tensors, loss={loss.item():.9f}", flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


def verify_probe(args):
    reference = torch.load(args.output / "reference.pt", weights_only=True)
    base = Qwen2ForCausalLM.from_pretrained(args.output / "base", dtype=torch.bfloat16,
                                          attn_implementation="eager").cuda()
    model = PeftModel.from_pretrained(base, args.output / "adapter",
                                     autocast_adapter_dtype=False).eval()
    count = check_adapter_tensors(model, args.output / "adapter")
    inputs = reference["inputs"].cuda()
    with torch.no_grad():
        actual = model(input_ids=inputs, labels=inputs)
    logit_error = (actual.logits.cpu() - reference["logits"]).abs().max().item()
    loss_error = abs(actual.loss.item() - reference["loss"].item())
    torch.testing.assert_close(actual.logits.cpu(), reference["logits"], rtol=0, atol=0.002)
    assert loss_error <= 0.002, loss_error
    print(f"Fresh-process PASS: {count} tensors; max logit error={logit_error:.9g}; "
          f"loss error={loss_error:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lora", "dora"], default="lora")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify_probe(args) if args.verify else train_probe(args)
