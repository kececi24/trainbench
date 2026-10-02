`bash scripts/orchestrator.sh --profile`

Training logs include cumulative `estimated_flops` at every step, including validation checkpoints. The estimate counts dense matrix multiplications using the actual padded batch shape, trainable versus frozen weights, quadratic attention, and configured gradient checkpointing. It excludes normalization, activations, optimizer work, quantization overhead, and hardware-specific kernels. The log's `flops_estimate_method` identifies the estimate version. `--profile` writes a separate, short CUPTI trace for diagnostics; its FLOP counts are not used for the cumulative estimate. Older logs without `estimated_flops` need a new training run to populate compute-to-target.

To verify a LoRA or DoRA checkpoint after training exits, run from the repository root in a fresh process:

```bash
python src/verify_checkpoint.py --config configs/lora/llama2_7b.yaml
```

The verifier compares saved adapter tensors with the reloaded model, then evaluates the same held-out split and compares its loss and response-token count with the final training log. Use `--checkpoint` and `--log` if the artifacts are in another directory. The default maximum absolute loss difference is 0.02; adjust it with `--atol` when justified by the training precision. FSDP export removes only `_fsdp_wrapped_module` key segments when present, rejects key collisions or missing adapter tensors, and cannot reconstruct weights that were never saved.

If an **existing** adapter file contains `_fsdp_wrapped_module` segments in its tensor names, make a repaired copy, then verify that copy:

```bash
python src/repair_adapter_keys.py --source checkpoints/llama2_7b_lora --output checkpoints/llama2_7b_lora_repaired
python src/verify_checkpoint.py --config configs/lora/llama2_7b.yaml --checkpoint checkpoints/llama2_7b_lora_repaired
```

The repair command refuses an existing output directory and leaves the source untouched. If learned tensors are absent or zero, renaming cannot recover them; re-export from the trained process or retrain.
