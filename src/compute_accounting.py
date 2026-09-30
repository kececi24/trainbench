"""Consistent per-step training FLOP estimates for benchmark curves.

The estimate counts dense matrix multiplications and quadratic attention. It
does not include normalization, activations, optimizer work, quantization
overhead, or hardware-specific kernels. It is independent of the optional
PyTorch profiler, whose scheduled trace covers only selected steps.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlopEstimator:
    linear_parameters: int
    trainable_linear_parameters: int
    attention_layers: int
    hidden_size: int
    gradient_checkpointing: bool = False

    @classmethod
    def from_model(cls, model, gradient_checkpointing: bool = False) -> "FlopEstimator":
        linear_parameters = 0
        trainable_linear_parameters = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                continue
            # Only direct weights count. A PEFT wrapper's base layer and A/B
            # matrices are visited individually, so no parameter is counted twice.
            weight = module._parameters.get("weight")
            if weight is None:
                continue
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                # Quantized weights may be packed, but their matmul dimensions
                # still represent the original dense operation.
                size = module.in_features * module.out_features
            elif weight.ndim == 2:
                size = weight.numel()
            else:
                continue
            linear_parameters += size
            if weight.requires_grad:
                trainable_linear_parameters += size

        if linear_parameters == 0:
            raise ValueError("Cannot estimate FLOPs: model has no matrix layers")

        config = model.config
        layers = getattr(config, "num_hidden_layers", None)
        if layers is None:
            layers = getattr(config, "n_layer", None)
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(config, "n_embd", None)
        return cls(
            linear_parameters=linear_parameters,
            trainable_linear_parameters=trainable_linear_parameters,
            attention_layers=int(layers or 0),
            hidden_size=int(hidden or 0),
            gradient_checkpointing=gradient_checkpointing,
        )

    @property
    def method(self) -> str:
        return ("dense_matmul_attention_v1" if self.attention_layers and self.hidden_size
                else "dense_matmul_only_v1")

    def estimate_step(self, padded_token_slots: int, attention_positions: int) -> int:
        """Estimate one optimizer step from globally summed batch shapes.

        Each dense weight contributes 2 FLOPs per token to the forward pass
        and 2 to input gradients. Trainable weights add 2 for weight gradients.
        QK and AV attention matmuls contribute about 12*layers*hidden*B*S^2
        for forward and backward. Checkpointing adds one forward recomputation.
        """
        if padded_token_slots < 0 or attention_positions < 0:
            raise ValueError("Batch dimensions must be nonnegative")
        linear = (4 * self.linear_parameters + 2 * self.trainable_linear_parameters)
        attention = 12 * self.attention_layers * self.hidden_size
        if self.gradient_checkpointing:
            linear += 2 * self.linear_parameters
            attention += 4 * self.attention_layers * self.hidden_size
        return linear * padded_token_slots + attention * attention_positions
