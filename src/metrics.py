import os
import time
import json
import torch

class EfficiencyTracker():

    def __init__(self, run_name: str, log_dir: str = "logs",
                 flops_estimate_method: str | None = None) -> None:
        self.run_name = run_name
        self.flops_estimate_method = flops_estimate_method

        os.makedirs(log_dir, exist_ok=True)
        self.logfile_path = os.path.join(log_dir, f"{run_name}.jsonl")
        # A run name identifies one run. Reusing it must not concatenate two curves.
        with open(self.logfile_path, "w", encoding="utf-8"):
            pass

        self.start_time: float = 0.0
        self.total_tokens_seen = 0
        self.trained_tokens_seen = 0
        self.total_examples_seen = 0
        self.total_estimated_flops = 0
        self.history = []
        self.excluded_seconds = 0.0

    def start(self):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()

    def exclude_time(self, seconds: float):
        """Keep validation time out of training throughput and time-to-target."""
        self.excluded_seconds += seconds

    def log_step(self, step: int, loss: float | None, lr: float, batch_non_pad_tokens: int,
                 batch_size: int, val_loss: float | None = None,
                 eval_tokens: int = 0, batch_trained_tokens: int = 0,
                 batch_estimated_flops: int | None = None):
        if self.flops_estimate_method is not None and batch_estimated_flops is None:
            raise ValueError("A configured FLOP estimator must provide every step's estimate")
        if batch_estimated_flops is not None:
            if batch_estimated_flops < 0:
                raise ValueError("Step FLOPs must be nonnegative")
            self.total_estimated_flops += batch_estimated_flops
        self.total_tokens_seen += batch_non_pad_tokens
        self.trained_tokens_seen += batch_trained_tokens
        self.total_examples_seen += batch_size
        elapsed = max(0.0, time.perf_counter() - self.start_time - self.excluded_seconds)
        token_per_sec = self.total_tokens_seen / elapsed if elapsed > 0.0 else 0.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        record = {
            "step": step,
            "loss": round(loss, 4) if loss is not None else None,
            "lr": lr,
            "tokens_seen": self.total_tokens_seen,
            "trained_tokens_seen": self.trained_tokens_seen,
            "examples_seen": self.total_examples_seen,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(token_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "eval_tokens": eval_tokens,
            "estimated_flops": (self.total_estimated_flops
                                if batch_estimated_flops is not None else None),
            "flops_estimate_method": (self.flops_estimate_method
                                      if batch_estimated_flops is not None else None),
        }

        self.history.append(record)
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        loss_text = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"Step: {step:<4} | Loss: {loss_text} | TPS: {token_per_sec:.1f} | Tokens: {self.total_tokens_seen:,} | Elapsed: {elapsed:.2f}s | VRAM: {peak_vram_mb:.1f}MB")

if __name__ == "__main__":
    tracker = EfficiencyTracker("test_run")
    tracker.start()

    # Simulate 5 steps
    for i in range(0, 5):
        tracker.log_step(step=i, loss=3.5 - i * 0.1, lr=1e-4, batch_non_pad_tokens=250, batch_size=4)
