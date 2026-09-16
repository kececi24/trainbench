import os
import time
import json
import torch

class EfficiencyTracker():

    def __init__(self, run_name: str, log_dir: str = "logs") -> None:
        self.run_name = run_name

        os.makedirs(log_dir, exist_ok=True)
        self.logfile_path = f"{log_dir}/{run_name}.jsonl"

        self.start_time: float = 0.0
        self.total_tokens_seen = 0
        self.total_examples_seen = 0
        self.history = []

    def start(self):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()

    def log_step(self, step: int, loss: float, lr: float, batch_non_pad_tokens: int, batch_size: int):
        self.total_tokens_seen += batch_non_pad_tokens
        self.total_examples_seen += batch_size
        elapsed = time.perf_counter() - self.start_time
        token_per_sec = self.total_tokens_seen / elapsed if elapsed > 0.0 else 0.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        record = {
            "step": step,
            "loss": round(loss, 4),
            "lr": lr,
            "tokens_seen": self.total_tokens_seen,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(token_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2)
        }

        self.history.append(record)
        with open(self.logfile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Step: {step:<4} | Loss: {loss:.4f} | TPS: {token_per_sec:.1f} | Tokens: {self.total_tokens_seen:,} | Elapsed: {elapsed:.2f}s | VRAM: {peak_vram_mb:.1f}MB")

if __name__ == "__main__":
    tracker = EfficiencyTracker("test_run")
    tracker.start()

    # Simulate 5 steps
    for i in range(0, 5):
        tracker.log_step(step=i, loss=3.5 - i * 0.1, lr=1e-4, batch_non_pad_tokens=250, batch_size=4)