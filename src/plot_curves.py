import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

try:
    from analysis import load_log
except ImportError:
    from src.analysis import load_log

# Visual palette for fine-tuning methods
METHOD_COLORS = {
    "fft": "#e41a1c",     # Strong Red
    "lora": "#377eb8",    # Professional Blue
    "dora": "#984ea3",    # Deep Purple
    "qlora": "#4daf4a",   # Emerald Green
}

METHOD_LABELS = {
    "fft": "FFT (Full Fine-Tuning)",
    "lora": "LoRA (r=16, alpha=32)",
    "dora": "DoRA (r=16, alpha=32)",
    "qlora": "QLoRA (4-bit NF4)",
}

# Approximate model parameter counts for subplot labels only
MODEL_PARAMS = {
    "gpt2_medium": 355_000_000,
    "pythia_410m": 410_000_000,
    "qwen2.5_0.5b": 494_032_768,
    "llama3.2_1b": 1_230_000_000,
    "pythia_1.4b": 1_400_000_000,
    "qwen2.5_1.5b": 1_540_000_000,
    "llama3.2_3b": 3_210_000_000,
    "qwen2.5_3b": 3_090_000_000,
    "minitron_4b": 4_000_000_000,
    "nemotron_mini_4b": 4_000_000_000,
    "llama2_7b": 6_740_000_000,
    "qwen2.5_7b": 7_615_616_000,
    "llama3.1_8b": 8_030_000_000,
    "llama2_13b": 13_020_000_000,
    "qwen2.5_14b": 14_770_000_000,
}

def parse_run_name(name: str) -> tuple[str, str]:
    """Extract (model_name, method_name) from run identifier."""
    for m in ["qlora", "dora", "lora", "fft"]:
        if name.endswith(f"_{m}"):
            return name[:-len(m)-1], m
    return name, "unknown"

def smooth_curve(values: list[float], weight: float = 0.8) -> list[float]:
    """Exponential Moving Average (EMA) smoothing for noisy step loss curves."""
    smoothed = []
    last = None
    for v in values:
        if v is None or math.isnan(v):
            smoothed.append(v)
            continue
        if last is None:
            last = v
        else:
            last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

def group_runs_by_model(runs: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """Organize flat runs dictionary into {model: {method: history}}."""
    models = {}
    for run_name, history in runs.items():
        model, method = parse_run_name(run_name)
        if model not in models:
            models[model] = {}
        models[model][method] = history
    return models

def _plot_faceted_metric(
    runs: dict[str, list[dict]],
    x_key: str,
    x_label: str,
    title: str,
    save_path: str,
    compute_flops: bool = False,
    override_params: int | None = None
):
    """Plot held-out validation curves organized by model with EMA smoothing."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    models = group_runs_by_model(runs)
    n_models = len(models)
    if n_models == 0:
        return

    if n_models == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
        model_names = list(models.keys())
    else:
        cols = min(4, n_models)
        rows = math.ceil(n_models / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), squeeze=False)
        axes = axes.flatten()
        model_names = sorted(models.keys())

    legend_handles = {}

    for i, model in enumerate(model_names):
        ax = axes[i]
        methods = models[model]
        params = override_params if override_params is not None else MODEL_PARAMS.get(model, 500_000_000)

        for method in ["fft", "lora", "dora", "qlora"]:
            if method not in methods:
                continue
            history = methods[method]
            valid = [h for h in history
                     if h.get("val_loss") is not None and math.isfinite(h["val_loss"])
                     and (not compute_flops or h.get("estimated_flops") is not None)]
            if not valid:
                continue

            if compute_flops:
                x = [h["estimated_flops"] for h in valid]
            else:
                x = [h[x_key] for h in valid]
            y = [h["val_loss"] for h in valid]

            color = METHOD_COLORS.get(method, "gray")
            label = METHOD_LABELS.get(method, method.upper())

            # Plot raw noisy loss with transparency and solid EMA trend line
            ax.plot(x, y, color=color, alpha=0.22, linewidth=1)
            y_smooth = smooth_curve(y, weight=0.8)
            line, = ax.plot(x, y_smooth, color=color, linewidth=2, label=label)

            if method not in legend_handles:
                legend_handles[method] = line

        param_str = f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        ax.set_title(f"{model} (~{param_str})", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Validation Loss (Cross-Entropy)", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(len(model_names), len(axes)):
        axes[j].axis("off")

    handles = [legend_handles[m] for m in ["fft", "lora", "dora", "qlora"] if m in legend_handles]
    labels = [METHOD_LABELS[m] for m in ["fft", "lora", "dora", "qlora"] if m in legend_handles]
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(handles)),
            frameon=True,
            fontsize=10
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_vs_tokens(runs: dict[str, list[dict]], save_path: str = "plots/fig1_loss_vs_tokens.png"):
    """Figure 1: Held-out loss vs. tokens seen (data efficiency)."""
    _plot_faceted_metric(
        runs=runs,
        x_key="tokens_seen",
        x_label="Tokens Seen",
        title="Validation Loss vs. Tokens Seen (Data Efficiency)",
        save_path=save_path
    )


def plot_loss_vs_time(runs: dict[str, list[dict]], save_path: str = "plots/fig2_loss_vs_time.png"):
    """Figure 2: Held-out loss vs. training seconds."""
    _plot_faceted_metric(
        runs=runs,
        x_key="elapsed_seconds",
        x_label="Elapsed Seconds",
        title="Validation Loss vs. Training Seconds",
        save_path=save_path
    )


def plot_loss_vs_compute(
    runs: dict[str, list[dict]],
    save_path: str = "plots/fig3_loss_vs_compute.png",
    total_params: int | None = None
):
    """Figure 3: Held-out loss vs. recorded FLOPs, when available."""
    if not any(h.get("estimated_flops") is not None and h.get("val_loss") is not None
               for history in runs.values() for h in history):
        print("Skipping FLOPs plot: logs contain no FLOP measurements.")
        return
    _plot_faceted_metric(
        runs=runs,
        x_key="tokens_seen",
        x_label="Recorded FLOPs",
        title="Validation Loss vs. Recorded Compute",
        save_path=save_path,
        compute_flops=True,
        override_params=total_params
    )


def plot_vram_comparison(runs: dict[str, list[dict]], save_path: str = "plots/fig4_vram_comparison.png"):
    """Figure 4: Grouped horizontal bar chart of peak VRAM across models and methods."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    models_dict = group_runs_by_model(runs)
    models = sorted(models_dict.keys())
    methods = ["fft", "lora", "dora", "qlora"]

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(models) * 0.72)))

    y = np.arange(len(models))
    total_bar_height = 0.75
    bar_height = total_bar_height / len(methods)

    for idx, method in enumerate(methods):
        vrams = []
        for model in models:
            if method in models_dict[model]:
                vrams.append(max((h.get("peak_vram_mb", 0) for h in models_dict[model][method]), default=0))
            else:
                vrams.append(0)

        offset = (idx - (len(methods) - 1) / 2) * bar_height
        bars = ax.barh(
            y + offset,
            vrams,
            height=bar_height * 0.9,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="none"
        )

        for bar, val in zip(bars, vrams):
            if val > 0:
                ax.text(
                    val + 180,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f} MB" if val < 10000 else f"{val/1024:.1f} GB",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#333333"
                )

    # Reference limit line for 24GB RTX 4090 (24,576 MB)
    ax.axvline(24576, color="#e41a1c", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(24576, -0.4, " 24 GB Limit (RTX 4090)", color="#e41a1c", fontsize=8.5, va="bottom", ha="left", fontweight="bold")

    # Reference limit line for 32GB RTX 5090 (32,768 MB)
    ax.axvline(32768, color="#0072b2", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.text(32768, -0.4, " 32 GB Limit (RTX 5090)", color="#0072b2", fontsize=8.5, va="bottom", ha="left", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Peak GPU Memory (MB)", fontsize=10)
    ax.set_title("Peak VRAM Comparison by Model & Adaptation Method", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5, framealpha=0.92)
    ax.set_xlim(0, 37500)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all(log_files: dict[str, str], save_path: str = "plots/"):
    """Load runs and generate validation, memory, and available compute figures."""
    runs = {}
    for name, log_path in log_files.items():
        runs[name] = load_log(log_path)

    if runs:
        os.makedirs(save_path, exist_ok=True)
        plot_loss_vs_tokens(runs, os.path.join(save_path, "fig1_loss_vs_tokens.png"))
        plot_loss_vs_time(runs, os.path.join(save_path, "fig2_loss_vs_time.png"))
        plot_loss_vs_compute(runs, os.path.join(save_path, "fig3_loss_vs_compute.png"))
        plot_vram_comparison(runs, os.path.join(save_path, "fig4_vram_comparison.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark plots for LLM fine-tuning runs.")
    parser.add_argument("--logs-dir", type=str, default=None, help="Directory containing .jsonl run logs (e.g. logs or results5090/logs).")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save generated PNG plots (e.g. plots or plots5090).")
    args = parser.parse_args()

    if args.logs_dir:
        target_save = args.save_dir or ("plots5090" if "5090" in args.logs_dir else "plots")
        all_logs = sorted(glob.glob(os.path.join(args.logs_dir, "*.jsonl")))
        if all_logs:
            log_mapping = {os.path.basename(f).replace(".jsonl", ""): f for f in all_logs}
            print(f"Plotting comparisons across {len(log_mapping)} run(s) from {args.logs_dir} to {target_save}/...")
            plot_all(log_mapping, save_path=target_save)
            print(f"Successfully generated available benchmark plots in {target_save}/!")
        else:
            print(f"No log files found in {args.logs_dir}.")
    else:
        generated_any = False
        # 1. Generate for results5090/logs if present
        if os.path.exists("results5090/logs"):
            logs_5090 = sorted(glob.glob("results5090/logs/*.jsonl"))
            if logs_5090:
                log_map_5090 = {os.path.basename(f).replace(".jsonl", ""): f for f in logs_5090}
                print(f"Plotting comparisons across {len(log_map_5090)} run(s) from results5090/logs to plots5090/...")
                plot_all(log_map_5090, save_path="plots5090")
                print("Successfully generated available benchmark plots in plots5090/!")
                generated_any = True

        # 2. Generate for logs/ if present
        if os.path.exists("logs"):
            logs_default = sorted(glob.glob("logs/*.jsonl"))
            if logs_default:
                log_map_default = {os.path.basename(f).replace(".jsonl", ""): f for f in logs_default}
                print(f"Plotting comparisons across {len(log_map_default)} run(s) from logs to plots/...")
                plot_all(log_map_default, save_path="plots")
                print("Successfully generated available benchmark plots in plots/!")
                generated_any = True

        if not generated_any:
            print("No log files found in logs/ or results5090/logs.")
