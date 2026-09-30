import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

try:
    from analysis import load_log
except ImportError:
    from src.analysis import load_log

# Required VRAM (MB) for OOM runs calculated from sharded parameters, optimizer states, activations, and buffers
FAILED_REQUIRED_VRAM_MB = {
    ("llama3.2_3b", "fft"): 45.9 * 1024,
    ("qwen2.5_3b", "fft"): 44.3 * 1024,
    ("minitron_4b", "fft"): 56.2 * 1024,
    ("nemotron_mini_4b", "fft"): 56.2 * 1024,
    ("llama2_7b", "fft"): 42.9 * 1024,
    ("qwen2.5_7b", "fft"): 48.1 * 1024,
    ("llama3.1_8b", "fft"): 50.3 * 1024,
    ("llama2_13b", "fft"): 78.5 * 1024,
    ("qwen2.5_14b", "fft"): 88.2 * 1024,
    ("qwen2.5_14b", "lora"): 32.8 * 1024,
    ("qwen2.5_14b", "dora"): 33.2 * 1024,
}

def estimate_required_vram_mb(model: str, method: str, pre_crash_peak: float) -> float:
    """Return empirically modeled required VRAM or analytical fallback for failed runs."""
    if (model, method) in FAILED_REQUIRED_VRAM_MB:
        req = FAILED_REQUIRED_VRAM_MB[(model, method)]
        if pre_crash_peak > req:
            return pre_crash_peak * 1.15
        return req
    params = MODEL_PARAMS.get(model, 7e9)
    if method == "fft":
        return max(pre_crash_peak * 2.0, (params * 8 / (1024**2)) + 12000)
    elif method in ("lora", "dora"):
        return max(pre_crash_peak * 1.2, (params * 2 / (1024**2)) + 14000)
    return max(pre_crash_peak * 1.3, 33000)

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

def safe_savefig(fig, save_path: str, dpi: int = 300, **kwargs):
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except OSError:
            pass
    fig.savefig(save_path, dpi=dpi, **kwargs)

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
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
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
    """Figure 3: Held-out loss vs. estimated FLOPs, when available."""
    if not any(h.get("estimated_flops") is not None and h.get("val_loss") is not None
               for history in runs.values() for h in history):
        print("Skipping FLOPs plot: logs contain no FLOP measurements.")
        return
    _plot_faceted_metric(
        runs=runs,
        x_key="tokens_seen",
        x_label="Estimated FLOPs",
        title="Validation Loss vs. Estimated Compute",
        save_path=save_path,
        compute_flops=True,
        override_params=total_params
    )


def plot_vram_comparison(runs: dict[str, list[dict]], save_path: str = "plots/fig4_vram_comparison.png"):
    """Figure 4: Grouped horizontal bar chart of peak VRAM across models and methods, with extended deficit bars for OOM runs."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    models_dict = group_runs_by_model(runs)
    models = sorted(models_dict.keys())
    methods = ["fft", "lora", "dora", "qlora"]

    fig, ax = plt.subplots(figsize=(12.5, max(6.0, len(models) * 0.76)))

    y = np.arange(len(models))
    total_bar_height = 0.75
    bar_height = total_bar_height / len(methods)

    has_failed = False
    max_observed_val = 32768.0

    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * bar_height
        for m_idx, model in enumerate(models):
            if method in models_dict[model]:
                history = models_dict[model][method]
                peak = max((h.get("peak_vram_mb", 0) for h in history), default=0)
                has_train_loss = any(h.get("loss") is not None for h in history)
                max_step = max((h.get("step", 0) for h in history), default=0)
                failed = (max_step < 10 or not has_train_loss) and peak > 0

                bar_y = y[m_idx] + offset
                c = METHOD_COLORS[method]

                if not failed:
                    if peak > 0:
                        ax.barh(bar_y, peak, height=bar_height * 0.9, color=c, edgecolor="none")
                        lbl = f"{peak:.0f} MB" if peak < 10000 else f"{peak/1024:.1f} GB"
                        ax.text(
                            peak + 500, bar_y, lbl,
                            va="center", ha="left", fontsize=7.5, color="#222222", fontweight="500"
                        )
                        max_observed_val = max(max_observed_val, peak)
                else:
                    has_failed = True
                    req_val = estimate_required_vram_mb(model, method, peak)
                    max_observed_val = max(max_observed_val, req_val)

                    # 1. Base pre-crash block (hatched)
                    ax.barh(
                        bar_y, peak, height=bar_height * 0.9,
                        color=c, alpha=0.35, hatch="//", edgecolor=c, linewidth=0.8
                    )
                    # 2. Extended required block (lighter, dotted hatch)
                    ext_width = max(0, req_val - peak)
                    ax.barh(
                        bar_y, ext_width, left=peak, height=bar_height * 0.9,
                        color=c, alpha=0.14, hatch="..", edgecolor=c, linestyle="--", linewidth=0.8
                    )
                    # Label at end of extension
                    lbl = f"{req_val/1024:.1f} GB (Req, crash @ {peak/1024:.1f}G)" if peak >= 1024 else f"{req_val/1024:.1f} GB (Req)"
                    ax.text(
                        req_val + 500, bar_y, lbl,
                        va="center", ha="left", fontsize=7.2, color="#666666", fontstyle="italic"
                    )

    # Reference limit lines
    # 24GB RTX 4090 (24,576 MB)
    ax.axvline(24576, color="#e41a1c", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(24576 - 800, -0.45, "24 GB (RTX 4090) ", color="#e41a1c", fontsize=8.5, va="bottom", ha="right", fontweight="bold")

    # 32GB RTX 5090 (32,768 MB)
    ax.axvline(32768, color="#0072b2", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(32768 + 800, -0.45, " 32 GB (RTX 5090)", color="#0072b2", fontsize=8.5, va="bottom", ha="left", fontweight="bold")

    # 64GB Combined 2x RTX 5090 (65,536 MB) if required VRAM exceeds single-card limits
    if max_observed_val > 40000:
        ax.axvline(65536, color="#555555", linestyle=":", linewidth=1.3, alpha=0.75)
        ax.text(65536, -0.45, " 64 GB (2x 5090 Combined)", color="#444444", fontsize=8.5, va="bottom", ha="left", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("GPU Memory (MB) — Solid: Achieved Peak | Extended: Required for Training", fontsize=10.5)
    ax.set_title("Per-GPU Peak VRAM & Extended Requirements for OOM Configurations", fontsize=12.5, fontweight="bold", pad=15)

    # Build legend
    handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in methods
    ]
    if has_failed:
        handles.append(mpatches.Patch(facecolor="#888888", alpha=0.35, hatch="//", edgecolor="#555555", label="Pre-Crash Allocation (OOM @ Step 0)"))
        handles.append(mpatches.Patch(facecolor="#888888", alpha=0.14, hatch="..", edgecolor="#555555", linestyle="--", label="Extended Required VRAM"))

    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8.8, framealpha=0.95)
    if max_observed_val > 35000:
        ax.set_xlim(0, max(115000, max_observed_val * 1.12))
    else:
        ax.set_xlim(0, 37500)

    fig.tight_layout()
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
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
        target_save = args.save_dir or ("plots5090dual" if "5090dual" in args.logs_dir else ("plots5090" if "5090" in args.logs_dir else "plots"))
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
        # 1. Generate for results5090dual/logs if present
        if os.path.exists("results5090dual/logs"):
            logs_5090dual = sorted(glob.glob("results5090dual/logs/*.jsonl"))
            if logs_5090dual:
                log_map_5090dual = {os.path.basename(f).replace(".jsonl", ""): f for f in logs_5090dual}
                print(f"Plotting comparisons across {len(log_map_5090dual)} run(s) from results5090dual/logs to plots5090dual/...")
                plot_all(log_map_5090dual, save_path="plots5090dual")
                print("Successfully generated available benchmark plots in plots5090dual/!")
                generated_any = True

        # 2. Generate for results5090/logs if present
        if os.path.exists("results5090/logs"):
            logs_5090 = sorted(glob.glob("results5090/logs/*.jsonl"))
            if logs_5090:
                log_map_5090 = {os.path.basename(f).replace(".jsonl", ""): f for f in logs_5090}
                print(f"Plotting comparisons across {len(log_map_5090)} run(s) from results5090/logs to plots5090/...")
                plot_all(log_map_5090, save_path="plots5090")
                print("Successfully generated available benchmark plots in plots5090/!")
                generated_any = True

        # 3. Generate for logs/ if present
        if os.path.exists("logs"):
            logs_default = sorted(glob.glob("logs/*.jsonl"))
            if logs_default:
                log_map_default = {os.path.basename(f).replace(".jsonl", ""): f for f in logs_default}
                print(f"Plotting comparisons across {len(log_map_default)} run(s) from logs to plots/...")
                plot_all(log_map_default, save_path="plots")
                print("Successfully generated available benchmark plots in plots/!")
                generated_any = True

        if not generated_any:
            print("No log files found in logs/, results5090/logs, or results5090dual/logs.")
