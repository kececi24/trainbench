import os
import matplotlib.pyplot as plt

from analysis import load_log

def plot_loss_vs_tokens(runs: dict[str, list[dict]], save_path: str = "plots/fig1_loss_vs_tokens.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, r in runs.items():
        tokens_seen = [l["tokens_seen"] for l in r]
        loss = [l["loss"] for l in r]

        ax.plot(tokens_seen, loss, label=key, linewidth=2)
        ax.set_xlabel("Tokens Seen")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs. Tokens Seen")

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_loss_vs_time(runs: dict[str, list[dict]], save_path: str = "plots/fig2_loss_vs_time.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, r in runs.items():
        elapsed = [l["elapsed_seconds"] for l in r]
        loss = [l["loss"] for l in r]

        ax.plot(elapsed, loss, label=key, linewidth=2)
        ax.set_xlabel("Elapsed Seconds")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs. Elapsed Seconds")

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_loss_vs_compute(runs: dict[str, list[dict]], save_path: str = "plots/fig3_loss_vs_compute.png", total_params: int = 494_032_768):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, r in runs.items():
        flops = [6 * total_params * l["tokens_seen"] for l in r]
        loss = [l["loss"] for l in r]

        ax.plot(flops, loss, label=key, linewidth=2)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs. Compute")

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_vram_comparison(runs: dict[str, list[dict]], save_path: str = "plots/fig4_vram_comparison.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5))

    vrams = [max(l["peak_vram_mb"] for l in r) for r in runs.values()]
    methods = list(runs.keys())

    bars = ax.bar(methods, vrams, color="steelblue", width=0.4)
    ax.set_xlabel("Method")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Peak GPU Memory by Method")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_all(log_files: dict[str, str], save_path: str = "plots/"):
    runs = {}

    for name, log_path in log_files.items():
        runs[name] = load_log(log_path)

    if runs:
        plot_loss_vs_tokens(runs, f"{save_path}fig1_loss_vs_tokens.png")
        plot_loss_vs_time(runs, f"{save_path}fig2_loss_vs_time.png")
        plot_loss_vs_compute(runs, f"{save_path}fig3_loss_vs_compute.png")
        plot_vram_comparison(runs, f"{save_path}fig4_vram_comparison.png")


if __name__ == "__main__":
    import glob
    os.makedirs("plots", exist_ok=True)
    all_logs = sorted(glob.glob("logs/*.jsonl"))
    if all_logs:
        log_mapping = {os.path.basename(f).replace(".jsonl", ""): f for f in all_logs}
        print(f"Plotting comparisons across {len(log_mapping)} run(s) found in logs/...")
        plot_all(log_mapping)
    else:
        print("No log files found in logs/.")