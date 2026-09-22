import os
import json
import numpy

def load_log(filepath: str) -> list[dict]:
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))  # json.loads parses a single string line
    except FileNotFoundError:
        print(f"Log file {filepath} not found")
    return data

def tokens_to_target(history: list[dict], target_loss: float) -> int | None:
    for item in history:
        if item["loss"] <= target_loss:
            return item["tokens_seen"]

    return None

def time_to_target(history: list[dict], target_loss: float) -> float | None:
    for item in history:
        if item["loss"] <= target_loss:
            return item["elapsed_seconds"]

    return None

def compute_to_target(history: list[dict], target_loss: float, total_params: int = 494_032_768) -> float | None:
    tokens = tokens_to_target(history, target_loss)
    if tokens is not None:
        return 6 * total_params * tokens

    return None

# Normalized Area Under Learning Curve
def calculate_naulc(history: list[dict], token_budget: int = 0) -> float:
    filtered = [h for h in history if token_budget == 0 or h["tokens_seen"] <= token_budget]
    if not filtered:
        return 0

    x = [h["tokens_seen"] for h in filtered]
    y = [h["loss"] for h in filtered]

    return numpy.trapezoid(y, x) / max(x)

def find_pareto_frontier(results: list[dict], metric_x: str, metric_y: str) -> list[dict]:
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: (r[metric_x], r[metric_y]))

    frontier = []
    min_y = float("inf")

    for result in sorted_results:
        x, y = result[metric_x], result[metric_y]

        if y < min_y:
            frontier.append(result)
            min_y = y
        elif(frontier and x == frontier[-1][metric_x] and y == frontier[-1][metric_y]):
            frontier.append(result)


    return frontier

def generate_summary_table(log_files: list[str], target_loss: float = 1.20, output_path: str | None = None):
    header = "| Method | Final Loss | PPL | Tokens-to-Target | Time-to-Target | FLOPs-to-Target | Peak VRAM | NAULC |"
    lines = [header, "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    print(header)

    for file in log_files:
        log = load_log(filepath=file)
        if not log:
            continue

        fl = log[-1]["loss"]
        ppl = numpy.exp(fl) if not numpy.isnan(fl) else float("nan")
        tokentt = tokens_to_target(log, target_loss)
        timett = time_to_target(log, target_loss)
        floptt = compute_to_target(log, target_loss)
        peakvram = log[-1]["peak_vram_mb"]
        naulc = calculate_naulc(log)

        line = f"| {os.path.basename(file).replace('.jsonl', '')} | {fl:.4f} | {ppl:.2f} | {tokentt} | {timett} | {floptt} | {peakvram:.1f} | {naulc:.4f} |"
        print(line)
        lines.append(line)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSummary table saved to {output_path}")

if __name__ == "__main__":
    # Test run

    history = load_log("logs/lora_stage0_test.jsonl")
    print("Tokens to loss 1.20:", tokens_to_target(history, target_loss=1.20))
    print("Time to loss 1.20:  ", time_to_target(history, target_loss=1.20))
    print("NAULC (Area):       ", calculate_naulc(history)) 

