import os
import json
import numpy

def _validation_history(history: list[dict]) -> list[dict]:
    return [item for item in history
            if item.get("val_loss") is not None and numpy.isfinite(item["val_loss"])]

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
    for item in _validation_history(history):
        if item["val_loss"] <= target_loss:
            return item["tokens_seen"]

    return None

def time_to_target(history: list[dict], target_loss: float) -> float | None:
    for item in _validation_history(history):
        if item["val_loss"] <= target_loss:
            return item["elapsed_seconds"]

    return None

def compute_to_target(history: list[dict], target_loss: float) -> float | None:
    for item in _validation_history(history):
        if item["val_loss"] <= target_loss:
            return item.get("estimated_flops")
    return None

def calculate_naulc(history: list[dict], token_budget: int = 0) -> float:
    """Mean validation loss up to a shared token budget (lower is better)."""
    points = sorted((h["tokens_seen"], h["val_loss"]) for h in _validation_history(history))
    if len(points) < 2:
        return float("nan")
    x, y = map(list, zip(*points))
    end = token_budget or x[-1]
    if end <= x[0] or end > x[-1]:
        return float("nan")
    if end < x[-1]:
        y_end = float(numpy.interp(end, x, y))
        selected = [(tokens, loss) for tokens, loss in points if tokens < end]
        selected.append((end, y_end))
        x, y = map(list, zip(*selected))
    return float(numpy.trapezoid(y, x) / (end - x[0]))

def find_pareto_frontier(results: list[dict], metric_x: str, metric_y: str,
                         maximize_y: bool = True) -> list[dict]:
    """Return points minimizing x and maximizing y (or minimizing y if requested)."""
    if not results:
        return []

    direction = -1 if maximize_y else 1
    sorted_results = sorted(results, key=lambda r: (r[metric_x], direction * r[metric_y]))

    frontier = []
    best_y = float("-inf") if maximize_y else float("inf")

    for result in sorted_results:
        x, y = result[metric_x], result[metric_y]

        improves = y > best_y if maximize_y else y < best_y
        if improves:
            frontier.append(result)
            best_y = y
        elif(frontier and x == frontier[-1][metric_x] and y == frontier[-1][metric_y]):
            frontier.append(result)


    return frontier

def generate_summary_table(log_files: list[str], target_loss: float = 1.20, output_path: str | None = None):
    runs = [(file, load_log(file)) for file in log_files]
    endpoints = [_validation_history(log)[-1]["tokens_seen"]
                 for _, log in runs if _validation_history(log)]
    common_budget = min(endpoints) if endpoints else 0
    header = f"| Method | Final Val Loss | Val PPL | Tokens-to-Target | Time-to-Target | Estimated FLOPs-to-Target | Peak VRAM | Mean Val Loss (to {common_budget:,} tokens) |"
    lines = [header, "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    print(header)

    for file, log in runs:
        if not log:
            continue

        validation = _validation_history(log)
        fl = validation[-1]["val_loss"] if validation else float("nan")
        ppl = numpy.exp(fl) if numpy.isfinite(fl) else float("nan")
        tokentt = tokens_to_target(log, target_loss)
        timett = time_to_target(log, target_loss)
        floptt = compute_to_target(log, target_loss)
        peakvram = max(item.get("peak_vram_mb", 0) for item in log)
        naulc = calculate_naulc(log, token_budget=common_budget)

        fl_text = f"{fl:.4f}" if numpy.isfinite(fl) else "N/A"
        ppl_text = f"{ppl:.2f}" if numpy.isfinite(ppl) else "N/A"
        auc_text = f"{naulc:.4f}" if numpy.isfinite(naulc) else "N/A"
        line = f"| {os.path.basename(file).replace('.jsonl', '')} | {fl_text} | {ppl_text} | {tokentt} | {timett} | {floptt} | {peakvram:.1f} | {auc_text} |"
        print(line)
        lines.append(line)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSummary table saved to {output_path}")

if __name__ == "__main__":
    # Test run

    history = load_log("logs/lora_stage0_test.jsonl")
    print("Tokens to loss 1.20:", tokens_to_target(history, target_loss=1.20))
    print("Time to loss 1.20:  ", time_to_target(history, target_loss=1.20))
    print("Mean validation loss: ", calculate_naulc(history))

