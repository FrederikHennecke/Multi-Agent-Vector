from pathlib import Path

from data.datasets_mmlu_pro import build_eval_set_mmlu_pro

def parse_dataset_tag(tag: str):
    """
    Accepts:
      mmlu_pro:<category>   (e.g., mmlu_pro:law)
    Returns ('mmlu_pro', 'law') etc.
    """
    if not tag:
        raise ValueError("Dataset tag cannot be empty.")
    if ":" in tag:
        name, cfg = tag.split(":", 1)
    else:
        name, cfg = tag, None
    return name, cfg

def build_eval_set(dataset_tag: str, max_eval: int):
    name, cfg = parse_dataset_tag(dataset_tag)
    if name != "mmlu_pro":
        raise ValueError(f"Unsupported dataset_tag '{name}'. Only 'mmlu_pro:<category>' is supported.")
    if not cfg:
        raise ValueError("MMLU-Pro requires an explicit category tag like 'mmlu_pro:law'.")
    return build_eval_set_mmlu_pro(Path(f"experiments_8B/mmlu_pro_splits/{cfg}_test.jsonl"), cfg, max_eval)

def write_summary_row(csv_path: Path, setup: str, dataset_tag: str, n: int, elapsed_s: float):
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        import csv
        w = csv.writer(f)
        if new_file:
            w.writerow(["setup", "dataset", "num_examples", "seconds"])
        w.writerow([setup, dataset_tag, n, f"{elapsed_s:.2f}"])
