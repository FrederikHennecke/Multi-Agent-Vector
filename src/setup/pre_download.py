import re
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from data.datasets_mmlu_pro import split_and_cache_mmlu_pro


def _infer_model_size_label(model_id: str) -> str:
    match = re.search(r"Llama-[\d.]+-(\d+)B", model_id)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)B", model_id)
    return match.group(1) if match else "8"

def main():
    # Base model(s) you plan to use
    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    # Cache MMLU-Pro splits so offline runs can reuse them.
    for m in models:
        print(f"Downloading {m} ...")
        AutoTokenizer.from_pretrained(
            m,
            token="", # fill token here
            local_files_only=False,
        )
        AutoModelForCausalLM.from_pretrained(
            m,
            token="", # fill token here
            local_files_only=False,
        )

    for size in sorted({_infer_model_size_label(m) for m in models}):
        cache_dir = Path(f"experiments_{size}B") / "mmlu_pro_splits"
        split_and_cache_mmlu_pro(cache_dir, seed=42, train_samples=10, force=False)


if __name__ == "__main__":
    main()
