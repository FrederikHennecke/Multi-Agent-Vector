#!/usr/bin/env python3
"""
Benchmark generation throughput for prompt, steering-vector, and LoRA setups.

This script mirrors the resource layout used by main.py but focuses solely on
timing token generation for the generator role. It loads prompts, runs each
requested setup, reports tokens/sec, and saves a simple plot.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Use a non-interactive backend so plots work in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from AgentFactory import AgentFactory  # noqa: E402


def infer_model_size_label(base_model: str) -> str:
    """
    Infer the size label (e.g., '8' for 8B) from a model id.
    Falls back to '8' if no pattern is found.
    """
    m = re.search(r"Llama-[\d.]+-(\d+)B", base_model)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)B", base_model)
    if m:
        return m.group(1)
    return "8"


def default_prompt_file(work_dir: Path) -> Path | None:
    candidates = [
        work_dir / "data" / "generator_malt_v1.jsonl",
        work_dir / "data" / "generator.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_prompts(path: Path | None, limit: int, keys: Iterable[str], seed: int) -> List[str]:
    """
    Load prompts from a jsonl/text file, or fall back to a few generic prompts.
    """
    prompts: List[str] = []
    if path and path.exists():
        if path.suffix == ".jsonl":
            with path.open() as f:
                for line in f:
                    if len(prompts) >= limit:
                        break
                    rec = json.loads(line)
                    text = next((rec.get(k, "") for k in keys if rec.get(k)), "")
                    if text:
                        prompts.append(text)
        else:
            with path.open() as f:
                for line in f:
                    if len(prompts) >= limit:
                        break
                    text = line.strip()
                    if text:
                        prompts.append(text)
    if not prompts:
        prompts = [
            "Explain the photoelectric effect to a high school student.",
            "List three key differences between mitosis and meiosis.",
            "Solve: A train travels 120 km in 2 hours. What is its average speed in km/h?",
        ]
    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:limit]


def make_vector_paths(vector_dir: Path, variant: str) -> Dict[str, str]:
    roles = ["generator", "verifier", "refiner"]
    paths: Dict[str, str] = {}
    for role in roles:
        path = vector_dir / f"{role}_{variant}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing steering vector for role '{role}': {path}")
        paths[role] = str(path)
    return paths


def make_lora_paths(lora_dir: Path) -> Dict[str, str]:
    roles = ["generator", "verifier", "refiner"]
    paths: Dict[str, str] = {}
    for role in roles:
        path = lora_dir / role
        if not path.exists():
            raise FileNotFoundError(f"Missing LoRA adapter for role '{role}': {path}")
        paths[role] = str(path)
    return paths


def build_generator(setup: str, args, data_dir: Path, lora_dir: Path, vector_dir: Path):
    """
    Return (generator_agent, host) for the requested setup.
    """
    setup = setup.lower()
    if setup not in {"prompt", "vector", "lora"}:
        raise ValueError(f"Unsupported setup '{setup}'. Choose from: prompt, vector, lora.")

    factory_kwargs: Dict[str, object] = {
        "base_model_id": args.base_model,
        "data_dir": data_dir,
        "layer_idx": args.layer_idx,
    }

    if setup == "vector":
        factory_kwargs["steering_paths"] = make_vector_paths(vector_dir, args.vector_variant)
    if setup == "lora":
        factory_kwargs["lora_adapters"] = make_lora_paths(lora_dir)

    factory = AgentFactory(**factory_kwargs)
    factory.max_new_tokens = args.max_new_tokens

    if setup == "prompt":
        agents, host = factory.build_prompt()
    elif setup == "vector":
        agents, host = factory.build_vector(use_prompts=True, minimal_hints=False)
    else:  # lora
        agents, host = factory.build_lora(use_prompts=True, minimal_hints=False)

    return agents["generator"], host


def benchmark_agent(setup_name: str, agent, prompts: List[str], sync_cuda: bool) -> List[Dict[str, object]]:
    tok = agent.host.tok
    records: List[Dict[str, object]] = []
    warmup_prompt = prompts[0]

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    # One warmup pass to avoid measuring compile/startup overhead.
    _ = agent.generate(warmup_prompt)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    for idx, prompt in enumerate(prompts):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = agent.generate(prompt)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen_tokens = len(tok(output, add_special_tokens=False).input_ids)
        tps = gen_tokens / elapsed if elapsed > 0 else float("inf")
        records.append(
            {
                "setup": setup_name,
                "prompt_idx": idx,
                "elapsed_s": elapsed,
                "generated_tokens": gen_tokens,
                "tokens_per_s": tps,
                "prompt_preview": prompt[:140],
                "output_preview": output[:200],
            }
        )
    return records


def plot_throughput(df: pd.DataFrame, out_path: Path):
    summary = (
        df.groupby("setup")["tokens_per_s"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "tps_mean", "std": "tps_std"})
    )
    plt.figure(figsize=(6, 4))
    plt.bar(summary["setup"], summary["tps_mean"], yerr=summary["tps_std"].fillna(0.0), capsize=6)
    plt.ylabel("Tokens per second")
    plt.title("Generation throughput by setup")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation throughput for different setups.")
    parser.add_argument("--setups", nargs="+", default=["prompt", "vector", "lora"],
                        help="Which setups to run. Choices: prompt, vector, lora.")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="mmlu_pro:chemistry",
                        help="Dataset key used to locate vectors/LoRA under experiments_<size>B/<dataset>/")
    parser.add_argument("--prompt_file", type=Path, default=None,
                        help="Optional prompt source. Defaults to generator_malt_v1.jsonl under the dataset directory.")
    parser.add_argument("--prompt_keys", default="user,raw_question,question,input",
                        help="Comma-separated keys to read prompts from when using a jsonl file.")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--vector_variant", default="so_prompt",
                        help="Suffix used in vector filenames, e.g., wi_prompt, fs_prompt, so_prompt.")
    parser.add_argument("--plot_path", type=Path, default=None,
                        help="Where to save the throughput plot. Default: results_<size>B/generation_benchmark.png")
    parser.add_argument("--csv_path", type=Path, default=None,
                        help="Where to save raw benchmark data as CSV. Default: results_<size>B/generation_benchmark.csv")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sync_cuda", action="store_true",
                        help="Synchronize CUDA before/after timing for more accurate GPU timings.")
    args = parser.parse_args()

    model_size = infer_model_size_label(args.base_model)
    work_dir = Path(f"experiments_{model_size}B") / args.dataset
    data_dir, lora_dir, vector_dir = work_dir / "data", work_dir / "lora", work_dir / "vectors"

    if not work_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {work_dir}")

    prompt_file = args.prompt_file or default_prompt_file(work_dir)
    prompt_keys = [k.strip() for k in args.prompt_keys.split(",") if k.strip()]
    prompts = load_prompts(prompt_file, args.num_prompts, prompt_keys, args.seed)
    if not prompts:
        raise RuntimeError("No prompts available for benchmarking.")

    records: List[Dict[str, object]] = []
    for setup in args.setups:
        print(f"[INFO] Running setup: {setup}")
        agent, host = build_generator(setup, args, data_dir, lora_dir, vector_dir)
        try:
            records.extend(benchmark_agent(setup, agent, prompts, args.sync_cuda))
        finally:
            host.unload()

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("Benchmark produced no data.")

    summary = (
        df.groupby("setup")
        .agg(
            avg_tokens=("generated_tokens", "mean"),
            avg_latency_s=("elapsed_s", "mean"),
            avg_tps=("tokens_per_s", "mean"),
        )
        .reset_index()
    )
    print("\n=== Throughput summary ===")
    for _, row in summary.iterrows():
        print(
            f"{row['setup']}: "
            f"{row['avg_tps']:.2f} tokens/s, "
            f"{row['avg_latency_s']:.2f}s avg latency, "
            f"{row['avg_tokens']:.1f} tokens avg length"
        )

    results_dir = Path(f"results_{model_size}B")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv_path or (results_dir / "generation_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Raw results saved to {csv_path}")

    plot_path = args.plot_path or (results_dir / "generation_benchmark.png")
    plot_throughput(df, plot_path)
    print(f"[INFO] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
