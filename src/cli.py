import argparse
import json
import re
import time
from typing import List, Optional, Union, Dict, Set, Tuple

from transformers import AutoTokenizer

from AgentFactory import AgentFactory
from MultiAgentEvaluator import MultiAgentEvaluator
from data.dataset_builder import regenerate_verifier_refiner_with_lora_gen, build_role_datasets
from data.dataset_registry import write_summary_row, build_eval_set
from data.datasets_mmlu_pro import split_and_cache_mmlu_pro
from training.malt_data_generator import MALTDataGenerator
from training.role_trainer import RoleTrainer
from steering.vector_collector import SteeringVectorCollector, SteeringVectorConfig
from steering.domain_vector_manager import ensure_domain_vectors, ensure_combined_vectors, sanitize_domain_tag
from config import ROLE_SYSTEM_PROMPTS
from pathlib import Path
import torch



def run_config(base_model, condition, data_dir, max_eval, lora_adapters=None, steering_paths=None,
               fewshot_k=3, use_prompts=True, minimal_hints=False, verifier_decision="gen",
               layer_idx: Optional[Union[int, List[int]]] = None, fewshot_seed: Optional[int] = None):
    host_layer_idx = layer_idx
    if isinstance(host_layer_idx, (list, tuple)):
        host_layer_idx = host_layer_idx[0] if host_layer_idx else None
    if host_layer_idx is None:
        host_layer_idx = 20
    factory = AgentFactory(base_model, data_dir=data_dir, layer_idx=host_layer_idx,
                           lora_adapters=lora_adapters, steering_paths=steering_paths)
    if condition == "baseline":
        agents, host = factory.build_baseline()
    elif condition == "prompt":
        agents, host = factory.build_prompt(verifier_decision=verifier_decision)
    elif condition == "prompt_fs":
        agents, host = factory.build_prompt_fs(k=fewshot_k, verifier_decision=verifier_decision,
                                               seed=fewshot_seed)
    elif condition == "vector":
        agents, host = factory.build_vector(use_prompts=use_prompts, minimal_hints=minimal_hints, verifier_decision=verifier_decision)
    elif condition == "lora":
        agents, host = factory.build_lora(use_prompts=use_prompts, minimal_hints=minimal_hints, verifier_decision=verifier_decision)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return MultiAgentEvaluator(agents, host, max_data=max_eval)


def ensure_datasets(base_model, data_dir: Path, max_examples_per_dataset: int, builder_specs: str):
    """
    builder_specs: train dataset tag (e.g., mmlu_pro:law)
    Creates generator/verifier/refiner JSONLs in data_dir if not present.
    """
    required = ["generator.jsonl", "verifier.jsonl", "refiner.jsonl"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        print(f"[INFO] Building role datasets in {data_dir} (missing: {', '.join(missing)}) from: {builder_specs}")
        name, cfg, split = builder_tuple_from_train_key(builder_specs)[0]
        build_role_datasets(
            base_model_id=base_model,
            out_dir=data_dir,
            datasets_to_use=[(name, cfg)],
            max_examples_per_dataset=max_examples_per_dataset,
            split=split,
        )
    else:
        print("[INFO] Using cached datasets in", data_dir)


def ensure_lora(base_model, data_dir: Path, lora_dir: Path,train_from: str = "malt", malt_prefix: str = "malt_v1"):
    roles = ["generator", "verifier", "refiner"]
    trainer = RoleTrainer(base_model)
    adapters = {}
    if train_from == "malt":
        for role in roles:
            path = lora_dir / role
            if not path.exists():
                print(f"[INFO] Training LoRA ({role}) from MALT …")
                trainer.train_role_from_malt_dataset(
                    role,
                    data_dir / f"{role}_{malt_prefix}.jsonl",
                    path
                )
            adapters[role] = str(path)
        return adapters

    gen_path = lora_dir / "generator"
    need_gen = not gen_path.exists()

    if need_gen:
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] Training LoRA for role: generator…")
        trainer.train_role("generator", data_dir / "generator.jsonl", gen_path)
        print("[INFO] Regenerating verifier/refiner datasets using LoRA generator outputs…")
        regenerate_verifier_refiner_with_lora_gen(
            base_model_id=base_model,
            lora_gen_path=gen_path,
            data_dir=data_dir
        )

    for role in ["verifier", "refiner"]:
        path = lora_dir / role
        if not path.exists():
            if not need_gen:
                print("[INFO] Regenerating verifier/refiner datasets from existing LoRA generator before training…")
                regenerate_verifier_refiner_with_lora_gen(
                    base_model_id=base_model,
                    lora_gen_path=gen_path,
                    data_dir=data_dir
                )
            print(f"[INFO] Training LoRA for role: {role}…")
            trainer.train_role(role, data_dir / f"{role}.jsonl", path)

    for role in roles:
        adapters[role] = str(lora_dir / role)
    return adapters


def _load_fewshot_pool_for_vectors(data_dir: Path, role: str, cap: int = 200):
    """
    Load a compact few-shot pool from role JSONLs for vector collection.
    For verifier: extracts gen_output from input; for refiner: wrong_output.
    """
    pool = []
    with open(data_dir / f"{role}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= cap:
                break
            rec = json.loads(line)
            entry = {
                "raw_question": rec.get("raw_question", ""),
                "label": rec.get("label", "")
            }
            if isinstance(entry["label"], str):
                verdict_match = re.findall(r"\b(CORRECT|INCORRECT)\b", entry["label"].upper())
                entry["verdict"] = verdict_match[-1] if verdict_match else ""
            else:
                entry["verdict"] = ""
            s = rec.get("input", "")
            if role == "verifier":
                key = "SOLUTION:\n"
                entry["gen_output"] = s.split(key, 1)[1] if key in s else ""
            if role == "refiner":
                wrong = ""
                feedback = ""
                key = "INCORRECT SOLUTION:\n"
                if key in s:
                    rest = s.split(key, 1)[1]
                    fb_key = "\nVERIFIER FEEDBACK:\n"
                    if fb_key in rest:
                        wrong, fb_section = rest.split(fb_key, 1)
                        wrong = wrong.strip()
                        feedback = fb_section.strip()
                    else:
                        wrong = rest.strip()
                entry["wrong_output"] = wrong
                entry["verifier_feedback"] = feedback
            pool.append(entry)
    return pool


def _build_fewshot_user(role: str, examples: list, task: str) -> str:
    """
    Build the 'user' message content for few-shot vector collection, aligned with prompt_fs.
    """
    blocks = []
    for i, ex in enumerate(examples, 1):
        if role == "generator":
            blocks.append(f"### Example {i}\nQuestion:\n{ex.get('raw_question','')}\n{ex.get('label','')}")
        elif role == "verifier":
            gen_out = ex.get("gen_output", "")
            label_text = ex.get("label", "")
            if not isinstance(label_text, str):
                label_text = str(label_text)
            label_text = label_text.strip()
            if not label_text:
                label_text = ex.get("verdict", "")
            blocks.append(
                f"### Example {i}\nQUESTION:\n{ex.get('raw_question','')}\nSOLUTION:\n{gen_out}\n{label_text}"
            )
        elif role == "refiner":
            wrong = ex.get("wrong_output", "")
            feedback = ex.get("verifier_feedback", "")
            feedback_block = (
                f"\nVERIFIER FEEDBACK:\n{feedback.strip()}"
                if isinstance(feedback, str) and feedback.strip()
                else ""
            )
            blocks.append(
                f"### Example {i}\nQUESTION:\n{ex.get('raw_question','')}\nINCORRECT SOLUTION:\n{wrong}{feedback_block}\n{ex.get('label','')}"
            )
    if role == "verifier":
        trailer = (
            "### Now decide for the following. Respond with:\n"
            "VERDICT: CORRECT or INCORRECT\n"
            "CRITIQUE: <brief justification>"
        )
    elif role in ("generator", "refiner"):
        trailer = "### Now solve the following task. End your answer with: ANSWER: <answer>."
    else:
        trailer = "### Now handle the following task."
    return ("\n\n".join(blocks) + "\n\n" + trailer + "\n\n" + task) if blocks else task


def ensure_vectors_multi(base_model, data_dir: Path, vector_dir: Path,
                         modes: set,
                         dataset_tag: str,
                         vector_alpha: float, vector_max_samples: int, vector_layer_idx: Union[int, List[int]],
                         variants_by_mode: Optional[Dict[str, Set[str]]] = None,
                         fewshot_k: int = 2, fewshot_frac: float = 0.1, seed: int = 42,
                         malt_prefix: str = "malt_v1"):
    """
    Build steering vectors with configurable prompt variants.
    Uses MMLU-Pro train splits for vector questions and MALT data for few-shot prompts.

    Returns:
        dict[mode][variant][role] -> path
    """
    import random

    roles = ["generator", "verifier", "refiner"]
    vector_dir.mkdir(parents=True, exist_ok=True)

    if variants_by_mode is None:
        variants_by_mode = {mode: {"prompt"} for mode in modes}

    minimal_hint_suffix = {
        "generator": "End your answer with: ANSWER: <answer>.",
        "refiner": "End your answer with: ANSWER: <answer>.",
        "verifier": "Respond with:\nVERDICT: CORRECT or INCORRECT\nCRITIQUE: <brief justification>.",
    }

    def apply_minimal_hint(role: str, text: str) -> str:
        hint = minimal_hint_suffix.get(role)
        if not hint:
            return text
        if hint.strip() in text:
            return text
        joiner = "" if text.endswith("\n") or not text else "\n"
        return f"{text}{joiner}{hint}"

    def variant_components(role: str, variant: str, base_text: str, base_is_raw_task: bool) -> Tuple[Optional[str], str]:
        if variant == "prompt":
            return ROLE_SYSTEM_PROMPTS[role], base_text
        if variant == "task_only":
            prompt = ROLE_SYSTEM_PROMPTS[role]
            if prompt and prompt in base_text:
                user = base_text
            elif prompt:
                prefix = f"{prompt}\n\n" if base_text else prompt
                user = f"{prefix}{base_text}"
            else:
                user = base_text
            return None, user
        if variant == "minimal_hints":
            user = apply_minimal_hint(role, base_text if base_text else "")
            return None, user
        raise ValueError(f"Unsupported variant '{variant}'")

    def build_chat(system_msg: Optional[str], user_msg: Optional[str]) -> str:
        sys = system_msg if system_msg else None
        usr = user_msg if user_msg else None
        return collector._chat(tok, sys, usr)

    # --- Load raw questions ---
    if not dataset_tag or not dataset_tag.startswith("mmlu_pro:"):
        raise ValueError(f"Unsupported dataset_tag '{dataset_tag}'. Use 'mmlu_pro:<category>'.")
    category = dataset_tag.split(":", 1)[1]
    from data.datasets_mmlu_pro import load_mmlu_pro_split
    rows = load_mmlu_pro_split(data_dir, category, split="train")
    raw_questions = [r["question"] for r in rows]

    # --- Shuffle & split into two pools ---
    random.seed(seed)
    random.shuffle(raw_questions)

    n = len(raw_questions)
    n_fewshot = max(1, int(fewshot_frac * max(n, 1)))  # reserve fraction for few-shot pool
    fewshot_questions = raw_questions[:n_fewshot]
    vector_questions = raw_questions[n_fewshot:] if n > n_fewshot else raw_questions

    print(f"[INFO] Total questions={n}, few-shot pool={len(fewshot_questions)}, vector pool={len(vector_questions)}")

    # --- Collector setup ---
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    collector = SteeringVectorCollector(base_model)
    cfg = SteeringVectorConfig(
        layer_idx=vector_layer_idx,
        alpha=vector_alpha,
        max_samples=vector_max_samples,
        max_length=1024,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    vectors_by_mode: Dict[str, Dict[str, Dict[str, str]]] = {}

    for mode in modes:
        variant_set = variants_by_mode.get(mode, {"prompt"}) or {"prompt"}
        variant_map: Dict[str, Dict[str, str]] = {}
        for variant in sorted(variant_set):
            paths: Dict[str, str] = {}
            suffix = "wi" if mode == "with_inputs" else ("so" if mode == "system_only" else "fs")
            for role in roles:
                path = vector_dir / f"{role}_{suffix}_{variant}.pt"
                if not path.exists():
                    print(f"[INFO] Collecting steering vector for role={role}, mode={mode}, variant={variant} …")

                    if mode == "with_inputs":
                        pairs = []
                        for q in vector_questions[:cfg.max_samples]:
                            baseline = build_chat(None, q)
                            sys_msg, user_msg = variant_components(role, variant, q, True)
                            yes_text = build_chat(sys_msg, user_msg)
                            pairs.append((baseline, yes_text))
                        vec = collector.collect(
                            ROLE_SYSTEM_PROMPTS[role],
                            mode="custom_pairs",
                            custom_pairs=pairs if pairs else [(build_chat(None, ""), build_chat(*variant_components(role, variant, "", True)))]
                            ,
                            cfg=cfg
                        )

                    elif mode == "fewshot":
                        malt_role = f"{role}_{malt_prefix}"
                        malt_path = data_dir / f"{malt_role}.jsonl"
                        if not malt_path.exists():
                            raise FileNotFoundError(
                                f"MALT data not found for few-shot prompts: {malt_path}. "
                                "Run with --run_malt_generation."
                            )
                        pool = _load_fewshot_pool_for_vectors(data_dir, malt_role)

                        def sample_k(k):
                            return random.sample(pool, min(k, len(pool))) if pool else []

                        pairs = []
                        for q in fewshot_questions[:cfg.max_samples]:
                            baseline = build_chat(None, q)
                            fs_user = _build_fewshot_user(role, sample_k(fewshot_k), q)
                            sys_msg, user_msg = variant_components(role, variant, fs_user, False)
                            yes_text = build_chat(sys_msg, user_msg)
                            pairs.append((baseline, yes_text))
                        vec = collector.collect(
                            ROLE_SYSTEM_PROMPTS[role],
                            mode="custom_pairs",
                            custom_pairs=pairs if pairs else [(build_chat(None, ""), build_chat(*variant_components(role, variant, "", True)))]
                            ,
                            cfg=cfg
                        )

                    elif mode == "system_only":
                        baseline = build_chat(None, None)
                        sys_msg, user_msg = variant_components(role, variant, "", True)
                        yes_text = build_chat(sys_msg, user_msg)
                        vec = collector.collect(
                            ROLE_SYSTEM_PROMPTS[role],
                            mode="custom_pairs",
                            custom_pairs=[(baseline, yes_text)] * max(1, cfg.max_samples),
                            cfg=cfg
                        )
                    else:
                        raise ValueError(f"Unsupported vector mode: {mode}")

                    torch.save(vec, path)
                else:
                    print(f"[INFO] Using cached steering vector for role={role}, mode={mode}, variant={variant}")

                paths[role] = str(path)
            variant_map[variant] = paths
        vectors_by_mode[mode] = variant_map

    return vectors_by_mode


def normalize_setups(setups_list):
    # Allow comma-separated and space-separated values
    expanded = []
    for item in setups_list:
        expanded.extend([s for s in item.split(",") if s])
    return expanded


# ----- map eval dataset_tag -> TRAIN KEY (used for dirs & HF names) -----
def train_key_from_eval_tag(dataset_tag: str) -> str:
    dt = dataset_tag.lower()
    if not dt.startswith("mmlu_pro:"):
        raise ValueError(f"Unsupported dataset tag '{dataset_tag}'. Use 'mmlu_pro:<category>'.")
    return dt

# ----- translate TRAIN KEY -> dataset_builder tuple (dataset_name, config, split) -----
def builder_tuple_from_train_key(train_key: str):
    """
    Return a list of (dataset_name, config, split) for the training builder.
    dataset_name must be supported by build_role_datasets (see note below).
    """
    tk = train_key.lower()
    if not tk.startswith("mmlu_pro:"):
        raise ValueError(f"Unsupported train key: {train_key}. Use 'mmlu_pro:<category>'.")
    return [("mmlu_pro", tk.split(":", 1)[1], "train")]

def results_filename(setup: str, train_dataset: Optional[str], eval_dataset: Optional[str]) -> str:
    train_tag = (train_dataset or "unknown").strip()
    eval_tag = (eval_dataset or "unknown").strip()
    return f"{setup}__train={train_tag}__eval={eval_tag}.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setups",
        nargs="+",
        required=True,
        help="One or more setups to run (space or comma separated). Choices: baseline, prompt, prompt_fs, "
             "vector, vector_np, vector_mh, vector_fs, vector_fs_np, vector_fs_mh, "
             "vector_so, vector_so_np, vector_so_mh, vector_dom_add, vector_dom_weighted, vector_dom_orthogonal, "
             "lora, lora_np, lora_mh"
    )
    parser.add_argument(
        "--train_mode",
        choices=["per_dataset", "union"],
        default="per_dataset",
        help="per_dataset: train LoRA/vectors separately for each dataset; union: train once on the union and reuse."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=None,
        help="Datasets to use for training resources. Default: derived from --eval_datasets (per-dataset) or all evals (union). "
             "Format: mmlu_pro:<category>"
    )
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    #parser.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--eval_datasets", nargs="+", default=["mmlu_pro:law"],
                        help="Datasets to evaluate, e.g. mmlu_pro:law")
    parser.add_argument("--max_eval_data", type=int, default=100)
    parser.add_argument("--max_train_per_dataset", type=int, default=10)
    parser.add_argument("--fewshot_k", type=int, default=3)
    parser.add_argument("--fewshot_seed", type=int, default=1234,
                        help="Random seed for sampling few-shot exemplars during prompting.")

    parser.add_argument("--vector_alpha", type=float, default=2.5)
    parser.add_argument("--vector_max_samples", type=int, default=300)
    parser.add_argument("--vector_layer_idx", nargs="+", type=int, default=[18, 22, 26])
    parser.add_argument("--domain_vector_targets", nargs="+", default=[],
                        help="Dataset tags that should receive domain vectors (e.g., mmlu_pro:law). "
                             "Must match evaluation tags when composing.")
    parser.add_argument("--domain_prompt_template", type=str,
                        default="You are a domain expert for {domain}. Think step-by-step and include all reasoning.",
                        help="Template to build the system prompt when collecting domain vectors.")
    parser.add_argument("--domain_max_examples", type=int, default=300,
                        help="Maximum examples per domain tag to seed domain vector collection.")
    parser.add_argument("--domain_combine_strategies", nargs="+", default=[],
                        help="One or more composition strategies to blend role and domain vectors. "
                             "Choices: add, weighted, orthogonal.")
    parser.add_argument("--domain_weight", type=float, default=0.2,
                        help="Weight for the domain component when using the 'weighted' combination strategy.")
    parser.add_argument("--domain_base_mode", choices=["with_inputs", "system_only"], default="system_only",
                        help="Which role vectors to combine with domain vectors.")
    parser.add_argument("--domain_target_alpha", type=float, default=None,
                        help="Optional alpha magnitude applied after combining vectors. Defaults to --vector_alpha.")

    parser.add_argument('--run_malt_generation', action='store_true')
    parser.add_argument('--train_lora_from_malt', action='store_true')
    parser.add_argument('--malt_search_width', type=int, default=2)
    parser.add_argument('--malt_max_depth', type=int, default=2)
    parser.add_argument('--malt_seed', type=int, default=42)
    parser.add_argument("--malt_prefix", type=str, default="malt_v1",
                        help="Filename suffix for MALT jsonls: generator_<prefix>.jsonl etc.")

    args = parser.parse_args()

    # Helper: ensure MALT data exists (reads generator.jsonl if present; else seeds from GSM8K test)
    def ensure_malt_data(
            base_model,
            data_dir: Path,
            prefix: str,
            search_width: int,
            max_depth: int,
            seed: int = 42,
            max_seed_examples: int = 200,
            eval_dataset_tags: Optional[List[str]] = None,
            split_root: Optional[Path] = None,
    ):
        """
        Ensure MALT rollouts exist. No dependency on classic generator/verifier/refiner jsonls.
        Seed questions come from (in order):
          1) data_dir/seed_malt_questions.jsonl (lines: {"question": "...", "label": "..."}), if present
          2) build_eval_set() over the provided eval_dataset_tags (round-robin) if given
             (fields mapped to {"raw_question": question, "gold": label})
        """
        g_path = data_dir / f"generator_{prefix}.jsonl"
        v_path = data_dir / f"verifier_{prefix}.jsonl"
        r_path = data_dir / f"refiner_{prefix}.jsonl"
        if g_path.exists() and v_path.exists() and r_path.exists():
            print(f"[INFO] Using cached MALT data: {g_path}, {v_path}, {r_path}")
            return

        # Build seed list
        seeds: List[Dict[str, str]] = []

        # (1) Local seed files (optional, dataset-aware)
        seed_files: List[Path] = []
        if split_root and split_root.exists():
            tags_for_seeds = eval_dataset_tags or []
            for tag in tags_for_seeds:
                domain = tag.split(":", 1)[1] if ":" in tag else tag
                candidates = []
                candidates.append(domain)
                candidates.append(domain.replace("_", " "))
                candidates.append(domain.replace("_", "-"))
                # Drop duplicates while preserving order
                seen = set()
                unique_candidates = []
                for cand in candidates:
                    if cand not in seen:
                        seen.add(cand)
                        unique_candidates.append(cand)
                for cand in unique_candidates:
                    path = split_root / f"{cand}_train.jsonl"
                    if path.exists():
                        seed_files.append(path)
                        break

        if seed_files:
            for seed_file in seed_files:
                print(f"[INFO] Seeding MALT from {seed_file}")
                with seed_file.open() as f:
                    for line in f:
                        if len(seeds) >= max_seed_examples:
                            break
                        rec = json.loads(line)
                        q = rec.get("question") or rec.get("raw_question") or ""
                        y = rec.get("label") or rec.get("gold") or rec.get("gold_letter") or ""
                        if q:
                            seeds.append({"raw_question": q, "gold": y})
                if len(seeds) >= max_seed_examples:
                    break

        # (2) Otherwise, use eval datasets (round-robin)
        if not seeds:
            tags = (eval_dataset_tags or [])
            if not tags:
                # Sensible default if nothing provided: small GSM8K test slice
                tags = ["mmlu_pro:law"]
            print(f"[INFO] Seeding MALT from train datasets: {tags}")
            per_tag_cap = max(1, max_seed_examples // max(1, len(tags)))
            pools: List[List[Dict[str, str]]] = []
            for tag in tags:
                rows = build_eval_set(tag, per_tag_cap)  # existing helper returns [{"question":..., "label":...}, ...]
                pools.append(rows)

            # round-robin merge until we hit max_seed_examples
            i = 0
            while len(seeds) < max_seed_examples and any(pools):
                for p in list(pools):
                    if not p:
                        pools.remove(p)
                        continue
                    row = p.pop(0)
                    seeds.append({"raw_question": row["question"], "gold": row.get("label", "")})
                    if len(seeds) >= max_seed_examples:
                        break
                i += 1

        if not seeds:
            raise RuntimeError(
                "ensure_malt_data: no seed questions could be constructed. "
                "Provide eval datasets or a local seed_malt_questions.jsonl."
            )

        print(f"[INFO] MALT seeding with {len(seeds)} questions")

        # Generate rollouts
        malt_gen = MALTDataGenerator(
            base_model,
            data_dir,
            search_width=search_width,
            max_depth=max_depth,
            seed=seed,
        )
        malt_gen.generate_and_save(seeds, out_prefix=prefix)

    model_size = re.findall(r"meta-llama/Llama-\d.\d-(\d+)B", args.base_model)[0]
    vector_layer_arg = args.vector_layer_idx
    if isinstance(vector_layer_arg, (list, tuple)):
        host_layer_idx = vector_layer_arg[0] if vector_layer_arg else 20
    else:
        host_layer_idx = vector_layer_arg or 20

    setups = normalize_setups(args.setups)
    valid = {
        "baseline", "prompt", "prompt_fs",
        "vector", "vector_np", "vector_mh",
        "vector_fs", "vector_fs_np", "vector_fs_mh",
        "vector_so", "vector_so_np", "vector_so_mh",
        "lora", "lora_np", "lora_mh",
        "vector_dom_add", "vector_dom_weighted", "vector_dom_orthogonal"
    }
    unknown = [s for s in setups if s not in valid]
    split_and_cache_mmlu_pro(
        Path(f"experiments_{model_size}B") / "mmlu_pro_splits",
        seed=42,
        train_samples=args.max_train_per_dataset,
        force=False,
    )
    eval_dataset_tags = normalize_setups(args.eval_datasets)
    if unknown:
        raise ValueError(f"Unknown setups: {unknown}. Valid: {sorted(valid)}")
    domain_setups = {s for s in setups if s.startswith("vector_dom_")}
    inferred_domain_strategies = sorted({s[len("vector_dom_"):].lower() for s in domain_setups})
    domain_strategies = [s.lower() for s in args.domain_combine_strategies] if args.domain_combine_strategies else []
    if not domain_strategies and inferred_domain_strategies:
        domain_strategies = inferred_domain_strategies
    for train_dataset in args.train_datasets:
        train_tag = (train_dataset or "unknown").strip()
        base_model = args.base_model
        work_dir = Path(f"experiments_{model_size}B") / train_dataset
        data_dir, lora_dir, vector_dir = work_dir / "data", work_dir / "lora", work_dir / "vectors"
        results_dir = Path(f"results_{model_size}B")
        results_dir.mkdir(exist_ok=True, parents=True)
        for d in [data_dir, lora_dir, vector_dir]:
            d.mkdir(parents=True, exist_ok=True)

        current_train_key = train_key_from_eval_tag(train_dataset)

        seed_tags = [tag for tag in eval_dataset_tags if train_key_from_eval_tag(tag) == current_train_key]
        if not seed_tags:
            seed_tags = eval_dataset_tags
        # If the train dataset explicitly requests a train split (suffix "_train"),
        # seed MALT from that tag instead of defaulting to the eval/test tags.
        if train_dataset and train_dataset.lower().endswith("_train"):
            preferred_tag = train_dataset.lower()
            seed_tags = [preferred_tag]

        if args.run_malt_generation:
            malt_t0 = time.perf_counter()
            ensure_malt_data(base_model, data_dir, args.malt_prefix,
                             args.malt_search_width, args.malt_max_depth,
                             args.malt_seed, max_seed_examples=args.max_train_per_dataset,
                             eval_dataset_tags=seed_tags,
                             split_root=Path(f"experiments_{model_size}B") / "mmlu_pro_splits")
            malt_elapsed = time.perf_counter() - malt_t0
            print(f"[TIME] MALT data generation ({train_dataset}): {malt_elapsed:.2f}s")

            # Optional: immediately train from MALT
            if args.train_lora_from_malt:
                trainer = RoleTrainer(base_model)
                lora_t0 = time.perf_counter()
                print("[INFO] Training LoRA adapters from MALT datasets…")
                trainer.train_role_from_malt_dataset('generator', data_dir / f'generator_{args.malt_prefix}.jsonl',
                                                     lora_dir / 'generator')
                trainer.train_role_from_malt_dataset('verifier', data_dir / f'verifier_{args.malt_prefix}.jsonl',
                                                     lora_dir / 'verifier')
                trainer.train_role_from_malt_dataset('refiner', data_dir / f'refiner_{args.malt_prefix}.jsonl',
                                                     lora_dir / 'refiner')
                lora_elapsed = time.perf_counter() - lora_t0
                print(f"[TIME] LoRA training from MALT ({train_dataset}): {lora_elapsed:.2f}s")

        # Determine which datasets to use for training resources (metadata only; kept for parity)
        if train_dataset is None:
            if args.train_mode == "per_dataset":
                train_keys = sorted({train_key_from_eval_tag(t) for t in eval_dataset_tags})
            else:
                train_keys = ["_union"]
        else:
            train_keys = [train_dataset.lower()]
            if args.train_mode == "union" and len(train_keys) > 1:
                train_keys = ["_union"]
        print(f"[INFO] Training mode: {args.train_mode}, train_keys={train_keys}")

        results_dir.mkdir(exist_ok=True, parents=True)
        summary_csv = results_dir / "summary.csv"

        # --- Shared resources only if needed ---
        needs_lora = any(s.startswith("lora") for s in setups)
        modes_needed = set()
        for s in setups:
            if s.startswith("vector"):
                if s.startswith("vector_fs"):
                    modes_needed.add("fewshot")
                elif s.startswith("vector_so"):
                    modes_needed.add("system_only")
                else:
                    modes_needed.add("with_inputs")
        if args.domain_vector_targets and domain_strategies:
            modes_needed.add(args.domain_base_mode)

        variant_requirements: Dict[str, Set[str]] = {mode: set() for mode in modes_needed}
        for setup in setups:
            if not setup.startswith("vector"):
                continue
            if setup.startswith("vector_fs"):
                mode = "fewshot"
            elif setup.startswith("vector_so"):
                mode = "system_only"
            else:
                mode = "with_inputs"
            if mode not in modes_needed:
                continue
            variant = "prompt"
            if setup.endswith("_np"):
                variant = "task_only"
            elif setup.endswith("_mh"):
                variant = "minimal_hints"
            variant_requirements.setdefault(mode, set()).add(variant)
        if domain_setups:
            variant_requirements.setdefault(args.domain_base_mode, set()).add("prompt")
        for mode in modes_needed:
            if not variant_requirements.get(mode):
                variant_requirements.setdefault(mode, set()).add("prompt")

        lora_adapters = None
        if needs_lora:
            lora_t0 = time.perf_counter()
            lora_adapters = ensure_lora(base_model, data_dir, lora_dir)
            lora_elapsed = time.perf_counter() - lora_t0
            print(f"[TIME] LoRA training/check ({train_dataset}): {lora_elapsed:.2f}s")

        # Steering vectors
        vectors_by_mode = {}
        if modes_needed:
            vector_t0 = time.perf_counter()
            vectors_by_mode = ensure_vectors_multi(
                base_model, data_dir, vector_dir,
                modes=modes_needed,
                dataset_tag=current_train_key,
                vector_alpha=args.vector_alpha,
                vector_max_samples=args.vector_max_samples,
                vector_layer_idx=args.vector_layer_idx,
                variants_by_mode=variant_requirements,
                fewshot_k=args.fewshot_k,
                malt_prefix=args.malt_prefix,
            )
            vector_elapsed = time.perf_counter() - vector_t0
            print(f"[TIME] Steering vector collection ({train_dataset}): {vector_elapsed:.2f}s")

        domain_vectors = {}
        combined_vectors = {}
        if args.domain_vector_targets:
            domain_alpha = args.vector_alpha
            domain_vectors = ensure_domain_vectors(
                base_model=base_model,
                domain_tags=args.domain_vector_targets,
                out_dir=vector_dir / "domain_vectors",
                alpha=domain_alpha,
                max_samples=args.vector_max_samples,
                layer_idx=args.vector_layer_idx,
                prompt_template=args.domain_prompt_template,
                max_examples_per_domain=args.domain_max_examples,
                collector_cfg_overrides={
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "max_length": 1024,
                },
            )

        if domain_vectors and domain_strategies:
            role_variant_map = vectors_by_mode.get(args.domain_base_mode, {})
            if not role_variant_map:
                print(f"[WARN] Domain combinations requested but no base role vectors are available for mode={args.domain_base_mode}.")
            else:
                role_vectors = role_variant_map.get("prompt")
                if not role_vectors:
                    role_vectors = next((v for v in role_variant_map.values() if v), {})
                if not role_vectors:
                    print("[WARN] Unable to locate role vectors for domain combinations.")
                else:
                    target_alpha = args.domain_target_alpha if args.domain_target_alpha is not None else args.vector_alpha
                    combined_vectors = ensure_combined_vectors(
                        base_role_vectors=role_vectors,
                        domain_vectors=domain_vectors,
                        out_dir=vector_dir / "combined_vectors",
                        strategies=domain_strategies,
                        weight=args.domain_weight,
                        target_alpha=target_alpha,
                    )

        combined_logs = []

        def _resolve_domain_entry(tag: str, combined_map: Dict[str, Dict[str, Dict[str, str]]]):
            """
            Try multiple fallbacks to locate a domain entry for a given eval tag.
            """
            candidates = []
            candidates.append(tag)
            candidates.append(sanitize_domain_tag(tag))

            try:
                tk = train_key_from_eval_tag(tag)
            except Exception:
                tk = None
            if tk:
                candidates.extend([tk, f"{tk}_train", sanitize_domain_tag(tk), sanitize_domain_tag(f"{tk}_train")])

            for suffix in ("_test", "_val", "_validation"):
                if tag.endswith(suffix):
                    base = tag[: -len(suffix)]
                    candidates.extend([base, f"{base}_train", sanitize_domain_tag(base), sanitize_domain_tag(f"{base}_train")])

            for cand in candidates:
                if cand in combined_map:
                    return combined_map[cand]
            return None

        def get_vector_paths(mode: str, variant: str) -> Dict[str, str]:
            mode_map = vectors_by_mode.get(mode, {})
            if not mode_map:
                return {}
            if variant in mode_map and mode_map[variant]:
                return mode_map[variant]
            if "prompt" in mode_map and mode_map["prompt"]:
                return mode_map["prompt"]
            for paths in mode_map.values():
                if paths:
                    return paths
            return {}

        for setup in setups:
            if setup.startswith("vector_dom_"):
                strategy = setup[len("vector_dom_"):].lower()
                if not combined_vectors:
                    print(f"[WARN] No combined domain vectors available; skipping setup '{setup}'.")
                    continue
                for dataset_tag in eval_dataset_tags:
                    domain_entry = _resolve_domain_entry(dataset_tag, combined_vectors)
                    if not domain_entry or strategy not in domain_entry:
                        print(f"[WARN] Missing domain combination '{strategy}' for dataset '{dataset_tag}'. Skipping.")
                        continue

                    spaths = domain_entry[strategy]
                    evaluator = run_config(
                        base_model, "vector", data_dir, args.max_eval_data,
                        steering_paths=spaths, use_prompts=True, minimal_hints=False,
                        verifier_decision="gen", layer_idx=host_layer_idx
                    )

                    dataset_eval = build_eval_set(dataset_tag, args.max_eval_data)
                    print(f"[INFO] Evaluation set size: {len(dataset_eval)} ({dataset_tag})")

                    t0 = time.perf_counter()
                    acc, logs = evaluator.evaluate(dataset_eval)
                    elapsed = time.perf_counter() - t0
                    print(f"[TIME] {setup} on {dataset_tag}: {elapsed:.2f}s | acc={acc:.4f}")

                    annotated = []
                    for item in logs:
                        gold = item.get("label") or item.get("gold") or ""
                        new_item = dict(item)
                        new_item["train_dataset"] = train_tag
                        new_item["dataset"] = dataset_tag
                        new_item["condition"] = setup
                        annotated.append(new_item)

                    print(f"[TIME] {setup} on {dataset_tag}: {elapsed:.2f}s | acc(raw)={acc:.4f}")

                    per_file = results_dir / results_filename(setup, train_tag, dataset_tag)
                    with open(per_file, "w") as f:
                        for rec in annotated:
                            f.write(json.dumps(rec) + "\n")
                    print(f"[INFO] Saved evaluation logs to {per_file}")

                    write_summary_row(summary_csv, setup, dataset_tag, len(dataset_eval), elapsed)
                    combined_logs.extend(annotated)
                    evaluator.unload()
                continue

            # Map setup -> condition + params
            if setup == "baseline":
                evaluator = run_config(base_model, "baseline", data_dir, args.max_eval_data,
                                       verifier_decision="gen", layer_idx=host_layer_idx)

            elif setup == "prompt":
                evaluator = run_config(base_model, "prompt", data_dir, args.max_eval_data,
                                       verifier_decision="gen", layer_idx=host_layer_idx)

            elif setup == "prompt_fs":
                evaluator = run_config(base_model, "prompt_fs", data_dir, args.max_eval_data,
                                       fewshot_k=args.fewshot_k, verifier_decision="gen",
                                       layer_idx=host_layer_idx, fewshot_seed=args.fewshot_seed)

            elif setup in ("vector", "vector_np", "vector_mh"):
                variant = "prompt"
                if setup.endswith("_np"):
                    variant = "task_only"
                elif setup.endswith("_mh"):
                    variant = "minimal_hints"
                spaths = get_vector_paths("with_inputs", variant)
                if not spaths:
                    print(f"[WARN] Missing steering vectors for mode=with_inputs, variant={variant}; skipping.")
                    continue
                use_prompts = (setup == "vector")
                minimal_hints = (setup == "vector_mh")
                evaluator = run_config(base_model, "vector", data_dir, args.max_eval_data,
                                       steering_paths=spaths, use_prompts=use_prompts, minimal_hints=minimal_hints,
                                       verifier_decision="gen", layer_idx=host_layer_idx)

            elif setup in ("vector_fs", "vector_fs_np", "vector_fs_mh"):
                variant = "prompt"
                if setup.endswith("_np"):
                    variant = "task_only"
                elif setup.endswith("_mh"):
                    variant = "minimal_hints"
                spaths = get_vector_paths("fewshot", variant)
                if not spaths:
                    print(f"[WARN] Missing steering vectors for mode=fewshot, variant={variant}; skipping.")
                    continue
                use_prompts = (setup == "vector_fs")
                minimal_hints = (setup == "vector_fs_mh")
                evaluator = run_config(base_model, "vector", data_dir, args.max_eval_data,
                                       steering_paths=spaths, use_prompts=use_prompts, minimal_hints=minimal_hints,
                                       verifier_decision="gen", layer_idx=host_layer_idx)

            elif setup in ("vector_so", "vector_so_np", "vector_so_mh"):
                variant = "prompt"
                if setup.endswith("_np"):
                    variant = "task_only"
                elif setup.endswith("_mh"):
                    variant = "minimal_hints"
                spaths = get_vector_paths("system_only", variant)
                if not spaths:
                    print(f"[WARN] Missing steering vectors for mode=system_only, variant={variant}; skipping.")
                    continue
                use_prompts = (setup == "vector_so")
                minimal_hints = (setup == "vector_so_mh")
                evaluator = run_config(base_model, "vector", data_dir, args.max_eval_data,
                                       steering_paths=spaths, use_prompts=use_prompts, minimal_hints=minimal_hints,
                                       verifier_decision="gen", layer_idx=host_layer_idx)

            elif setup in ("lora", "lora_np", "lora_mh"):
                use_prompts = (setup == "lora")
                minimal_hints = (setup == "lora_mh")
                evaluator = run_config(base_model, "lora", data_dir, args.max_eval_data,
                                       lora_adapters=lora_adapters, use_prompts=use_prompts,
                                       minimal_hints=minimal_hints,
                                       verifier_decision="gen", layer_idx=host_layer_idx)
            else:
                raise ValueError(f"Unsupported setup: {setup}")

            # Evaluate on each requested dataset tag
            for dataset_tag in eval_dataset_tags:
                dataset_eval = build_eval_set(dataset_tag, args.max_eval_data)
                print(f"[INFO] Evaluation set size: {len(dataset_eval)} ({dataset_tag})")

                t0 = time.perf_counter()
                acc, logs = evaluator.evaluate(dataset_eval)
                elapsed = time.perf_counter() - t0
                print(f"[TIME] {setup} on {dataset_tag}: {elapsed:.2f}s | acc={acc:.4f}")

                annotated = []
                for item in logs:
                    gold = item.get("label") or item.get("gold") or ""
                    new_item = dict(item)
                    new_item["train_dataset"] = train_tag
                    new_item["dataset"] = dataset_tag
                    new_item["condition"] = setup
                    annotated.append(new_item)

                print(f"[TIME] {setup} on {dataset_tag}: {elapsed:.2f}s | acc(raw)={acc:.4f}")

                per_file = results_dir / results_filename(setup, train_tag, dataset_tag)
                with open(per_file, "w") as f:
                    for rec in annotated:
                        f.write(json.dumps(rec) + "\n")
                print(f"[INFO] Saved evaluation logs to {per_file}")

                write_summary_row(summary_csv, setup, dataset_tag, len(dataset_eval), elapsed)
                combined_logs.extend(annotated)

            evaluator.unload()

        combined_file = results_dir / "combined_results.jsonl"
        with open(combined_file, "w") as f:
            for rec in combined_logs:
                f.write(json.dumps(rec) + "\n")
        print(f"[INFO] Saved combined evaluation logs to {combined_file}")

if __name__ == "__main__":
    main()
