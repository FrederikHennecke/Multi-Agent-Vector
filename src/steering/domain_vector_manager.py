import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
from transformers import AutoTokenizer

from data.dataset_registry import build_eval_set
from steering.vector_collector import SteeringVectorCollector, SteeringVectorConfig
from config import ROLE_SYSTEM_PROMPTS


def _sanitize_tag(tag: str) -> str:
    """Convert dataset/domain tags into safe filesystem-friendly tokens."""
    token = re.sub(r"[^a-z0-9]+", "_", tag.lower())
    return token.strip("_") or "domain"


def sanitize_domain_tag(tag: str) -> str:
    """Public helper mirroring the internal sanitiser."""
    return _sanitize_tag(tag)


def _humanize_domain(tag: str) -> str:
    """
    Produce a human-friendly domain string for prompt templates.
    Examples:
      "mmlu_pro:law" -> "law"
      "mmlu_pro:math" -> "math"
    """
    if ":" in tag:
        _, suffix = tag.split(":", 1)
    else:
        suffix = tag
    suffix = suffix.replace("_", " ").replace("-", " ").strip()
    return suffix or tag


def ensure_domain_vectors(
    base_model: str,
    domain_tags: Iterable[str],
    out_dir: Path,
    alpha: float = 3.0,
    max_samples: int = 64,
    layer_idx: Union[int, List[int]] = 20,
    prompt_template: str = "You are a domain expert for {domain}. Reason carefully and stay factual.",
    max_examples_per_domain: int = 256,
    collector_cfg_overrides: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Build steering vectors that capture domain-specific behaviour.

    Returns:
        dict[domain_tag][role] -> path_to_vector
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roles = ["generator", "verifier", "refiner"]

    tags = list(dict.fromkeys(domain_tags))  # preserve order, drop duplicates
    if not tags:
        return {}

    cfg_kwargs = {
        "layer_idx": layer_idx,
        "alpha": alpha,
        "max_samples": max_samples,
    }
    if collector_cfg_overrides:
        cfg_kwargs.update(collector_cfg_overrides)
    cfg = SteeringVectorConfig(**cfg_kwargs)

    collector = SteeringVectorCollector(base_model)
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    saved: Dict[str, Dict[str, str]] = {}
    mode = "with_inputs"

    for tag in tags:
        safe = _sanitize_tag(tag)
        saved.setdefault(tag, {})

        prompt = prompt_template.format(domain=_humanize_domain(tag))
        print(f"[INFO] Collecting domain vectors for {tag}")

        role_dir = out_dir / safe
        role_dir.mkdir(parents=True, exist_ok=True)

        examples = build_eval_set(tag, max_examples_per_domain)
        raw_inputs = [ex["question"] for ex in examples[: max_samples or len(examples)]]
        if not raw_inputs:
            raise ValueError(f"No examples available to build domain vector for tag '{tag}'.")

        for role in roles:
            vec_path = role_dir / f"{role}_{mode}.pt"
            if vec_path.exists():
                print(f"[INFO] Using cached domain vector for {tag} ({role}) @ {vec_path}")
                saved[tag][role] = str(vec_path)
                continue

            role_prompt = ROLE_SYSTEM_PROMPTS[role]
            domain_prompt = role_prompt
            if prompt:
                join = "\n\n" if role_prompt else ""
                domain_prompt = f"{role_prompt}{join}{prompt}"

            pairs = []
            for q in raw_inputs:
                base = collector._chat(tok, role_prompt, q)
                augmented = collector._chat(tok, domain_prompt, q)
                pairs.append((base, augmented))
            vec = collector.collect(
                role_prompt,
                mode="custom_pairs",
                custom_pairs=pairs,
                cfg=cfg,
            )

            torch.save(vec, vec_path)
            saved[tag][role] = str(vec_path)
            print(f"[INFO] Saved domain vector for {tag} ({role}) -> {vec_path}")

    return saved


def _combine_vectors(role_vec: torch.Tensor, domain_vec: torch.Tensor, strategy: str, weight: float = 0.5) -> torch.Tensor:
    """
    Combine role and domain vectors using the requested strategy.
    Returns a normalized tensor (unit norm) in role_vec.dtype.

    weight:
      - 'weighted': scale of domain component relative to role component.
      - 'orthogonal': scale of orthogonalized domain component relative to role component.
    """
    eps = 1e-8
    r = role_vec.detach().to(torch.float32)
    d = domain_vec.detach().to(torch.float32)

    r_norm = torch.norm(r)
    d_norm = torch.norm(d)
    if r_norm.item() <= eps or d_norm.item() <= eps:
        return role_vec.clone()

    if strategy == "add":
        combined = r + d
    elif strategy == "weighted":
        combined = r / (r_norm + eps) + weight * (d / (d_norm + eps))
    elif strategy == "orthogonal":
        proj_coeff = torch.dot(d, r) / (r_norm * r_norm + eps)
        d_orth = d - proj_coeff * r
        d_orth_norm = torch.norm(d_orth)
        if d_orth_norm.item() <= eps:
            combined = r
        else:
            combined = r / (r_norm + eps) + weight * (d_orth / (d_orth_norm + eps))
    else:
        raise ValueError(f"Unknown combination strategy '{strategy}'.")

    combined_norm = torch.norm(combined)
    if combined_norm.item() <= eps:
        return role_vec.clone()

    combined = combined / (combined_norm + eps)
    return combined.to(role_vec.dtype)


def ensure_combined_vectors(
    base_role_vectors: Dict[str, str],
    domain_vectors: Dict[str, str],
    out_dir: Path,
    strategies: Iterable[str],
    weight: float = 0.5,
    target_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Precompute and cache combined vectors per (domain, strategy, role).

    Args:
        base_role_vectors: mapping role -> path (typically role vectors from RQ1).
        domain_vectors:    mapping domain_tag -> path.
        strategies:        iterable of combination names ('add', 'weighted', 'orthogonal').
        weight:            used for 'weighted' strategy.
        target_alpha:      optional magnitude to enforce after combination.

    Returns:
        nested dict domain_tag -> strategy -> role -> path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roles = list(base_role_vectors.keys())
    if not roles or not domain_vectors:
        return {}

    strategies = list(dict.fromkeys(strategies))
    if not strategies:
        return {}

    cache: Dict[str, Dict[str, Dict[str, str]]] = {}

    for domain_tag, domain_role_paths in domain_vectors.items():
        cache.setdefault(domain_tag, {})
        dom_safe = _sanitize_tag(domain_tag)
        cache.setdefault(dom_safe, cache[domain_tag])

        for strategy in strategies:
            strategy_safe = _sanitize_tag(strategy)
            cache[domain_tag].setdefault(strategy, {})

            for role in roles:
                domain_path = domain_role_paths.get(role)
                if not domain_path:
                    continue
                role_path = base_role_vectors[role]
                role_tensor = torch.load(role_path, map_location="cpu")
                domain_tensor = torch.load(domain_path, map_location="cpu")
                combined = _combine_vectors(role_tensor, domain_tensor, strategy=strategy, weight=weight)

                if target_alpha is not None:
                    combined = combined / (combined.norm() + 1e-8) * target_alpha

                save_path = out_dir / f"{dom_safe}_{strategy_safe}_{role}.pt"
                torch.save(combined, save_path)
                cache[domain_tag][strategy][role] = str(save_path)

    return cache
