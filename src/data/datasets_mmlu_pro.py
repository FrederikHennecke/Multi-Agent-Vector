from __future__ import annotations
import json, hashlib, random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

LETTER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _gold_letter(idx: int) -> str:
    return LETTER[idx] if 0 <= idx < len(LETTER) else "A"

def _format_mc_question(question: str, options: List[str]) -> str:
    # A simple, consistent MC format used across your project
    lines = [question.strip()]
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}: {opt.strip()}")
    return "\n".join(lines).strip()

def list_mmlu_pro_categories(cache_dir: Path) -> List[str]:
    """If splits exist, read categories from index; else read from HF quickly (head)."""
    cache_dir = Path(cache_dir)
    index = cache_dir / "mmlu_pro_categories.json"
    if index.exists():
        return json.loads(index.read_text())
    # Fallback: scan full dataset once to collect categories
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    cats = sorted(set(rec["category"] for rec in ds))
    return cats

def split_and_cache_mmlu_pro(
    cache_dir: Path,
    seed: int = 42,
    train_samples: int = 100,
    force: bool = False,
) -> List[str]:
    """
    Deterministically split MMLU-Pro by category and write:
      {category}_train.jsonl (up to `train_samples` examples), {category}_test.jsonl (the rest)
    Each line has: {question_id, category, question, options, gold_letter, gold_index, cot_content}
    Returns sorted list of categories.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "mmlu_pro_splits.DONE"
    index  = cache_dir / "mmlu_pro_categories.json"
    if marker.exists() and index.exists() and not force:
        return json.loads(index.read_text())

    print("[MMLU-Pro] Loading dataset from HF hub (requires internet unless cached)…")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]

    # Bucket by category
    by_cat: Dict[str, List[dict]] = {}
    for rec in ds:
        by_cat.setdefault(rec["category"], []).append(rec)

    rng = random.Random(seed)
    cats = sorted(by_cat.keys())

    for cat in cats:
        items = by_cat[cat]
        # Deterministic shuffle via hash of question_id to avoid cross-run drift
        items.sort(key=lambda r: hashlib.md5(str(r["question_id"]).encode()).hexdigest())
        n = len(items)
        n_train = min(max(train_samples, 0), n)
        train = items[:n_train]
        test = items[n_train:]

        def dump(split_name: str, rows: List[dict]):
            out = cache_dir / f"{cat}_{split_name}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for r in rows:
                    q = _format_mc_question(r["question"], r["options"])
                    gold_idx = int(r["answer_index"])
                    gold = _gold_letter(gold_idx)
                    line = {
                        "question_id": r["question_id"],
                        "category": r["category"],
                        "question": q,
                        "options": r["options"],
                        "gold_letter": gold,
                        "gold_index": gold_idx,
                        "cot_content": r.get("cot_content", None),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            return out

        tr_path = dump("train", train)
        te_path = dump("test", test)
        print(f"[MMLU-Pro] {cat:30s} -> train={len(train):4d} @ {tr_path.name} | test={len(test):4d} @ {te_path.name}")

    index.write_text(json.dumps(cats, ensure_ascii=False, indent=2))
    marker.write_text("ok")
    print(f"[MMLU-Pro] Cached in {cache_dir} (categories={len(cats)})")
    return cats

def load_mmlu_pro_split(cache_dir: Path, category: str, split: str) -> List[dict]:
    """Return rows as dicts from cached split."""
    p = cache_dir.parent.parent / f"mmlu_pro_splits/{category}_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"MMLU-Pro split not found: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
    return rows

def build_eval_set_mmlu_pro(cache_dir: Path, category: str, max_eval: int = None, split: str = "test") -> List[dict]:
    """
    Return [{'question': <formatted MC string>, 'label': <gold_letter>}...]
    """
    rows = load_mmlu_pro_split(cache_dir, category, split)
    examples = [{"question": r["question"], "label": r["gold_letter"]} for r in rows]
    return examples[:max_eval] if max_eval else examples
