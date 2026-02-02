#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LETTERS = list("ABCDEFGHIJ")
MC_PATTERNS = [
    r"(?i)answer\s*:\s*([A-J])\b",
    r"(?i)\b(?:option|choice)\s*([A-J])\b",
    r"\(([A-J])\)",
]
MATH_CHARS = set("+-*/=^%<>×÷π√")


def extract_choice_letter(text: str) -> str:
    t = text or ""
    if not t.strip():
        return ""
    for pat in MC_PATTERNS:
        m = re.search(pat, t)
        if m:
            return m.group(1).upper()
    toks = re.findall(r"\b([A-J])\b", t, flags=re.IGNORECASE)
    if toks:
        return toks[-1].upper()
    toks = re.findall(r"([A-J])", t, flags=re.IGNORECASE)
    return toks[-1].upper() if toks else ""


def final_text_from_conversation(rec) -> str:
    conv = rec.get("conversation")
    if isinstance(conv, list) and all(isinstance(item, list) and len(item) == 2 for item in conv):
        for pref_role in ("REFINER", "GENERATOR"):
            for role, msg in reversed(conv):
                if role == pref_role and isinstance(msg, str) and msg.strip():
                    return msg
        try:
            return "\n".join(m for _, m in conv if isinstance(m, str))
        except Exception:
            pass
    for k in ("prediction", "pred", "output", "final"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def first_generator_text(rec) -> str:
    conv = rec.get("conversation")
    if isinstance(conv, list):
        for role, msg in conv:
            if role == "GENERATOR" and isinstance(msg, str) and msg.strip():
                return msg
    return str(rec.get("generator_pred", "") or "").strip()


def gold_letter(rec) -> str:
    gold = rec.get("gold")
    if isinstance(gold, str):
        gold = gold.strip()
        if len(gold) == 1 and gold.upper() in LETTERS:
            return gold.upper()
        if gold.isdigit():
            idx = int(gold)
            return LETTERS[idx] if 0 <= idx < len(LETTERS) else gold
    if isinstance(gold, int):
        return LETTERS[gold] if 0 <= gold < len(LETTERS) else str(gold)
    return str(gold or "").strip().upper()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize_result_file(path: Path) -> Optional[Dict]:
    recs = load_jsonl(path)
    if not recs:
        return None
    setup = str(recs[0].get("condition") or path.name.split("__", 1)[0]).strip()
    dataset = str(recs[0].get("dataset") or path.stem.split("__", 1)[-1]).strip()

    refs: List[str] = []
    preds_raw: List[str] = []
    gen_first_raw: List[str] = []
    conv_lengths: List[int] = []
    answer_chars: List[int] = []
    answer_tokens: List[int] = []

    for r in recs:
        refs.append(gold_letter(r))
        final_txt = final_text_from_conversation(r)
        preds_raw.append(r.get("final_answer") or final_txt)
        gen_first_raw.append(r.get("generator_answer") or first_generator_text(r))
        conv = r.get("conversation")
        conv_lengths.append(len(conv) if isinstance(conv, list) else 0)
        clean_txt = (final_txt or "").strip()
        answer_chars.append(len(clean_txt))
        answer_tokens.append(len(clean_txt.split()))

    preds = [extract_choice_letter(p) for p in preds_raw]
    gen_first_preds = [extract_choice_letter(p) for p in gen_first_raw]
    n = len(refs)
    acc = sum(int(p == r) for p, r in zip(preds, refs)) / max(1, n)
    gen_first_acc = sum(int(p == r) for p, r in zip(gen_first_preds, refs)) / max(1, n)

    return {
        "path": path.name,
        "setup": setup,
        "dataset": dataset,
        "N": n,
        "acc": acc,
        "gen_first_acc": gen_first_acc,
        "avg_conv_len": sum(conv_lengths) / max(1, len(conv_lengths)),
        "avg_answer_chars": sum(answer_chars) / max(1, len(answer_chars)),
        "avg_answer_tokens": sum(answer_tokens) / max(1, len(answer_tokens)),
    }


def load_results(results_dir: Path, setups: Iterable[str], datasets: Optional[Iterable[str]]) -> List[Dict]:
    wanted = {s.strip() for s in setups}
    dataset_filter = {d.strip() for d in datasets} if datasets else None
    rows: List[Dict] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        summary = summarize_result_file(path)
        if not summary:
            continue
        if dataset_filter and not any(ds in summary["dataset"] for ds in dataset_filter):
            continue
        if summary["setup"] in wanted:
            rows.append(summary)
    return rows


def category_from_dataset(dataset: str) -> str:
    return dataset.split(":", 1)[-1] if ":" in dataset else dataset


def load_dataset_features(splits_dir: Path, dataset: str) -> Optional[Dict]:
    category = category_from_dataset(dataset)
    split_path = splits_dir / f"{category}_test.jsonl"
    if not split_path.exists():
        return None

    q_chars = q_words = digit_chars = math_chars = 0
    with_digits = 0
    option_chars = option_words = 0
    option_counts: List[int] = []
    cot_nonempty = 0
    total_questions = 0

    with split_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total_questions += 1
            q_text = str(rec.get("question", "") or "")
            q_chars += len(q_text)
            q_words += len(q_text.split())
            digit_chars += sum(1 for ch in q_text if ch.isdigit())
            math_chars += sum(1 for ch in q_text if ch in MATH_CHARS)
            if any(ch.isdigit() for ch in q_text):
                with_digits += 1

            opts = rec.get("options") or []
            option_counts.append(len(opts))
            for opt in opts:
                opt_text = str(opt or "")
                option_chars += len(opt_text)
                option_words += len(opt_text.split())

            cot = rec.get("cot_content")
            if isinstance(cot, str) and cot.strip():
                cot_nonempty += 1

    if total_questions == 0:
        return None

    total_option_items = sum(option_counts)
    return {
        "dataset": dataset,
        "avg_q_chars": q_chars / total_questions,
        "avg_q_words": q_words / total_questions,
        "question_digit_frac": digit_chars / max(1, q_chars),
        "question_math_symbol_frac": math_chars / max(1, q_chars),
        "pct_questions_with_digits": with_digits / total_questions,
        "avg_option_chars": option_chars / max(1, total_option_items),
        "avg_option_words": option_words / max(1, total_option_items),
        "avg_num_options": sum(option_counts) / total_questions,
        "pct_with_cot": cot_nonempty / total_questions,
    }


def build_pairwise_rows(
    summaries: List[Dict],
    base_setup: str,
    other_setup: str,
    splits_dir: Path,
) -> List[Dict]:
    by_setup: Dict[str, Dict[str, Dict]] = {}
    for row in summaries:
        by_setup.setdefault(row["setup"], {})[row["dataset"]] = row

    paired: List[Dict] = []
    base_ds = set(by_setup.get(base_setup, {}))
    other_ds = set(by_setup.get(other_setup, {}))
    for ds in sorted(base_ds & other_ds):
        b = by_setup[base_setup][ds]
        o = by_setup[other_setup][ds]
        row: Dict[str, object] = {
            "dataset": ds,
            f"acc__{base_setup}": b["acc"],
            f"acc__{other_setup}": o["acc"],
            "delta_acc": b["acc"] - o["acc"],
            f"N__{base_setup}": b["N"],
            f"N__{other_setup}": o["N"],
            "delta_avg_conv_len": b["avg_conv_len"] - o["avg_conv_len"],
            "delta_avg_answer_tokens": b["avg_answer_tokens"] - o["avg_answer_tokens"],
            "delta_avg_answer_chars": b["avg_answer_chars"] - o["avg_answer_chars"],
            "delta_gen_first_acc": b["gen_first_acc"] - o["gen_first_acc"],
        }
        feats = load_dataset_features(splits_dir, ds)
        if feats:
            row.update(feats)
        paired.append(row)
    return paired


def safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mx, my = safe_mean(xs), safe_mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    if den == 0:
        return math.nan
    return num / den


def describe_feature_gaps(rows: List[Dict], delta_key: str = "delta_acc") -> List[Dict]:
    if not rows:
        return []
    numeric_keys = set()
    for r in rows:
        for k, v in r.items():
            if k == delta_key:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.add(k)
    pos = [r for r in rows if isinstance(r.get(delta_key), (int, float)) and r.get(delta_key, 0) > 0]
    neg = [r for r in rows if isinstance(r.get(delta_key), (int, float)) and r.get(delta_key, 0) <= 0]
    deltas_all = [float(r.get(delta_key, math.nan)) for r in rows if isinstance(r.get(delta_key), (int, float))]

    gaps: List[Dict] = []
    for key in sorted(numeric_keys):
        pos_vals = [float(r[key]) for r in pos if isinstance(r.get(key), (int, float))]
        neg_vals = [float(r[key]) for r in neg if isinstance(r.get(key), (int, float))]
        pos_mean = safe_mean(pos_vals)
        neg_mean = safe_mean(neg_vals)
        mean_gap = pos_mean - neg_mean if not (math.isnan(pos_mean) or math.isnan(neg_mean)) else math.nan

        aligned_x = []
        aligned_y = []
        for r in rows:
            dv = r.get(delta_key)
            fv = r.get(key)
            if isinstance(dv, (int, float)) and isinstance(fv, (int, float)):
                aligned_x.append(float(dv))
                aligned_y.append(float(fv))
        corr = pearson_corr(aligned_x, aligned_y)
        gaps.append(
            {
                "feature": key,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "mean_gap": mean_gap,
                "corr_with_delta": corr,
            }
        )
    gaps.sort(key=lambda r: (abs(r["mean_gap"]) if not math.isnan(r["mean_gap"]) else 0), reverse=True)
    return gaps


def fmt_value(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:0.4f}"
    return str(val)


def format_table(rows: List[Dict], cols: List[str]) -> str:
    if not rows:
        return "(no data)"
    data = []
    for r in rows:
        data.append([fmt_value(r.get(c, "")) for c in cols])
    widths = [max(len(cols[i]), max(len(row[i]) for row in data)) for i in range(len(cols))]

    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    header = "  ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    lines = [header]
    for row in data:
        parts = []
        for i, cell in enumerate(row):
            if is_number(cell):
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        lines.append("  ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze where vector beats vector_so (and why) by correlating dataset traits."
    )
    ap.add_argument("--results-dir", default="results_8B", type=Path, help="Directory with result JSONL files.")
    ap.add_argument("--splits-dir", default="experiments_8B/mmlu_pro_splits", type=Path,
                    help="Directory with cached mmlu_pro splits for feature extraction.")
    ap.add_argument("--base", default="vector", help="Primary setup to compare.")
    ap.add_argument("--other", default="vector_so", help="Baseline setup to subtract from.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Optional dataset name filters (substring match on filename).")
    args = ap.parse_args()

    summaries = load_results(args.results_dir, setups=[args.base, args.other], datasets=args.datasets)
    if not summaries:
        print("No matching result files found.")
        return

    pair_rows = build_pairwise_rows(
        summaries, base_setup=args.base, other_setup=args.other, splits_dir=args.splits_dir
    )
    if not pair_rows:
        print("No datasets found where both setups are present.")
        return

    acc_cols = [
        "dataset",
        f"acc__{args.base}",
        f"acc__{args.other}",
        "delta_acc",
        f"N__{args.base}",
        f"N__{args.other}",
    ]
    acc_view = sorted(pair_rows, key=lambda r: r["delta_acc"], reverse=True)
    print("\nAccuracy deltas (base - other):")
    print(format_table(acc_view, acc_cols))

    delta_cols = [
        "dataset",
        "delta_acc",
        "delta_avg_conv_len",
        "delta_avg_answer_tokens",
        "delta_avg_answer_chars",
        "delta_gen_first_acc",
    ]
    delta_view = sorted(pair_rows, key=lambda r: r["delta_acc"], reverse=True)
    print("\nBehavior deltas (base - other):")
    print(format_table(delta_view, delta_cols))

    feature_gaps = describe_feature_gaps(pair_rows)
    if feature_gaps:
        print("\nFeature gaps (pos=base better, neg=other better):")
        gap_cols = ["feature", "pos_mean", "neg_mean", "mean_gap", "corr_with_delta"]
        print(format_table(feature_gaps[:10], gap_cols))
    else:
        print("\nNo feature-level contrasts available (missing splits or numeric columns).")


if __name__ == "__main__":
    main()
