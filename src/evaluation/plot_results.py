#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluation.results_common import (
    LETTERS,
    ROLE_INTEGRITY_KEYS,
    accuracy,
    canonical_from_cache,
    compute_verifier_stats,
    domain_from_dataset,
    extract_verifier_verdict,
    final_text_from_conversation,
    first_generator_text,
    infer_eval_from_path,
    infer_metric,
    is_correct,
    load_jsonl,
    parse_train_eval_from_path,
    print_result_line,
    split_from_domain,
    split_question_stem_and_options,
    valid_role_integrity_scores,
)

sns.set_theme(style="whitegrid", font_scale=1.0)

# Match figure sizing to the LaTeX text width (acmlarge). See 00_main.log: textwidth=452.295pt.
PT_PER_INCH = 72.27
TEXT_WIDTH_PT = 452.295
TEXT_WIDTH_IN = TEXT_WIDTH_PT / PT_PER_INCH
FIG_WIDTH_FULL = TEXT_WIDTH_IN
FIG_WIDTH_WIDE = TEXT_WIDTH_IN * 0.9
FIG_WIDTH_MEDIUM = TEXT_WIDTH_IN * 0.8
FIG_WIDTH_NARROW = TEXT_WIDTH_IN * 0.6

BASE_FONT_SIZE = 9
TITLE_SIZE = 11
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9
LEGEND_TITLE_SIZE = 10
ANNOT_SIZE = 8

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_SIZE,
    }
)

SETUP_LABELS = {
    "prompt": "Prompt",
    "prompt_fs": "Prompt-FS",
    "lora": "LoRA",
    "vector": "Vector",
    "vector_so": "Vector-SO",
    "vector_fs": "Vector-FS",
    "vector_dom_add": "V-Add",
    "vector_dom_weighted": "V-Wt",
    "vector_dom_orthogonal": "V-Orth",
}
DOMAIN_LABELS = {
    "avg": "Macro Average",
    "average": "Macro Average",
}
DEFAULT_SETUP_ORDER = [
    "prompt",
    "prompt_fs",
    "lora",
    "vector",
    "vector_so",
    "vector_fs",
    "vector_dom_add",
    "vector_dom_weighted",
    "vector_dom_orthogonal",
]
DOMAIN_ORDER = [
    "biology",
    "business",
    "chemistry",
    "economics",
    "engineering",
    "health",
    "law",
    "math",
    "physics",
    "psychology",
    "avg",
]
TRAIN_FREE_SETUPS = {"prompt", "vector_so"}
BAR_PALETTE_NAME = "colorblind"
BAR_PALETTE = sns.color_palette(BAR_PALETTE_NAME, n_colors=max(10, len(DEFAULT_SETUP_ORDER)))
SETUP_COLOR_MAP = {setup: BAR_PALETTE[i] for i, setup in enumerate(DEFAULT_SETUP_ORDER)}


def scaled_figsize(base_width: float, base_height: float, target_width: float) -> tuple[float, float]:
    scale = target_width / base_width
    return target_width, base_height * scale


def setup_display_name(setup: str) -> str:
    return SETUP_LABELS.get(setup, str(setup))


def domain_display_name(domain: str) -> str:
    name = "" if domain is None else str(domain).strip()
    if not name:
        return "Unknown"
    lowered = name.lower()
    if lowered in DOMAIN_LABELS:
        return DOMAIN_LABELS[lowered]
    display = name.replace("_", " ").strip()
    if display and display != display.lower():
        return display
    return display.title()


def order_setups(setups: Iterable[str]) -> list[str]:
    seen = []
    for setup in DEFAULT_SETUP_ORDER:
        if setup in setups and setup not in seen:
            seen.append(setup)
    for setup in sorted(set(setups) - set(seen)):
        seen.append(setup)
    return seen


def order_domains(domains: Iterable[str]) -> list[str]:
    seen = []
    for domain in DOMAIN_ORDER:
        if domain in domains and domain not in seen:
            seen.append(domain)
    for domain in sorted(set(domains) - set(seen)):
        seen.append(domain)
    return seen


def palette_for_setups(setups: Iterable[str], use_labels: bool = False) -> dict[str, tuple]:
    mapping: dict[str, tuple] = {}
    extra_index = len(DEFAULT_SETUP_ORDER)
    for setup in order_setups(setups):
        color = SETUP_COLOR_MAP.get(setup)
        if color is None:
            color = BAR_PALETTE[extra_index % len(BAR_PALETTE)]
            extra_index += 1
        key = setup_display_name(setup) if use_labels else setup
        mapping[key] = color
    return mapping


def _ensure_save(save_path):
    if save_path is None:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def split_in_cross_domain_rows(
    rows: List[dict[str, Any]],
) -> tuple[List[dict[str, Any]], List[dict[str, Any]]]:
    in_rows: List[dict[str, Any]] = []
    cross_rows: List[dict[str, Any]] = []
    for row in rows:
        train = str(row.get("train_dataset", "") or "").strip()
        eval_ds = str(row.get("eval_dataset", row.get("dataset", "")) or "").strip()
        if train and eval_ds and train != eval_ds:
            cross_rows.append(row)
        else:
            in_rows.append(row)
    return in_rows, cross_rows


def split_in_cross_domain_df(
    df: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    train_col = "train_dataset" if "train_dataset" in df.columns else None
    eval_col = "eval_dataset" if "eval_dataset" in df.columns else ("dataset" if "dataset" in df.columns else None)
    if not train_col or not eval_col:
        return df.copy(), pd.DataFrame()
    train = df[train_col].fillna("").astype(str).str.strip()
    eval_ds = df[eval_col].fillna("").astype(str).str.strip()
    cross_mask = (train != "") & (train != eval_ds)
    return df.loc[~cross_mask].copy(), df.loc[cross_mask].copy()


# ---------- core summary (cached) ----------


def summarize_cached_file(
    path: Path,
    tol: float,
    example_rows: Optional[list[dict]] = None,
    role_rows: Optional[list[dict]] = None,
) -> dict[str, str | int | float | Any] | None:
    records = list(load_jsonl(path))
    if not records:
        print(f"{path.name:40} | EMPTY")
        return None

    path_train, path_eval = parse_train_eval_from_path(path)

    # Group by (condition,dataset,train_dataset) if present, else single group
    groups = defaultdict(list)
    has_keys = any(("dataset" in r or "condition" in r) for r in records)
    if has_keys:
        for r in records:
            cond = str(r.get("condition", "")).strip() or "unknown"
            ds = str(r.get("dataset", "")).strip() or (path_eval or "unknown")
            train_ds = str(r.get("train_dataset", "")).strip() or (path_train or "")
            groups[(cond, ds, train_ds)].append(r)
    else:
        eval_tag = infer_eval_from_path(path) or "unknown"
        setup_tag = path.stem.split("__", 1)[0]
        train_tag = path_train or ""
        groups[(setup_tag, eval_tag, train_tag)] = records

    summary_row = None
    for (cond, ds, train_ds), recs in groups.items():
        refs = []
        final_texts = []
        gen_first_texts = []
        verdicts = []
        answer_lengths = []
        answer_token_counts = []
        conv_lengths = []

        for r in recs:
            gold = r.get("gold") or 0
            res = str(gold).strip()
            if res in LETTERS:
                refs.append(res)
            elif res.isdigit():
                idx = int(res)
                refs.append(LETTERS[idx] if 0 <= idx < len(LETTERS) else res)
            else:
                refs.append(res)

            final_txt = final_text_from_conversation(r)
            final_texts.append(final_txt)
            clean_txt = (final_txt or "").strip()
            answer_lengths.append(len(clean_txt))
            answer_token_counts.append(len(clean_txt.split()))

            gen_first_texts.append(first_generator_text(r))
            verdicts.append(extract_verifier_verdict(r))

            conv = r.get("conversation")
            if isinstance(conv, list):
                conv_lengths.append(len(conv))
            else:
                conv_lengths.append(0)

        metric = infer_metric(ds, refs[: min(20, len(refs))])
        preds_canon: List[str] = []
        gen_first_canon: List[str] = []
        missing_final_cache = 0
        missing_gen_cache = 0
        for r, final_txt, gen_txt in zip(recs, final_texts, gen_first_texts):
            cached_final = canonical_from_cache(r.get("final_answer"), metric)
            if not cached_final:
                missing_final_cache += 1
                cached_final = canonical_from_cache(final_txt, metric)
            preds_canon.append(cached_final)

            cached_gen = canonical_from_cache(r.get("generator_answer"), metric)
            if not cached_gen:
                missing_gen_cache += 1
                cached_gen = canonical_from_cache(gen_txt, metric)
            gen_first_canon.append(cached_gen)

        if missing_final_cache:
            print(
                f"[warn] {path.name}: {missing_final_cache}/{len(recs)} missing final_answer cache; "
                "using regex fallback."
            )
        if missing_gen_cache:
            print(
                f"[warn] {path.name}: {missing_gen_cache}/{len(recs)} missing generator_answer cache; "
                "using regex fallback."
            )

        gen_first_correct = [is_correct(p, r, metric, tol) for p, r in zip(gen_first_canon, refs)]
        final_correct = [is_correct(p, r, metric, tol) for p, r in zip(preds_canon, refs)]
        acc = accuracy(preds_canon, refs, metric=metric, tol=tol)
        gen_first_acc = sum(gen_first_correct) / max(1, len(gen_first_correct))
        ver_acc, ver_prec, ver_rec = compute_verifier_stats(verdicts, gen_first_correct)
        n = len(refs)

        avg_conv_len = sum(conv_lengths) / max(1, len(conv_lengths))
        avg_answer_len = sum(answer_lengths) / max(1, len(answer_lengths))
        avg_answer_tokens = sum(answer_token_counts) / max(1, len(answer_token_counts))

        dataset_label = f"train={train_ds}__eval={ds}" if train_ds else ds
        print(
            f"{path.name:40} | setup={cond:15} dataset={dataset_label:28} "
            f"| metric={metric:7} | N={n:4d} | acc={acc:.4f} | gen_first_acc={gen_first_acc:.4f} "
            f"| avg_conv_len={avg_conv_len:.2f}"
        )

        summary_row = {
            "path": path.name,
            "setup": cond,
            "dataset": ds,
            "eval_dataset": ds,
            "train_dataset": train_ds,
            "dataset_label": dataset_label,
            "metric": metric,
            "N": n,
            "acc": acc,
            "gen_first_acc": gen_first_acc,
            "verifier_acc": ver_acc,
            "verifier_precision": ver_prec,
            "verifier_recall": ver_rec,
            "avg_conv_len": avg_conv_len,
            "avg_answer_len": avg_answer_len,
            "avg_answer_tokens": avg_answer_tokens,
        }

        if role_rows is not None:
            for rec in recs:
                scores = rec.get("role_integrity_scores")
                if valid_role_integrity_scores(scores):
                    role_rows.append(
                        {
                            "setup": cond,
                            "dataset": ds,
                            "g_reasoning": int(scores.get("g_reasoning", 0)),
                            "g_final_format": int(scores.get("g_final_format", 0)),
                            "v_verdict": int(scores.get("v_verdict", 0)),
                            "v_grounding": int(scores.get("v_grounding", 0)),
                            "r_revision": int(scores.get("r_revision", 0)),
                        }
                    )

        if example_rows is not None:
            for rec, ref, pred, gen_pred, ok_final, ok_gen, verdict, conv_len, ans_chars, ans_toks in zip(
                recs,
                refs,
                preds_canon,
                gen_first_canon,
                final_correct,
                gen_first_correct,
                verdicts,
                conv_lengths,
                answer_lengths,
                answer_token_counts,
            ):
                q_raw = rec.get("question") or rec.get("prompt") or rec.get("input") or rec.get("query") or ""
                q_text = str(q_raw).strip()
                stem, _ = split_question_stem_and_options(str(q_raw))
                question_words = len(stem.split()) if stem else 0
                question_words_full = len(q_text.split()) if q_text else 0
                tags = rec.get("question_tags")
                if not isinstance(tags, list) or not any(str(x).strip() for x in tags):
                    tags = ["other"]
                tags = [str(t).strip() for t in tags if str(t).strip()]
                primary = tags[0] if tags else "other"
                example_rows.append(
                    {
                        "path": path.name,
                        "setup": cond,
                        "dataset": ds,
                        "eval_dataset": ds,
                        "train_dataset": train_ds,
                        "dataset_label": dataset_label,
                        "domain": domain_from_dataset(ds),
                        "id": rec.get("id"),
                        "gold": ref,
                        "pred_final": pred,
                        "pred_gen": gen_pred,
                        "correct_final": bool(ok_final),
                        "correct_gen_first": bool(ok_gen),
                        "verdict": verdict,
                        "conv_len": conv_len,
                        "answer_chars": ans_chars,
                        "answer_tokens": ans_toks,
                        "question_stem": stem,
                        "question_words": question_words,
                        "question_words_full": question_words_full,
                        "question_tags": tags,
                        "primary_tag": primary,
                    }
                )

        # We only expect a single group per file, so break after the first.
        break

    return summary_row


# ---------- plotting utilities ----------


def weighted_mean(values: pd.Series, weights: Optional[pd.Series]) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(vals.mean())
    w = pd.to_numeric(weights, errors="coerce")
    dfw = pd.DataFrame({"v": vals, "w": w}).dropna()
    if dfw.empty:
        return float("nan")
    wsum = dfw["w"].sum()
    if wsum <= 0:
        return float(dfw["v"].mean())
    return float((dfw["v"] * dfw["w"]).sum() / wsum)


def clean_accuracy(df: pd.DataFrame, col: str = "accuracy") -> pd.DataFrame:
    """
    Coerce accuracy column to numeric and drop rows outside [0,1], which often come
    from bad parses (e.g., seconds leaking into the column).
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df[(df[col] >= 0) & (df[col] <= 1)]
    dropped = before - len(df)
    if dropped > 0:
        print(f"[warn] Dropped {dropped} rows with out-of-range {col}.")
    return df


def zoomed_ylim(values: Iterable[float], pad: float = 0.02, min_range: float = 0.06) -> tuple[float, float]:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if vals.empty:
        return 0.0, 1.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    span = vmax - vmin
    if span < min_range:
        mid = 0.5 * (vmin + vmax)
        vmin = mid - min_range / 2.0
        vmax = mid + min_range / 2.0
    vmin -= pad
    vmax += pad
    vmin = max(0.0, vmin)
    vmax = min(1.0, vmax)
    if vmax - vmin < min_range:
        mid = 0.5 * (vmin + vmax)
        vmin = max(0.0, mid - min_range / 2.0)
        vmax = min(1.0, mid + min_range / 2.0)
    return vmin, vmax


def plot_accuracy_bar(results, sort_by_acc=True, save_path=None):
    """
    Plot accuracy per dataset/setup as a bar chart.
    """
    df = pd.DataFrame(list(results))
    if df.empty:
        print("[warn] No results to plot in plot_accuracy_bar.")
        return

    label_col = "dataset_label" if "dataset_label" in df.columns else "dataset"
    dataset_col = "dataset" if "dataset" in df.columns else label_col
    df["domain_label"] = df[dataset_col].fillna("").astype(str).apply(domain_from_dataset)
    df.loc[df[dataset_col].fillna("").astype(str) == "avg", "domain_label"] = "average"
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")

    # Sort overall rows if requested.
    if sort_by_acc:
        df = df.sort_values("acc", ascending=False)

    # Preserve both per-dataset rows and the synthetic avg rows in one plot.
    dataset_order = list(dict.fromkeys(df["domain_label"]))
    base_width = 12.0
    base_height = max(9.0, 0.9 * len(dataset_order))
    setup_order = order_setups(df["setup"].dropna().astype(str).unique())
    palette = palette_for_setups(setup_order)
    fig_width, fig_height = scaled_figsize(base_width, base_height, FIG_WIDTH_FULL)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        df,
        x="acc",
        y="domain_label",
        hue="setup",
        order=dataset_order,
        hue_order=setup_order,
        palette=palette,
        errorbar=None,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            [setup_display_name(label) for label in labels],
            title="Method",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
        )
    ax.set_xlabel("Accuracy", fontsize=LABEL_SIZE)
    ax.set_ylabel("Domain", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_yticklabels([domain_display_name(label) for label in dataset_order], fontsize=TICK_SIZE)
    ax.set_xlim(0, 1.0)
    plt.title("Model Accuracy per Dataset/Setup", fontsize=TITLE_SIZE)

    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_domain_accuracy_spread(results_df: pd.DataFrame, save_path=None):
    """
    Box/strip plot of per-domain accuracy spread across setups (IQR + points).
    """
    if results_df is None or results_df.empty:
        print("[warn] No results available; skipping domain-spread plot.")
        return
    df = results_df.copy()
    df = clean_accuracy(df, col="acc")
    if df.empty:
        print("[warn] No valid accuracy rows for domain-spread plot.")
        return
    dataset_col = "dataset" if "dataset" in df.columns else ("eval_dataset" if "eval_dataset" in df.columns else None)
    if not dataset_col:
        print("[warn] No dataset column found for domain-spread plot.")
        return
    df["domain"] = df[dataset_col].fillna("").astype(str).apply(domain_from_dataset)
    df = df[df["domain"] != "avg"].dropna(subset=["setup", "acc"])
    if df.empty:
        print("[warn] No per-domain rows for domain-spread plot.")
        return
    setup_order = order_setups(df["setup"].dropna().astype(str).unique())
    palette = palette_for_setups(setup_order)
    base_width = max(8.0, 0.9 * len(setup_order))
    base_height = 5.5
    fig_width, fig_height = scaled_figsize(base_width, base_height, FIG_WIDTH_MEDIUM)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.boxplot(
        df,
        x="setup",
        y="acc",
        order=setup_order,
        palette=palette,
        showfliers=False,
    )
    sns.stripplot(
        df,
        x="setup",
        y="acc",
        order=setup_order,
        color="black",
        size=4,
        alpha=0.6,
        jitter=0.2,
    )
    ax.set_xticklabels([setup_display_name(s) for s in setup_order], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ymin, ymax = zoomed_ylim(df["acc"], min_range=0.12)
    ax.set_ylim(0, 1)
    ax.set_title("Per-domain accuracy spread (IQR)")
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def append_setup_averages(results: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """
    Return a new results list with one synthetic average row per setup.
    """
    averages: List[dict[str, Any]] = []
    grouped = defaultdict(list)
    for r in results:
        if not r:
            continue
        grouped[r["setup"]].append(r)

    for setup, rows in grouped.items():
        total_n = sum(float(r.get("N", 0) or 0) for r in rows)
        if total_n > 0:
            acc = sum((float(r.get("acc", 0) or 0) * float(r.get("N", 0) or 0)) for r in rows) / total_n
            conv = (
                sum((float(r.get("avg_conv_len", 0) or 0) * float(r.get("N", 0) or 0)) for r in rows) / total_n
            )
        else:
            acc = sum(float(r.get("acc", 0) or 0) for r in rows) / max(1, len(rows))
            conv = sum(float(r.get("avg_conv_len", 0) or 0) for r in rows) / max(1, len(rows))
        averages.append(
            {
                "path": f"{setup}__avg",
                "setup": setup,
                "dataset": "avg",
                "dataset_label": "avg",
                "metric": "avg",
                "N": total_n,
                "acc": acc,
                "avg_conv_len": conv,
            }
        )
    return results + averages


def filter_domain_vector_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop domain-vector variants (vector_dom_*) from a plotting dataframe.
    """
    if df is None or df.empty:
        return df
    setup_col = "setup" if "setup" in df.columns else ("method" if "method" in df.columns else None)
    if not setup_col:
        return df
    mask = ~df[setup_col].astype(str).str.startswith("vector_dom_")
    return df.loc[mask].copy()


def plot_end_to_end_accuracy(results_df: pd.DataFrame, in_domains: Iterable[str], save_path=None):
    """
    Use the accuracies computed in this script (regex/LLaMA canonicalization) instead of external summary CSVs.
    """
    df = results_df.copy()
    df = clean_accuracy(df, col="acc")
    weight_col = "N" if "N" in df.columns else ("num_examples" if "num_examples" in df.columns else None)
    if weight_col:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df["domain"] = df["dataset"].apply(domain_from_dataset)
    df["split"] = df["domain"].apply(lambda d: split_from_domain(d, in_domains))
    rows = []
    for (setup, split), g in df.groupby(["setup", "split"], dropna=True):
        weights = g[weight_col] if weight_col and weight_col in g else None
        acc = weighted_mean(g["acc"], weights)
        rows.append({"setup": setup, "split": split, "accuracy": acc})
    grouped = pd.DataFrame(rows)
    order = order_setups(grouped["setup"].dropna().astype(str).unique())
    palette = palette_for_setups(order)
    fig_width, fig_height = scaled_figsize(8.0, 5.0, FIG_WIDTH_MEDIUM)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        grouped,
        x="setup",
        y="accuracy",
        hue="setup",
        palette=palette,
        order=order,
        hue_order=order,
        legend=False,
    )
    ax.set_title("End-to-end accuracy (Prompt vs LoRA vs Vector)")
    ax.set_xticklabels([setup_display_name(s) for s in order], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ymin, ymax = zoomed_ylim(grouped["accuracy"])
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_generator_first_accuracy(results_df: pd.DataFrame, in_domains: Iterable[str], save_path=None):
    df = results_df.dropna(subset=["gen_first_acc"]).copy()
    if df.empty:
        print("[warn] No generator-first accuracy data available; skipping generator plot.")
        return
    df["gen_first_acc"] = pd.to_numeric(df["gen_first_acc"], errors="coerce")
    df = df[(df["gen_first_acc"] >= 0) & (df["gen_first_acc"] <= 1)]
    df["domain"] = df["dataset"].apply(domain_from_dataset)
    df["split"] = df["domain"].apply(lambda d: split_from_domain(d, in_domains))
    df["weight"] = pd.to_numeric(df.get("N", 1), errors="coerce")

    df["weighted_acc"] = df["gen_first_acc"] * df["weight"]
    df["weighted_sq"] = df["gen_first_acc"] * df["gen_first_acc"] * df["weight"]
    grouped = (
        df.groupby(["setup", "split"], dropna=True)
        .agg(
            weighted_sum=("weighted_acc", "sum"),
            weighted_sq=("weighted_sq", "sum"),
            weight_sum=("weight", "sum"),
        )
        .reset_index()
    )
    grouped["accuracy"] = grouped["weighted_sum"] / grouped["weight_sum"].replace(0, float("nan"))
    grouped["std"] = (
        grouped["weighted_sq"] / grouped["weight_sum"].replace(0, float("nan")) - grouped["accuracy"] ** 2
    ).clip(lower=0) ** 0.5
    order = order_setups(grouped["setup"].dropna().astype(str).unique())
    grouped["setup_label"] = grouped["setup"].apply(setup_display_name)
    order_labels = [setup_display_name(s) for s in order]
    palette = palette_for_setups(order, use_labels=True)
    grouped = grouped.set_index("setup").loc[order].reset_index()
    fig_width, fig_height = scaled_figsize(8.0, 5.0, FIG_WIDTH_MEDIUM)
    plt.figure(figsize=(fig_width, fig_height))
    colors = [palette[setup_display_name(s)] for s in order]
    ax = plt.gca()
    ax.bar(order_labels, grouped["accuracy"], color=colors, capsize=4)
    ax.set_title("Generator first-answer accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ymin, ymax = zoomed_ylim(grouped["accuracy"])
    ax.set_ylim(ymin, ymax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_verifier_metrics(results_df: pd.DataFrame, save_path=None):
    df = results_df.dropna(subset=["verifier_acc"], how="all").copy()
    if df.empty:
        print("[warn] No verifier metrics available; skipping verifier plot.")
        return
    df["verifier_acc"] = pd.to_numeric(df["verifier_acc"], errors="coerce")
    grouped = df.groupby("setup", dropna=True)["verifier_acc"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    order = order_setups(grouped["setup"].dropna().astype(str).unique())
    grouped = grouped.set_index("setup").loc[order].reset_index()
    labels = [setup_display_name(s) for s in order]
    palette = palette_for_setups(order)
    fig_width, fig_height = scaled_figsize(8.0, 5.0, FIG_WIDTH_MEDIUM)
    plt.figure(figsize=(fig_width, fig_height))
    colors = [palette[setup] for setup in order]
    ax = plt.gca()
    ax.bar(labels, grouped["mean"], color=colors, capsize=4)
    ax.set_title("Verifier accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ymin, ymax = zoomed_ylim(grouped["mean"])
    ax.set_ylim(ymin, ymax)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# ---------- Cross-domain plot ----------


def plot_cross_domain_accuracy_delta(results_df: pd.DataFrame, save_path=None):
    if results_df is None or results_df.empty:
        print("[warn] No results available; skipping cross-domain plot.")
        return
    in_df, cross_df = split_in_cross_domain_df(results_df)
    if cross_df.empty:
        print("[warn] No cross-domain runs found; skipping cross-domain plot.")
        return
    in_df = clean_accuracy(in_df, col="acc")
    cross_df = clean_accuracy(cross_df, col="acc")

    weight_col = "N" if "N" in in_df.columns else ("num_examples" if "num_examples" in in_df.columns else None)
    eval_col = "eval_dataset" if "eval_dataset" in in_df.columns else ("dataset" if "dataset" in in_df.columns else None)
    if not eval_col:
        print("[warn] No eval dataset column found; skipping cross-domain plot.")
        return

    def aggregate_by_domain(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["domain"] = df[eval_col].apply(domain_from_dataset)
        rows = []
        for (setup, domain), g in df.groupby(["setup", "domain"], dropna=True):
            weights = g[weight_col] if weight_col and weight_col in g else None
            acc = weighted_mean(g["acc"], weights)
            rows.append({"setup": setup, "domain": str(domain), "acc": acc})
        return pd.DataFrame(rows)

    in_grouped = aggregate_by_domain(in_df)
    cross_grouped = aggregate_by_domain(cross_df)
    if in_grouped.empty or cross_grouped.empty:
        print("[warn] Missing in-domain or cross-domain aggregates; skipping cross-domain plot.")
        return

    in_map = {(row.setup, row.domain): row.acc for row in in_grouped.itertuples()}
    cross_map = {(row.setup, row.domain): row.acc for row in cross_grouped.itertuples()}

    domains = order_domains(cross_grouped["domain"].unique().tolist())
    setups = order_setups(cross_grouped["setup"].unique().tolist())
    rows = []
    for domain in domains:
        for setup in setups:
            in_acc = in_map.get((setup, domain))
            cross_acc = cross_map.get((setup, domain))
            delta = None
            if in_acc is not None and cross_acc is not None:
                delta = cross_acc - in_acc
            rows.append({"setup": setup, "domain": domain, "delta": delta})

    if not rows:
        print("[warn] No cross-domain deltas computed; skipping cross-domain plot.")
        return

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="setup", columns="domain", values="delta")
    pivot = pivot.reindex(index=setups, columns=domains)
    if pivot.empty or pivot.dropna(how="all").empty:
        print("[warn] No cross-domain deltas computed; skipping cross-domain plot.")
        return
    max_abs = pivot.abs().max().max()
    if pd.isna(max_abs) or max_abs <= 0:
        max_abs = 0.05
    annot = pivot.applymap(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    mask = pivot.isna()
    base_width = max(6.5, 1.4 * len(domains))
    base_height = max(4.0, 0.9 * len(setups) + 1.4)
    fig_width, fig_height = scaled_figsize(base_width, base_height, FIG_WIDTH_NARROW)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        pivot,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="vlag",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": ANNOT_SIZE},
        cbar_kws={"label": "Accuracy change"},
    )
    ax.set_title("Cross-domain accuracy change", fontsize=TITLE_SIZE)
    ax.set_xlabel("Evaluation domain", fontsize=LABEL_SIZE)
    ax.set_ylabel("Setup", fontsize=LABEL_SIZE)
    ax.set_yticklabels([setup_display_name(s) for s in pivot.index], rotation=0, fontsize=TICK_SIZE)
    ax.set_xticklabels(
        [domain_display_name(d) for d in pivot.columns],
        rotation=25,
        ha="right",
        fontsize=TICK_SIZE,
    )
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        cbar.set_label("Accuracy change", fontsize=LABEL_SIZE)
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# ---------- Tag-based plots ----------


def _tag_order(tags: Iterable[str]) -> list[str]:
    uniq = sorted({str(t).strip() for t in tags if str(t).strip()})
    order = {
        t: i
        for i, t in enumerate(
            [
                "Definition",
                "Basic Facts & Properties",
                "Structure",
                "Processes & Causal",
                "Teleology / Purpose",
                "Algebraic",
                "Experiments",
                "Spatial / Kinematic",
                "other",
            ]
        )
    }
    return sorted(uniq, key=lambda x: order.get(x, 10_000))


def _explode_tags(ex_df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    df = ex_df.copy()
    if tag_col not in df.columns:
        return df
    df[tag_col] = df[tag_col].apply(lambda x: x if isinstance(x, list) else ([] if x is None else [x]))
    df = df.explode(tag_col)
    df[tag_col] = df[tag_col].fillna("other").astype(str).str.strip()
    df.loc[df[tag_col] == "", tag_col] = "other"
    return df


def compute_group_accuracy(ex_df: pd.DataFrame, group_cols: list[str], correct_col: str = "correct_final") -> pd.DataFrame:
    df = ex_df.copy()
    if df.empty:
        return pd.DataFrame()
    df[correct_col] = df[correct_col].astype(bool)
    grouped = df.groupby(group_cols, dropna=False).agg(
        accuracy=(correct_col, "mean"),
        n=(correct_col, "size"),
    )
    return grouped.reset_index()


def compute_group_mean(ex_df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    df = ex_df.copy()
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    grouped = df.groupby(group_cols, dropna=False).agg(
        value=(value_col, "mean"),
        n=(value_col, "count"),
    )
    return grouped.reset_index()


def plot_tag_accuracy_heatmap(
    ex_df: pd.DataFrame,
    tag_col: str,
    min_n: int = 10,
    save_path=None,
    title: str | None = None,
    height_scale: float = 0.5,
    min_height: float = 6,
):
    """
    Heatmap: setup x tag -> accuracy.
    """
    df = _explode_tags(ex_df, tag_col=tag_col)
    if df.empty or "setup" not in df.columns or tag_col not in df.columns:
        print(f"[warn] Missing data for tag heatmap ({tag_col}); skipping.")
        return
    grouped = compute_group_accuracy(df, ["setup", tag_col])
    if grouped.empty:
        print(f"[warn] No grouped accuracy rows for tag heatmap ({tag_col}); skipping.")
        return
    # Filter low-support tags.
    tag_counts = grouped.groupby(tag_col)["n"].sum()
    keep_tags = tag_counts[tag_counts >= min_n].index.tolist()
    grouped = grouped[grouped[tag_col].isin(keep_tags)]
    if grouped.empty:
        print(f"[warn] No tags meet min_n={min_n} for tag heatmap ({tag_col}); skipping.")
        return

    tags = _tag_order(grouped[tag_col].unique())
    setups = order_setups(grouped["setup"].dropna().astype(str).unique())
    pivot = grouped.pivot(index="setup", columns=tag_col, values="accuracy").reindex(index=setups, columns=tags)

    base_width = 1.4 * max(5, len(tags))
    base_height = max(min_height, height_scale * max(6, len(setups)))
    fig_width, fig_height = scaled_figsize(base_width, base_height, FIG_WIDTH_WIDE)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"fontsize": ANNOT_SIZE},
    )
    ax.set_yticklabels([setup_display_name(s) for s in pivot.index], rotation=0, fontsize=TICK_SIZE)
    ax.set_xlabel("Question tag", fontsize=LABEL_SIZE)
    ax.set_ylabel("Setup", fontsize=LABEL_SIZE)
    ax.set_title(title or f"Accuracy by setup and tag ({tag_col})", fontsize=TITLE_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=TICK_SIZE)
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_tag_counts(ex_df: pd.DataFrame, tag_col: str, save_path=None, title: str | None = None):
    """
    Bar plot: overall example count per tag.
    """
    df = _explode_tags(ex_df, tag_col=tag_col)
    if df.empty or tag_col not in df.columns:
        print(f"[warn] Missing data for tag counts ({tag_col}); skipping.")
        return
    counts = df.groupby(tag_col).size().reset_index(name="n")
    counts[tag_col] = counts[tag_col].astype(str)
    order = _tag_order(counts[tag_col].tolist())
    palette = sns.color_palette(BAR_PALETTE_NAME, n_colors=max(3, len(order)))
    fig_width, fig_height = scaled_figsize(8.0, 4.5, FIG_WIDTH_MEDIUM)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        counts,
        x=tag_col,
        y="n",
        order=order,
        hue=tag_col,
        palette=palette,
        legend=False,
    )
    ax.set_xlabel("Question tag")
    ax.set_ylabel("Count (examples)")
    ax.set_title(title or f"Tag distribution ({tag_col})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_tag_accuracy_facet(ex_df: pd.DataFrame, tag_col: str, min_n: int = 30, save_path=None):
    """
    Small multiples: one panel per tag, x=setup, y=accuracy.
    """
    df = _explode_tags(ex_df, tag_col=tag_col)
    if df.empty:
        print(f"[warn] No example rows for facet plot ({tag_col}); skipping.")
        return
    grouped = compute_group_accuracy(df, ["setup", tag_col])
    if grouped.empty:
        print(f"[warn] No grouped accuracy rows for facet plot ({tag_col}); skipping.")
        return
    tag_counts = grouped.groupby(tag_col)["n"].sum()
    keep_tags = tag_counts[tag_counts >= min_n].index.tolist()
    grouped = grouped[grouped[tag_col].isin(keep_tags)]
    if grouped.empty:
        print(f"[warn] No tags meet min_n={min_n} for facet plot ({tag_col}); skipping.")
        return

    grouped[tag_col] = grouped[tag_col].astype(str)
    tags = _tag_order(grouped[tag_col].unique())
    setups = order_setups(grouped["setup"].dropna().astype(str).unique())
    grouped["setup_label"] = grouped["setup"].apply(setup_display_name)
    order_labels = [setup_display_name(s) for s in setups]
    palette = palette_for_setups(setups, use_labels=True)
    g = sns.catplot(
        data=grouped,
        x="setup_label",
        y="accuracy",
        col=tag_col,
        col_order=tags,
        kind="bar",
        order=order_labels,
        hue="setup_label",
        hue_order=order_labels,
        col_wrap=3,
        height=3.2,
        aspect=1.1,
        sharey=True,
        palette=palette,
        legend=False,
    )
    g.set_axis_labels("Setup", "Accuracy")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
    if g._legend is not None:
        g._legend.remove()
    g.fig.suptitle(f"Accuracy by setup (faceted by {tag_col})", y=1.02)
    plt.tight_layout()
    save_path = _ensure_save(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# ---------- Optional CSV helpers ----------


def load_optional_csv(path_str: Optional[str]) -> Optional[pd.DataFrame]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        print(f"[warn] CSV not found at {p}; skipping.")
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        print(f"[warn] Failed to read {p}: {exc}")
        return None


def parse_param_map(arg: Optional[str]) -> dict[str, float]:
    if not arg:
        return {}
    try:
        return {k: float(v) for k, v in json.loads(arg).items()}
    except Exception:
        p = Path(arg)
        if p.exists():
            try:
                with p.open() as f:
                    data = json.load(f)
                return {k: float(v) for k, v in data.items()}
            except Exception as exc:
                print(f"[warn] Failed to parse parameter map from {p}: {exc}")
    return {}


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(description="Plot results from cached JSONL files.")
    ap.add_argument(
        "files",
        nargs="*",
        help="One or more .jsonl result files (glob is fine).",
        default="results_8B/*.jsonl",
    )
    ap.add_argument("--tol", type=float, default=0.0, help="Numeric tolerance (abs) for numeric tasks.")
    ap.add_argument(
        "--summary-csv",
        default="results_8B/summary.csv",
        help="CSV with aggregated accuracies (setup,dataset,accuracy,num_examples,seconds).",
    )
    ap.add_argument(
        "--in-domain",
        action="append",
        default=[],
        help="Domains considered in-domain (e.g., --in-domain math --in-domain biology).",
    )
    ap.add_argument("--save-dir", default=None, help="Directory to save figures. If not set, show interactively.")
    ap.add_argument(
        "--exclude-domain-vectors",
        action="store_true",
        help="Exclude vector_dom_* setups from plots.",
    )
    ap.add_argument(
        "--role-integrity-csv",
        default=None,
        help="Optional CSV with columns role,setup/method,score for Figure 4.",
    )
    ap.add_argument(
        "--param-map",
        default=None,
        help="JSON string or path mapping setup->additional_params for Figure 6.",
    )
    ap.add_argument(
        "--latency-csv",
        default=None,
        help="Optional CSV with columns setup/method,role,latency[,dataset] for Figure 7.",
    )
    args = ap.parse_args()

    results = []
    example_rows: list[dict] = []
    role_rows: list[dict] = []

    files = args.files if isinstance(args.files, list) else [args.files]
    for pattern in files:
        paths = list(
            map(
                Path,
                sorted(Path().glob(pattern) if any(ch in pattern for ch in "*?[]") else [pattern]),
            )
        )
        if not paths:
            print(f"No match: {pattern}", file=sys.stderr)
            continue
        for p in paths:
            if "combined_results.jsonl" in p.as_posix() or p.suffix != ".jsonl" or "mmlu_pro" not in p.as_posix():
                continue
            results.append(
                summarize_cached_file(
                    p,
                    tol=args.tol,
                    example_rows=example_rows,
                    role_rows=role_rows,
                )
            )
    results = [r for r in results if r]

    print("--------------")
    for r in sorted(results, key=lambda x: (x["dataset"], -x["acc"])):
        print_result_line(r)

    if role_rows:
        score_cols = list(ROLE_INTEGRITY_KEYS)
        grouped: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for row in role_rows:
            setup = row.get("setup", "unknown")
            counts[setup] = counts.get(setup, 0) + 1
            if setup not in grouped:
                grouped[setup] = {k: 0.0 for k in score_cols}
            for k in score_cols:
                grouped[setup][k] += float(row.get(k, 0))

        print("--------------")
        print("Role integrity (mean scores per setup):")
        for setup in sorted(grouped.keys()):
            n = counts.get(setup, 1)
            row = grouped[setup]
            means = {k: (row[k] / n) for k in score_cols}
            print(
                f"{setup:15} | N={n:4d} | G_reason={means['g_reasoning']:.2f} | "
                f"G_final={means['g_final_format']:.2f} | V_verdict={means['v_verdict']:.2f} | "
                f"V_ground={means['v_grounding']:.2f} | R_revision={means['r_revision']:.2f}"
            )

    save_dir = Path(args.save_dir) if args.save_dir else None

    summary_df = load_optional_csv(args.summary_csv)
    role_df = load_optional_csv(args.role_integrity_csv)
    latency_df = load_optional_csv(args.latency_csv)
    _ = parse_param_map(args.param_map)

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    examples_df = pd.DataFrame(example_rows) if example_rows else pd.DataFrame()

    in_domain_results, _ = split_in_cross_domain_rows(results)
    in_domain_results_df, _ = split_in_cross_domain_df(results_df)
    in_domain_examples_df, _ = split_in_cross_domain_df(examples_df)

    plot_results = in_domain_results
    plot_results_df = in_domain_results_df
    plot_examples_df = in_domain_examples_df
    plot_role_df = role_df
    plot_summary_df = summary_df
    plot_latency_df = latency_df

    if args.exclude_domain_vectors:
        plot_results = [r for r in plot_results if not str(r.get("setup", "")).startswith("vector_dom_")]
        plot_results_df = filter_domain_vector_setups(plot_results_df)
        plot_examples_df = filter_domain_vector_setups(plot_examples_df)
        plot_role_df = filter_domain_vector_setups(role_df) if role_df is not None else None
        plot_summary_df = filter_domain_vector_setups(summary_df) if summary_df is not None else None
        plot_latency_df = filter_domain_vector_setups(latency_df) if latency_df is not None else None

    if not plot_results_df.empty:
        plot_end_to_end_accuracy(
            plot_results_df,
            args.in_domain,
            save_path=(save_dir / "figure1_end_to_end.pdf") if save_dir else None,
        )
        plot_results = append_setup_averages(plot_results)
        plot_accuracy_bar(plot_results, save_path=(save_dir / "accuracy_bar.pdf") if save_dir else None)
        plot_domain_accuracy_spread(
            plot_results_df,
            save_path=(save_dir / "figure5_domain_spread.pdf") if save_dir else None,
        )
        plot_generator_first_accuracy(
            plot_results_df,
            args.in_domain,
            save_path=(save_dir / "figure2_generator_first.pdf") if save_dir else None,
        )
        plot_verifier_metrics(plot_results_df, save_path=(save_dir / "figure3_verifier.pdf") if save_dir else None)
        plot_cross_domain_accuracy_delta(
            results_df,
            save_path=(save_dir / "figure4_cross_domain.pdf") if save_dir else None,
        )

    # Tag-based plots (computed from per-question rows).
    if not plot_examples_df.empty:
        # Primary-tag plots (each question contributes once).
        plot_tag_counts(
            plot_examples_df,
            tag_col="primary_tag",
            save_path=(save_dir / "figure9_tag_counts_primary.pdf") if save_dir else None,
            title="Question-type distribution (primary tag)",
        )
        plot_tag_accuracy_heatmap(
            plot_examples_df,
            tag_col="primary_tag",
            min_n=15,
            save_path=(save_dir / "figure10_tag_heatmap_primary.pdf") if save_dir else None,
            title="Accuracy by setup and primary question tag",
            height_scale=0.8,
            min_height=8,
        )

        # All-tags plots (multi-label; questions may contribute to multiple tags).
        if "question_tags" in plot_examples_df.columns:
            plot_tag_counts(
                plot_examples_df,
                tag_col="question_tags",
                save_path=(save_dir / "figure15_tag_counts_alltags.pdf") if save_dir else None,
                title="Question-type distribution (all tags, multi-label)",
            )
            plot_tag_accuracy_heatmap(
                plot_examples_df,
                tag_col="question_tags",
                min_n=20,
                save_path=(save_dir / "figure16_tag_heatmap_alltags.pdf") if save_dir else None,
                title="Accuracy by setup and question tags (multi-label)",
            )

    if save_dir and not args.exclude_domain_vectors and not in_domain_results_df.empty:
        rq1_results = [r for r in in_domain_results if not str(r.get("setup", "")).startswith("vector_dom_")]
        rq1_results_df = filter_domain_vector_setups(in_domain_results_df)
        if not rq1_results_df.empty:
            plot_end_to_end_accuracy(
                rq1_results_df,
                args.in_domain,
                save_path=(save_dir / "figure1_end_to_end_nodom.pdf"),
            )
            rq1_results = append_setup_averages(rq1_results)
            plot_accuracy_bar(rq1_results, save_path=(save_dir / "accuracy_bar_nodom.pdf"))
            plot_domain_accuracy_spread(
                rq1_results_df,
                save_path=(save_dir / "figure5_domain_spread_nodom.pdf"),
            )
            plot_generator_first_accuracy(
                rq1_results_df,
                args.in_domain,
                save_path=(save_dir / "figure2_generator_first_nodom.pdf"),
            )
            plot_verifier_metrics(rq1_results_df, save_path=(save_dir / "figure3_verifier_nodom.pdf"))


if __name__ == "__main__":
    main()
