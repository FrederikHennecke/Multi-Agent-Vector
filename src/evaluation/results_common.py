#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
]
MC_RANGE = "ABCDEFGHIJ"

# ---------- robust numeric extraction: return LAST number normalized ----------
_SEP_CHARS = r"[,\.\s\u00A0\u2009']"  # comma, dot, space, NBSP, thin space, apostrophe
_CURRENCY = r"$\u20ac\u00a3\u00a5\u20b9"
_UNITS = r"%"
_NUM_RE = re.compile(
    rf"""
    (?:\()?\s*
    [+-]?
    [ {_CURRENCY} ]?
    \s*
    (
        (?:\d{{1,3}}(?:{_SEP_CHARS}\d{{3}})+|\d+)
        (?:[.,]\d+)?
    )
    \s*
    [ {_CURRENCY}{_UNITS} ]?
    \s*(?:\))?
    """,
    re.VERBOSE,
)


def extract_last_number(text: str) -> str:
    if not text:
        return ""
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return ""
    s = matches[-1].group(0).strip()
    neg = s.lstrip().startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("() ")
    s = re.sub(rf"[{_CURRENCY}{_UNITS}\s\u00A0\u2009]", "", s)
    s = s.replace("'", "")
    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        last_pos = max(s.rfind(","), s.rfind("."))
        last_sep = s[last_pos]
        after = len(s) - last_pos - 1
        if after in (1, 2):
            if last_sep == ",":
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            s = s.replace(",", "").replace(".", "")
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        cnt = s.count(sep)
        after = len(s.rsplit(sep, 1)[-1])
        if cnt >= 2:
            s = s.replace(sep, "")
        else:
            if after == 3:
                s = s.replace(sep, "")
            else:
                if sep == ",":
                    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.+-]", "", s)
    if s.startswith("+"):
        s = s[1:]
    if s in {"", ".", "+", "-"}:
        return ""
    if s.endswith("."):
        s = s[:-1]
    if "." in s:
        head, tail = s.split(".", 1)
        head = re.sub(r"^(-?)0+(?=\d)", r"\1", head) or ("-" if head.startswith("-") else "0")
        s = (head + "." + tail).rstrip("0").rstrip(".")
    else:
        s = re.sub(r"^(-?)0+(?=\d)", r"\1", s)
    if neg and not s.startswith("-"):
        s = "-" + s
    return s


# ---------- MC extraction ----------

def extract_choice_letter(text: str) -> str:
    """
    Recover a choice letter (A-J) from model text.
    """
    t = text or ""
    if not t.strip():
        return ""
    patterns = [
        r"(?i)answer\s*:\s*([A-J])\b",
        r"(?i)\b(?:option|choice)\s*([A-J])\b",
        r"\(([A-J])\)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1).upper()
    toks = re.findall(r"\b([A-J])\b", t, flags=re.IGNORECASE)
    if toks:
        return toks[-1].upper()
    toks = re.findall(r"([A-J])", t, flags=re.IGNORECASE)
    return toks[-1].upper() if toks else ""


# ---------- question parsing ----------

_OPTION_LINE_RE = re.compile(r"^\s*[A-J]\s*[:\)]\s*", re.IGNORECASE)


def split_question_stem_and_options(question: str) -> tuple[str, list[str]]:
    """
    Split an MMLU-style prompt into a question stem and option lines.

    The result JSONL typically stores the full prompt, including answer options:
    - Stem (free text)
    - Option lines, prefixed with "A:" .. "J:"
    """
    q = str(question or "")
    if not q.strip():
        return "", []

    stem_lines: list[str] = []
    option_lines: list[str] = []
    in_options = False
    for line in q.splitlines():
        if _OPTION_LINE_RE.match(line):
            in_options = True
        if in_options:
            if line.strip():
                option_lines.append(line.strip())
        else:
            if line.strip():
                stem_lines.append(line.strip())

    stem = " ".join(stem_lines).strip()
    return stem, option_lines


def try_float(x: str):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


# ---------- pull final text from conversation ----------

def final_text_from_conversation(rec) -> str:
    conv = rec.get("conversation")
    if isinstance(conv, list) and all(isinstance(item, list) and len(item) == 2 for item in conv):
        # Prefer the last REFINER message, else last GENERATOR, ignore VERIFIER
        for pref_role in ("REFINER", "GENERATOR"):
            for role, msg in reversed(conv):
                if role == pref_role and isinstance(msg, str) and msg.strip():
                    return msg
        # Fallback: join all messages
        try:
            return "\n".join(m for _, m in conv if isinstance(m, str))
        except Exception:
            pass
    # Fallbacks if conversation missing
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
    # fallbacks if conversation missing/empty
    return str(rec.get("generator_pred", "") or "").strip()


def extract_verifier_verdict(rec) -> Optional[str]:
    fb = rec.get("verifier_feedback")
    if isinstance(fb, list) and fb:
        first = fb[0]
        if isinstance(first, dict):
            verdict = first.get("verdict") or first.get("verdicts") or first.get("verdict_raw")
            if not verdict and first.get("raw"):
                m = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)", first["raw"], re.IGNORECASE)
                if m:
                    verdict = m.group(1)
            if verdict:
                v = str(verdict).strip().upper()
                if v in ("CORRECT", "INCORRECT"):
                    return v
    conv = rec.get("conversation")
    if isinstance(conv, list):
        for role, msg in conv:
            if role == "VERIFIER" and isinstance(msg, str):
                m = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)", msg, re.IGNORECASE)
                if m:
                    return m.group(1).upper()
    return None


ROLE_INTEGRITY_VERSION = "v1"
ROLE_INTEGRITY_SOURCE_HEURISTIC = f"heuristic_{ROLE_INTEGRITY_VERSION}"
ROLE_INTEGRITY_SOURCE_LLAMA = f"llama_{ROLE_INTEGRITY_VERSION}"
ROLE_INTEGRITY_KEYS = (
    "g_reasoning",
    "g_final_format",
    "v_verdict",
    "v_grounding",
    "r_revision",
)


def valid_role_integrity_scores(scores: Any) -> bool:
    if not isinstance(scores, dict):
        return False
    for k in ROLE_INTEGRITY_KEYS:
        if k not in scores:
            return False
        try:
            v = int(scores[k])
        except Exception:
            return False
        if v < 0 or v > 2:
            return False
    return True


def compute_verifier_stats(verdicts: List[Optional[str]], gen_correct: List[bool]):
    pairs = [(v, c) for v, c in zip(verdicts, gen_correct) if v in ("CORRECT", "INCORRECT")]
    if not pairs:
        return None, None, None
    tp = sum(1 for v, c in pairs if v == "CORRECT" and c)
    tn = sum(1 for v, c in pairs if v == "INCORRECT" and not c)
    fp = sum(1 for v, c in pairs if v == "CORRECT" and not c)
    fn = sum(1 for v, c in pairs if v == "INCORRECT" and c)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return acc, prec, rec


def infer_metric(dataset_tag: str, labels_sample) -> str:
    tag = (dataset_tag or "").lower()
    if tag.startswith(("mmlu_pro",)):
        return "mc"
    # heuristic fallback
    letters = sum(
        1
        for x in labels_sample
        if isinstance(x, str) and len(x.strip()) == 1 and x.strip().upper() in LETTERS
    )
    nums = sum(1 for x in labels_sample if try_float(x) is not None)
    if letters >= max(3, 0.6 * len(labels_sample)):
        return "mc"
    if nums >= max(3, 0.6 * len(labels_sample)):
        return "numeric"
    return "string"


def canonical_from_cache(val, metric: str) -> str:
    """
    Lightweight canonicalization that never calls the MC extractor; used when we have
    already cached the answer on disk.
    """
    if val is None:
        return ""
    if metric == "numeric":
        return extract_last_number(str(val))
    if metric == "mc":
        return extract_choice_letter(str(val))
    return str(val).strip()


def accuracy(preds, refs, metric: str, tol: float) -> float:
    correct = 0
    for p, r in zip(preds, refs):
        if metric == "numeric":
            pf, rf = try_float(p), try_float(r)
            ok = pf is not None and rf is not None and abs(pf - rf) <= tol
            if not ok:
                ok = str(p).strip() == str(r).strip()
        else:
            ok = str(p).strip() == str(r).strip()
        correct += int(ok)
    return correct / max(1, len(refs))


def is_correct(pred, ref, metric: str, tol: float) -> bool:
    if metric == "numeric":
        pf, rf = try_float(pred), try_float(ref)
        if pf is not None and rf is not None:
            return abs(pf - rf) <= tol
    return str(pred).strip() == str(ref).strip()


def domain_from_dataset(dataset: str) -> str:
    return (dataset or "unknown").split(":", 1)[-1]


def split_from_domain(domain: str, in_domains: Iterable[str]) -> str:
    dom = (domain or "").strip().lower()
    in_set = {d.strip().lower() for d in in_domains} if in_domains else set()
    return "in-domain" if dom in in_set else "cross-domain"


def print_result_line(r: dict[str, Any], prefix: str = ""):
    dataset_label = r.get("dataset_label") or r.get("dataset", "")
    print(
        f"{prefix}{r.get('path', ''):40} | setup={r.get('setup', ''):15} dataset={dataset_label:28} "
        f"| metric={r.get('metric', ''):7} | N={int(r.get('N', 0)):4d} | acc={float(r.get('acc', 0.0)):.4f} "
        f"| avg_conv_len={float(r.get('avg_conv_len', 0.0)):.2f}"
    )


_TRAIN_EVAL_RE = re.compile(r"^(?P<setup>.+?)__train=(?P<train>.+?)__eval=(?P<eval>.+?)\.jsonl$")


def parse_train_eval_from_path(path: Path) -> tuple[Optional[str], Optional[str]]:
    m = _TRAIN_EVAL_RE.match(path.name)
    if not m:
        return None, None
    return m.group("train"), m.group("eval")


def infer_eval_from_path(path: Path) -> Optional[str]:
    _, eval_tag = parse_train_eval_from_path(path)
    if eval_tag:
        return eval_tag
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return None


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, rows: list[dict]):
    """
    Overwrite a JSONL file with the provided rows.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
    tmp.replace(path)
