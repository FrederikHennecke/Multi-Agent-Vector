# evaluation/metrics.py
from typing import List, Tuple

def _try_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def accuracy_string(preds: List[str], refs: List[str]) -> float:
    correct = sum(1 for p, r in zip(preds, refs) if str(p).strip() == str(r).strip())
    return correct / max(1, len(refs))

def accuracy_numeric(preds: List[str], refs: List[str], tol: float = 0.0) -> float:
    """
    Numeric accuracy with optional absolute tolerance. If either side can't be parsed,
    fallback to string-equality for that example.
    """
    correct = 0
    for p, r in zip(preds, refs):
        pf, rf = _try_float(str(p).replace(",", "")), _try_float(str(r).replace(",", ""))
        if pf is not None and rf is not None:
            ok = abs(pf - rf) <= tol
        else:
            ok = str(p).strip() == str(r).strip()
        correct += int(ok)
    return correct / max(1, len(refs))

def compute_accuracy_for_tag(dataset_tag: str, preds: List[str], refs: List[str]) -> float:
    # Pick metric based on dataset family
    if dataset_tag.startswith("mmlu_pro"):
        return accuracy_string(preds, refs)  # letters A-J
    # default
    return accuracy_string(preds, refs)
