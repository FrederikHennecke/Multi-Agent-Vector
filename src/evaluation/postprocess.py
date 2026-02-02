# evaluation/postprocess.py
import re
from typing import Callable

MC_LETTERS = "ABCDEFGHIJ"
CHOICE_LETTERS = list(MC_LETTERS)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _extract_choice_letter(text: str) -> str:
    """
    Robustly extract a multiple-choice letter A-J from arbitrary text.
    Priority:
      1) explicit 'ANSWER: <letter>'
      2) first standalone capital letter A-J
      3) last capital letter A-J
    Returns one of 'A'..'J' or '' if none.
    """
    t = text.strip()
    # 1) ANSWER: X
    m = re.search(r"(?i)answer\s*:\s*([A-J])\b", t)
    if m:
        return m.group(1).upper()
    # 2) Standalone letter tokens
    toks = re.findall(r"\b([A-J])\b", t)
    if toks:
        return toks[0].upper()
    # 3) Any capital letter mention
    toks = re.findall(r"([A-J])", t)
    if toks:
        return toks[-1].upper()
    return ""

def _identity_answer(_q: str, text: str) -> str:
    """Return normalized free-form text when no special formatting is needed."""
    return _normalize(text)

def pp_mc_letter(question: str, pred_text: str) -> str:
    # For MMLU-Pro multiple-choice datasets
    letter = _extract_choice_letter(pred_text)
    return letter or ""  # empty if we couldn't find a valid letter

def pp_freeform(question: str, pred_text: str) -> str:
    return _identity_answer(question, pred_text)

def get_postprocessor(dataset_tag: str) -> Callable[[str, str], str]:
    if (dataset_tag or "").startswith("mmlu_pro"):
        return pp_mc_letter
    return pp_freeform
