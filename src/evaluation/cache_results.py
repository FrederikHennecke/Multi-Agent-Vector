#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as project_config
from evaluation.results_common import (
    LETTERS,
    ROLE_INTEGRITY_SOURCE_HEURISTIC,
    ROLE_INTEGRITY_SOURCE_LLAMA,
    ROLE_INTEGRITY_KEYS,
    accuracy,
    canonical_from_cache,
    compute_verifier_stats,
    extract_choice_letter,
    extract_last_number,
    extract_verifier_verdict,
    final_text_from_conversation,
    first_generator_text,
    infer_eval_from_path,
    infer_metric,
    is_correct,
    load_jsonl,
    parse_train_eval_from_path,
    print_result_line,
    split_question_stem_and_options,
    valid_role_integrity_scores,
    write_jsonl,
)


class MCAnswerExtractor:
    """
    Helper that optionally asks a LLaMA model to read a response and emit the final MC letter
    (and, if requested, a numeric answer). Falls back to regex/heuristics whenever inference fails.
    """

    def __init__(
        self,
        mode: str = "regex",
        llama_model: str | None = None,
        llama_max_new_tokens: int = 4,
        llama_device: str | None = None,
    ):
        self.mode = mode
        self.llama_model = llama_model
        self.llama_max_new_tokens = llama_max_new_tokens
        self.llama_device = None if llama_device in (None, "auto") else llama_device
        self._host = None

    def extract(self, text: str) -> str:
        text = text or ""
        if not text.strip():
            return ""
        if self.mode != "llama":
            return extract_choice_letter(text)
        try:
            return self._extract_with_llama(text)
        except Exception as exc:
            print(f"[warn] LLaMA extractor failed ({exc}); falling back to regex.")
            return extract_choice_letter(text)

    def extract_numeric(self, text: str) -> str:
        """
        Extract a numeric answer; when using LLaMA mode, ask the model to emit only the number.
        """
        text = text or ""
        if not text.strip():
            return ""
        if self.mode != "llama":
            return extract_last_number(text)
        try:
            return self._extract_numeric_with_llama(text)
        except Exception as exc:
            print(f"[warn] LLaMA numeric extractor failed ({exc}); falling back to regex.")
            return extract_last_number(text)

    def close(self):
        if self._host is not None:
            try:
                self._host.unload()
            finally:
                self._host = None

    def get_host(self):
        """
        Return an initialized ModelHost instance when using LLaMA mode.

        This enables optional sharing of the same loaded model across multiple helper
        components (for example, answer extraction and question tagging).
        """
        self._ensure_host()
        return self._host

    def _ensure_host(self):
        if self._host is None:
            if not self.llama_model:
                raise ValueError("llama_model must be provided for LLaMA extraction.")
            from ModelHost import ModelHost

            host_kwargs = {}
            if self.llama_device:
                host_kwargs["device"] = self.llama_device
            self._host = ModelHost(self.llama_model, **host_kwargs)

    def _extract_with_llama(self, text: str) -> str:
        self._ensure_host()
        prompt = self._host.build_chat_prompt(
            "You read a model response and emit only the final multiple-choice letter (A-J). "
            "Reply with UNKNOWN if no letter is stated.",
            f"Model response:\n{text}\n\nAnswer:",
        )
        gen = self._host.generate(prompt, max_new_tokens=self.llama_max_new_tokens)
        letter = extract_choice_letter(gen)
        if letter:
            return letter
        gen = (gen or "").strip().upper()
        if gen in LETTERS:
            return gen
        return extract_choice_letter(text)

    def _extract_numeric_with_llama(self, text: str) -> str:
        self._ensure_host()
        prompt = self._host.build_chat_prompt(
            "You read a model response and emit only the final numeric answer. "
            "Return a signed/decimal number with no extra words. Reply with UNKNOWN if no number is present.",
            f"Model response:\n{text}\n\nAnswer:",
        )
        gen = self._host.generate(prompt, max_new_tokens=self.llama_max_new_tokens)
        num = extract_last_number(gen)
        return num if num else extract_last_number(text)


# ---------- question-type tagging ----------

QUESTION_TYPE_TAG_DEFS = [
    ("Definition", "Definition or name of a concept or term."),
    ("Basic Facts & Properties", "Basic facts or properties of an entity or material."),
    ("Structure", "Parts, composition, or organization of a system or object."),
    ("Processes & Causal", "Processes, steps, causes, or effects."),
    ("Teleology / Purpose", "Function, role, or purpose."),
    ("Algebraic", "Symbolic or quantitative reasoning (including genetics-style crosses)."),
    ("Experiments", "Experimental design, hypotheses, variables, or interpreting experiments."),
    ("Spatial / Kinematic", "Spatial relations, motion, direction, or kinematics."),
    ("other", "None of the above."),
]
QUESTION_TYPE_TAGS = [tag for tag, _ in QUESTION_TYPE_TAG_DEFS]
_QUESTION_TYPE_ORDER = {t: i for i, t in enumerate(QUESTION_TYPE_TAGS)}
_QUESTION_TYPE_ALIASES = {
    "definition": "Definition",
    "basic facts properties": "Basic Facts & Properties",
    "basic facts and properties": "Basic Facts & Properties",
    "basic facts": "Basic Facts & Properties",
    "facts properties": "Basic Facts & Properties",
    "structure": "Structure",
    "processes causal": "Processes & Causal",
    "processes and causal": "Processes & Causal",
    "process causal": "Processes & Causal",
    "causal processes": "Processes & Causal",
    "teleology purpose": "Teleology / Purpose",
    "teleology and purpose": "Teleology / Purpose",
    "teleology": "Teleology / Purpose",
    "purpose": "Teleology / Purpose",
    "algebraic": "Algebraic",
    "experiment": "Experiments",
    "experiments": "Experiments",
    "experimental": "Experiments",
    "spatial kinematic": "Spatial / Kinematic",
    "spatial and kinematic": "Spatial / Kinematic",
    "kinematic": "Spatial / Kinematic",
    "spatial": "Spatial / Kinematic",
    "other": "other",
}


def _normalize_question_tag(tag: str) -> str | None:
    if not tag:
        return None
    raw = str(tag).strip()
    if not raw:
        return None
    if raw in QUESTION_TYPE_TAGS:
        return raw
    lowered = raw.lower()
    if lowered == "other":
        return "other"
    key = lowered.replace("&", "and").replace("/", " ")
    key = key.replace("_", " ")
    key = re.sub(r"[^a-z0-9 ]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return _QUESTION_TYPE_ALIASES.get(key)


def _ordered_tags(tags: Iterable[str]) -> list[str]:
    clean = []
    seen = set()
    for t in tags:
        norm = _normalize_question_tag(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        clean.append(norm)
    return sorted(clean, key=lambda x: _QUESTION_TYPE_ORDER.get(x, 10_000))


def heuristic_question_tags(stem: str) -> list[str]:
    """
    Heuristic, multi-label tagger for question type.

    Tags follow ARC knowledge types (Clark et al., 2018, Table 4):
    - Definition: definition or name of a concept/term
    - Basic Facts & Properties: recall of basic facts or properties
    - Structure: parts, composition, or organization
    - Processes & Causal: processes, steps, causes, or effects
    - Teleology / Purpose: function, role, or purpose
    - Algebraic: symbolic or quantitative reasoning
    - Experiments: experiments, hypotheses, variables, or results
    - Spatial / Kinematic: spatial relations or motion
    - other: none of the above
    """
    text = " ".join(str(stem or "").split())
    t = text.lower()
    if not t:
        return ["other"]

    tags: set[str] = set()

    has_digits = bool(re.search(r"\d", t))
    has_math_ops = bool(re.search(r"[=+\-*/^]", t))
    has_math_terms = bool(
        re.search(
            r"\b(calculate|compute|solve|simplify|approximate|probability|percent|percentage|ratio|proportion|"
            r"mean|median|variance|equation|formula|log|ln|sin|cos|tan|algebra|exponent|root)\b",
            t,
        )
    )
    has_genetics_terms = bool(
        re.search(
            r"\b(genotype|phenotype|allele|dominant|recessive|homozygous|heterozygous|offspring|cross|punnett)\b",
            t,
        )
    )
    if has_math_terms or has_genetics_terms or (has_digits and has_math_ops):
        tags.add("Algebraic")

    has_experiment_terms = bool(
        re.search(
            r"\b(experiment|experiments|experimental|investigation|hypothesis|control group|controlled experiment|"
            r"independent variable|dependent variable|trial|scientist|laboratory)\b",
            t,
        )
    )
    has_data_terms = bool(
        re.search(r"\b(data|measurement|measurements|measured|observe|observed|observation|results|recorded)\b", t)
    )
    has_table_terms = bool(re.search(r"\b(table|figure|graph|plot|chart|diagram)\b", t))
    has_context_terms = bool(re.search(r"\b(shown|below|following|given|based on)\b", t))
    if has_experiment_terms or (has_table_terms and has_context_terms and has_data_terms):
        tags.add("Experiments")
    elif has_table_terms and has_context_terms:
        tags.add("Experiments")

    if re.search(
        r"\b(function|purpose|role|used for|serves to|serve to|in order to|main function|primary function|"
        r"job of|helps? to)\b",
        t,
    ):
        tags.add("Teleology / Purpose")

    if re.search(
        r"\b(process|cycle|step|steps|first step|next step|sequence|formation|causes?|leads to|results in|"
        r"effect of|because|why|happens when|as a result|change|changes|increases?|decreases?)\b",
        t,
    ):
        tags.add("Processes & Causal")

    if re.search(
        r"\b(structure|layer|layers|part|parts|component|components|section|organ|organs|system|systems|"
        r"composed of|consists of|made of|arranged|organization|structure of|anatomy|crust|mantle|core|"
        r"cell|tissue)\b",
        t,
    ):
        tags.add("Structure")

    if re.search(
        r"\b(move|moves|moving|motion|speed|velocity|acceleration|direction|distance|position|location|orbit|"
        r"rotate|revolve|tilt|axis|path|trajectory|north|south|east|west|above|below|top|bottom|folded|fault|"
        r"angle)\b",
        t,
    ):
        tags.add("Spatial / Kinematic")

    if re.search(
        r"\b(define|definition|means|refers to|is called|are called|known as|term for|term used)\b",
        t,
    ):
        tags.add("Definition")

    has_property_terms = bool(
        re.search(
            r"\b(property|properties|characteristic|trait|feature|mass|weight|density|volume|temperature|"
            r"pressure|hardness|luster|color|state|phase|melting point|boiling point|atomic number|charge|"
            r"energy|element|gas|planet|mineral|substance|material|compound|organism|species)\b",
            t,
        )
    )
    has_fact_wh = bool(re.search(r"^(what|which|who|when|where)\b", t))
    has_causal_terms = bool(re.search(r"\b(why|how|cause|effect|process|function|purpose)\b", t))
    if has_property_terms or (not tags and has_fact_wh and not has_causal_terms):
        tags.add("Basic Facts & Properties")

    if not tags:
        tags.add("other")

    return _ordered_tags(tags)


def _parse_tag_list(raw: str) -> list[str]:
    """
    Parse a tag list from model output. Supports JSON arrays and loose comma-separated lists.
    """
    s = (raw or "").strip()
    if not s:
        return []

    # Try to extract a JSON array.
    m = re.search(r"\[[^\]]*\]", s, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

    # Fallback: split by commas/newlines.
    parts = re.split(r"[,\\n]+", s)
    return [p.strip() for p in parts if p.strip()]


class QuestionTypeTagger:
    """
    Assign coarse question-type tags, optionally using a local LLaMA model via ModelHost.

    The output is a list of tags drawn from QUESTION_TYPE_TAGS.
    """

    def __init__(
        self,
        mode: str = "heuristic",
        llama_model: str | None = None,
        llama_max_new_tokens: int = 96,
        llama_device: str | None = None,
        host=None,
    ):
        self.mode = mode
        self.llama_model = llama_model
        self.llama_max_new_tokens = llama_max_new_tokens
        self.llama_device = None if llama_device in (None, "auto") else llama_device
        self._host = host
        self._owns_host = host is None
        self._cache: dict[str, list[str]] = {}

    def close(self):
        if self._host is not None and self._owns_host:
            try:
                self._host.unload()
            finally:
                self._host = None

    def tag(self, question_stem: str) -> list[str]:
        stem = " ".join(str(question_stem or "").split())
        if stem in self._cache:
            return self._cache[stem]

        if self.mode == "none":
            tags: list[str] = []
            self._cache[stem] = tags
            return tags
        if self.mode != "llama":
            tags = heuristic_question_tags(stem)
            self._cache[stem] = tags
            return tags
        try:
            tags = self._tag_with_llama(stem)
            self._cache[stem] = tags
            return tags
        except Exception as exc:
            print(f"[warn] LLaMA tagger failed ({exc}); falling back to heuristic.")
            tags = heuristic_question_tags(stem)
            self._cache[stem] = tags
            return tags

    def _ensure_host(self):
        if self._host is None:
            if not self.llama_model:
                raise ValueError("llama_model must be provided for LLaMA tagging.")
            from ModelHost import ModelHost

            host_kwargs = {}
            if self.llama_device:
                host_kwargs["device"] = self.llama_device
            self._host = ModelHost(self.llama_model, **host_kwargs)

    def _tag_with_llama(self, question_stem: str) -> list[str]:
        self._ensure_host()
        allowed = ", ".join(QUESTION_TYPE_TAGS)
        tag_lines = "\n".join(f"- {tag}: {desc}" for tag, desc in QUESTION_TYPE_TAG_DEFS)
        system = (
            "You label multiple-choice questions with ARC knowledge-type tags (Table 4 in Clark et al., 2018). "
            "Choose 1-3 tags from the allowed list and output ONLY a JSON array of strings.\n\n"
            "Tag meanings:\n"
            f"{tag_lines}\n\n"
            f"Allowed tags: {allowed}"
        )
        user = f"Question stem:\n{question_stem}\n\nTags:"
        prompt = self._host.build_chat_prompt(system, user)
        gen = self._host.generate(prompt, max_new_tokens=self.llama_max_new_tokens)
        raw_tags = _parse_tag_list(gen)
        norm = _ordered_tags(raw_tags)
        return norm if norm else heuristic_question_tags(question_stem)


# ---------- role integrity scoring ----------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_FINAL_ANSWER_RE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*:\s*([A-J])\b")
_FINAL_ANSWER_END_RE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*:\s*([A-J])\b\s*[.!]?\s*$")
_OTHER_ANSWER_RE = re.compile(r"(?i)\b(?:answer|choice|option)\s*[:\-]?\s*([A-J])\b")
_STEP_RE = re.compile(r"(?m)^\s*\d+\.")
_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "for",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "is",
    "it",
    "its",
    "an",
    "a",
    "or",
    "if",
    "then",
    "than",
    "thus",
    "so",
    "because",
    "since",
    "therefore",
    "which",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "we",
    "you",
    "i",
    "they",
    "he",
    "she",
    "them",
    "us",
    "our",
    "their",
    "answer",
    "choice",
    "option",
    "correct",
    "incorrect",
    "verdict",
    "critique",
    "solution",
    "question",
    "problem",
    "given",
    "choose",
    "select",
}


def _conversation_messages(rec) -> list[tuple[str, str]]:
    conv = rec.get("conversation")
    if not isinstance(conv, list):
        return []
    out = []
    for item in conv:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            role, msg = item
            if isinstance(msg, str) and msg.strip():
                out.append((str(role), msg))
    return out


def _last_role_message(conv: list[tuple[str, str]], role: str) -> str:
    for r, msg in reversed(conv):
        if r == role and isinstance(msg, str) and msg.strip():
            return msg
    return ""


def _last_verifier_text(rec) -> str:
    fb = rec.get("verifier_feedback")
    if isinstance(fb, list) and fb:
        last = fb[-1]
        if isinstance(last, dict):
            raw = last.get("raw")
            if isinstance(raw, str) and raw.strip():
                return raw
            critique = last.get("critique")
            verdict = last.get("verdict") or last.get("verdicts") or last.get("verdict_raw")
            parts = []
            if verdict:
                parts.append(f"VERDICT: {verdict}")
            if critique:
                parts.append(f"CRITIQUE: {critique}")
            if parts:
                return "\n".join(parts)
    conv = _conversation_messages(rec)
    return _last_role_message(conv, "VERIFIER")


def _last_refiner_and_prev_verifier(conv: list[tuple[str, str]]) -> tuple[str, str]:
    refiner = ""
    prev_verifier = ""
    for idx in range(len(conv) - 1, -1, -1):
        role, msg = conv[idx]
        if role == "REFINER" and isinstance(msg, str) and msg.strip():
            refiner = msg
            for j in range(idx - 1, -1, -1):
                role2, msg2 = conv[j]
                if role2 == "VERIFIER" and isinstance(msg2, str) and msg2.strip():
                    prev_verifier = msg2
                    break
            break
    return refiner, prev_verifier


def _extract_critique_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    m = re.search(r"(?i)CRITIQUE:\s*(.*)", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip()


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _significant_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOPWORDS}


def _normalize_role_key(key: str) -> str:
    k = str(key).strip().lower().replace(" ", "_").replace("-", "_")
    if k.endswith("_score"):
        k = k[:-6]
    return k


def _parse_role_integrity_scores(raw: str) -> Optional[dict[str, int]]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, int] = {}
    for k, v in data.items():
        key = _normalize_role_key(k)
        if key not in ROLE_INTEGRITY_KEYS:
            continue
        try:
            val = int(float(v))
        except Exception:
            continue
        if 0 <= val <= 2:
            out[key] = val
    return out if valid_role_integrity_scores(out) else None


def _clip_text(text: str, max_chars: int = 2400) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.6)].rstrip()
    tail = s[-int(max_chars * 0.4) :].lstrip()
    return f"{head}\n...[truncated]...\n{tail}"


def score_generator_reasoning(text: str) -> int:
    if not text or not text.strip():
        return 0
    words = _tokenize(text)
    word_count = len(words)
    if word_count < 15:
        return 0
    has_step = bool(re.search(r"(?i)step\s*by\s*step|step-by-step|\bsteps?\b", text)) or bool(
        _STEP_RE.search(text)
    )
    has_reason = bool(re.search(r"(?i)\b(because|therefore|thus|hence|since|so|as a result)\b", text))
    sentence_count = len([s for s in re.split(r"[.!?]", text) if s.strip()])
    if has_step and word_count >= 60 and sentence_count >= 3:
        return 2
    if (has_step or has_reason or sentence_count >= 3) and word_count >= 25:
        return 1
    return 0


def score_generator_final_format(text: str) -> int:
    if not text or not text.strip():
        return 0
    if _FINAL_ANSWER_END_RE.search(text.strip()):
        return 2
    if _FINAL_ANSWER_RE.search(text) or _OTHER_ANSWER_RE.search(text):
        return 1
    return 0


def score_verifier_verdict(text: str) -> int:
    if not text or not text.strip():
        return 0
    if re.search(r"(?i)\bVERDICT\s*:\s*(CORRECT|INCORRECT)\b", text):
        return 2
    if re.search(r"(?i)\b(correct|incorrect|right|wrong)\b", text) or re.search(
        r"(?i)\bverdict\b", text
    ):
        return 1
    return 0


def score_verifier_grounding(critique_text: str, question_text: str) -> int:
    if not critique_text or not critique_text.strip():
        return 0
    crit_tokens = _significant_tokens(critique_text)
    if len(crit_tokens) < 3:
        return 0
    q_tokens = _significant_tokens(question_text)
    overlap = crit_tokens & q_tokens
    if len(overlap) >= 3:
        return 2
    if len(overlap) >= 1:
        return 1
    if re.search(r"(?i)\b(option|choice|answer)\s*[A-J]\b", critique_text):
        return 1
    return 0


def score_refiner_revision(refiner_text: str, critique_text: str, generator_text: str) -> int:
    if not refiner_text or not refiner_text.strip():
        return 0
    gen_letter = extract_choice_letter(generator_text)
    ref_letter = extract_choice_letter(refiner_text)
    changed_answer = bool(gen_letter and ref_letter and ref_letter != gen_letter)
    acknowledges = bool(
        re.search(r"(?i)\b(critique|verifier|feedback|correct|incorrect|fix|revise|update|error)\b", refiner_text)
    )
    if not critique_text or not critique_text.strip():
        return 1 if (changed_answer or acknowledges) else 0
    crit_tokens = _significant_tokens(critique_text)
    ref_tokens = _significant_tokens(refiner_text)
    overlap = len(crit_tokens & ref_tokens)
    if overlap >= 2 and (changed_answer or acknowledges):
        return 2
    if overlap >= 1 or changed_answer or acknowledges:
        return 1
    return 0


def heuristic_role_integrity_scores(rec: dict, question_stem: str, option_lines: list[str]) -> dict[str, int]:
    conv = _conversation_messages(rec)
    generator_text = first_generator_text(rec)
    verifier_text = _last_verifier_text(rec)
    critique_text = _extract_critique_text(verifier_text)
    refiner_text, ref_verifier_text = _last_refiner_and_prev_verifier(conv)
    ref_critique = _extract_critique_text(ref_verifier_text) if ref_verifier_text else critique_text
    question_text = " ".join([question_stem] + option_lines).strip()
    return {
        "g_reasoning": score_generator_reasoning(generator_text),
        "g_final_format": score_generator_final_format(generator_text),
        "v_verdict": score_verifier_verdict(verifier_text),
        "v_grounding": score_verifier_grounding(critique_text, question_text),
        "r_revision": score_refiner_revision(refiner_text, ref_critique, generator_text),
    }


class RoleIntegrityScorer:
    """
    Score role integrity using a local LLaMA model, with heuristic fallback.
    """

    def __init__(
        self,
        mode: str = "llama",
        llama_model: str | None = None,
        llama_max_new_tokens: int = 96,
        llama_device: str | None = None,
        host=None,
    ):
        self.mode = mode
        self.llama_model = llama_model
        self.llama_max_new_tokens = llama_max_new_tokens
        self.llama_device = None if llama_device in (None, "auto") else llama_device
        self._host = host
        self._owns_host = host is None

    @property
    def source(self) -> str:
        return ROLE_INTEGRITY_SOURCE_LLAMA if self.mode == "llama" else ROLE_INTEGRITY_SOURCE_HEURISTIC

    def close(self):
        if self._host is not None and self._owns_host:
            try:
                self._host.unload()
            finally:
                self._host = None

    def score(self, rec: dict, question_stem: str, option_lines: list[str]) -> tuple[dict[str, int], str]:
        if self.mode != "llama":
            return heuristic_role_integrity_scores(rec, question_stem, option_lines), ROLE_INTEGRITY_SOURCE_HEURISTIC
        try:
            scores = self._score_with_llama(rec, question_stem, option_lines)
            return scores, ROLE_INTEGRITY_SOURCE_LLAMA
        except Exception as exc:
            print(f"[warn] Role-integrity LLaMA scorer failed ({exc}); falling back to heuristic.")
            return heuristic_role_integrity_scores(rec, question_stem, option_lines), ROLE_INTEGRITY_SOURCE_HEURISTIC

    def _ensure_host(self):
        if self._host is None:
            if not self.llama_model:
                raise ValueError("llama_model must be provided for role-integrity scoring.")
            from ModelHost import ModelHost

            host_kwargs = {}
            if self.llama_device:
                host_kwargs["device"] = self.llama_device
            self._host = ModelHost(self.llama_model, **host_kwargs)

    def _score_with_llama(self, rec: dict, question_stem: str, option_lines: list[str]) -> dict[str, int]:
        self._ensure_host()
        conv = _conversation_messages(rec)
        generator_text = first_generator_text(rec)
        verifier_text = _last_verifier_text(rec)
        refiner_text, ref_verifier_text = _last_refiner_and_prev_verifier(conv)
        critique_text = _extract_critique_text(verifier_text)
        ref_critique = _extract_critique_text(ref_verifier_text) if ref_verifier_text else critique_text
        options = "\n".join(option_lines) if option_lines else "(none)"
        user = project_config.ROLE_INTEGRITY_USER_TEMPLATE.format(
            question_stem=_clip_text(question_stem),
            options=_clip_text(options),
            generator=_clip_text(generator_text) or "(none)",
            verifier=_clip_text(verifier_text) or "(none)",
            refiner=_clip_text(refiner_text) or "(none)",
            refiner_critique=_clip_text(ref_critique) or "(none)",
        )
        prompt = self._host.build_chat_prompt(project_config.ROLE_INTEGRITY_SYSTEM_PROMPT, user)
        gen = self._host.generate(prompt, max_new_tokens=self.llama_max_new_tokens)
        parsed = _parse_role_integrity_scores(gen)
        if parsed is None:
            raise ValueError("role-integrity scorer returned invalid JSON")
        return parsed


def canonicalize(text: str, metric: str, choice_extractor: MCAnswerExtractor | None = None) -> str:
    if metric == "numeric":
        if choice_extractor is not None:
            return choice_extractor.extract_numeric(text)
        return extract_last_number(text)
    if metric == "mc":
        if choice_extractor is not None:
            return choice_extractor.extract(text)
        return extract_choice_letter(text)
    return (text or "").strip()


def summarize_file(
    path: Path,
    tol: float,
    choice_extractor: MCAnswerExtractor | None = None,
    force_recompute: bool = False,
    persist_final_answers: bool = True,
    question_tagger: QuestionTypeTagger | None = None,
    force_retag: bool = False,
    persist_question_tags: bool = True,
    role_rows: Optional[list[dict]] = None,
    role_scorer: RoleIntegrityScorer | None = None,
    force_role_integrity: bool = False,
    persist_role_scores: bool = True,
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

    updated_final_cache = False
    updated_question_tags = False
    updated_role_cache = False
    summary_row = None
    for (cond, ds, train_ds), recs in groups.items():
        refs = []
        final_texts = []
        gen_first_preds, verdicts = [], []
        answer_lengths = []
        answer_token_counts = []
        conv_lengths = []

        for r in recs:
            q_raw = r.get("question") or r.get("prompt") or r.get("input") or r.get("query") or ""
            stem, option_lines = split_question_stem_and_options(str(q_raw))
            if question_tagger is not None:
                existing = r.get("question_tags")
                should_tag = force_retag or not (
                    isinstance(existing, list) and any(str(x).strip() for x in existing)
                )
                if should_tag:
                    tags = question_tagger.tag(stem)
                    if persist_question_tags and tags and existing != tags:
                        r["question_tags"] = tags
                        r["question_tags_source"] = question_tagger.mode
                        updated_question_tags = True

            existing_scores = r.get("role_integrity_scores")
            existing_source = r.get("role_integrity_source")
            desired_source = role_scorer.source if role_scorer else ROLE_INTEGRITY_SOURCE_HEURISTIC
            has_existing = valid_role_integrity_scores(existing_scores) and existing_source == desired_source
            if force_role_integrity or not has_existing:
                if role_scorer is not None:
                    scores, source = role_scorer.score(r, stem, option_lines)
                else:
                    scores = heuristic_role_integrity_scores(r, stem, option_lines)
                    source = ROLE_INTEGRITY_SOURCE_HEURISTIC
                if persist_role_scores:
                    if existing_scores != scores or existing_source != source:
                        r["role_integrity_scores"] = scores
                        r["role_integrity_source"] = source
                        updated_role_cache = True
            else:
                scores = existing_scores
            if role_rows is not None and isinstance(scores, dict):
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
            gen_first_preds.append(first_generator_text(r))
            verdicts.append(extract_verifier_verdict(r))

            conv = r.get("conversation")
            if isinstance(conv, list):
                conv_lengths.append(len(conv))
            else:
                conv_lengths.append(0)

        metric = infer_metric(ds, refs[: min(20, len(refs))])
        preds_canon: List[str] = []
        gen_first_canon: List[str] = []
        for r, final_txt, gen_txt in zip(recs, final_texts, gen_first_preds):
            cached_final_raw = canonical_from_cache(r.get("final_answer"), metric)
            cached_final = "" if force_recompute else cached_final_raw
            final_ans = cached_final
            if not final_ans:
                final_ans = canonicalize(final_txt, metric, choice_extractor=choice_extractor)
            if not final_ans and cached_final_raw:
                # If force-recompute produced nothing, keep the previous cache.
                final_ans = cached_final_raw
            if persist_final_answers and final_ans and r.get("final_answer") != final_ans:
                r["final_answer"] = final_ans
                updated_final_cache = True
            preds_canon.append(final_ans)

            cached_gen_raw = canonical_from_cache(r.get("generator_answer"), metric)
            cached_gen = "" if force_recompute else cached_gen_raw
            gen_ans = cached_gen
            if not gen_ans:
                gen_ans = canonicalize(gen_txt, metric, choice_extractor=choice_extractor)
            if not gen_ans and cached_gen_raw:
                gen_ans = cached_gen_raw
            if persist_final_answers and gen_ans and r.get("generator_answer") != gen_ans:
                r["generator_answer"] = gen_ans
                updated_final_cache = True
            gen_first_canon.append(gen_ans)

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

        # We only expect a single group per file, so break after the first.
        break

    if (
        (updated_final_cache and persist_final_answers)
        or (updated_question_tags and persist_question_tags)
        or (updated_role_cache and persist_role_scores)
    ):
        write_jsonl(path, records)

    return summary_row


def main():
    ap = argparse.ArgumentParser(description="Cache extracted answers/tags into results JSONL files.")
    ap.add_argument(
        "files",
        nargs="*",
        help="One or more .jsonl result files (glob is fine).",
        default="results_8B/*.jsonl",
    )
    ap.add_argument("--tol", type=float, default=0.0, help="Numeric tolerance (abs) for numeric tasks.")
    ap.add_argument(
        "--mc-extractor",
        choices=("regex", "llama"),
        default="llama",
        help="How to canonicalize multiple-choice answers (regex or LLaMA).",
    )
    ap.add_argument(
        "--mc-llama-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model to load when --mc-extractor=llama.",
    )
    ap.add_argument(
        "--mc-llama-max-new-tokens",
        type=int,
        default=8,
        help="Max tokens to decode when using LLaMA for MC extraction.",
    )
    ap.add_argument(
        "--mc-llama-device",
        default="auto",
        help="Device for running the LLaMA extractor (e.g., cuda:0, cpu, auto).",
    )
    ap.add_argument(
        "--role-integrity-scorer",
        choices=("llama", "heuristic"),
        default="llama",
        help="How to score role integrity (llama or heuristic).",
    )
    ap.add_argument(
        "--role-llama-model",
        default=None,
        help="Base model to load for role-integrity scoring (defaults to --mc-llama-model).",
    )
    ap.add_argument(
        "--role-llama-max-new-tokens",
        type=int,
        default=96,
        help="Max tokens to decode when using LLaMA for role-integrity scoring.",
    )
    ap.add_argument(
        "--role-llama-device",
        default="auto",
        help="Device for running the role-integrity model (e.g., cuda:0, cpu, auto).",
    )
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Re-extract answers even if final_answer is already cached in the JSONL files.",
    )
    ap.add_argument(
        "--no-write-final-cache",
        action="store_true",
        help="Do not persist extracted final answers back into the result files.",
    )
    ap.add_argument(
        "--question-tagger",
        choices=("none", "heuristic", "llama"),
        default="llama",
        help="Assign coarse question-type tags per question and cache them into the result JSONL files.",
    )
    ap.add_argument(
        "--tag-llama-model",
        default=None,
        help="Base model to load when --question-tagger=llama (defaults to --mc-llama-model).",
    )
    ap.add_argument(
        "--tag-llama-max-new-tokens",
        type=int,
        default=96,
        help="Max tokens to decode when using LLaMA for question tagging.",
    )
    ap.add_argument(
        "--tag-llama-device",
        default="auto",
        help="Device for running the tagger model (e.g., cuda:0, cpu, auto).",
    )
    ap.add_argument(
        "--force-retag",
        action="store_true",
        help="Recompute question tags even if question_tags are already cached in the JSONL files.",
    )
    ap.add_argument(
        "--no-write-tag-cache",
        action="store_true",
        help="Do not persist question tags back into the result files.",
    )
    ap.add_argument(
        "--force-role-integrity",
        action="store_true",
        help="Recompute role-integrity scores even if already cached in the JSONL files.",
    )
    ap.add_argument(
        "--no-write-role-cache",
        action="store_true",
        help="Do not persist role-integrity scores back into the result files.",
    )
    args = ap.parse_args()

    results = []
    role_rows: list[dict] = []

    files = args.files if isinstance(args.files, list) else [args.files]
    choice_extractor = None
    if args.mc_extractor == "llama":
        choice_extractor = MCAnswerExtractor(
            mode="llama",
            llama_model=args.mc_llama_model,
            llama_max_new_tokens=args.mc_llama_max_new_tokens,
            llama_device=args.mc_llama_device,
        )

    question_tagger = None
    if args.question_tagger != "none":
        tag_model = args.tag_llama_model or args.mc_llama_model
        shared_host = None
        if args.question_tagger == "llama" and args.mc_extractor == "llama" and choice_extractor is not None:
            if tag_model == args.mc_llama_model and args.tag_llama_device == args.mc_llama_device:
                try:
                    shared_host = choice_extractor.get_host()
                except Exception as exc:
                    print(f"[warn] Failed to reuse MC extractor host for tagging ({exc}); loading a separate model.")
        question_tagger = QuestionTypeTagger(
            mode=args.question_tagger,
            llama_model=tag_model,
            llama_max_new_tokens=args.tag_llama_max_new_tokens,
            llama_device=args.tag_llama_device,
            host=shared_host,
        )

    if args.role_integrity_scorer == "llama":
        role_model = args.role_llama_model or args.mc_llama_model
        role_device = args.role_llama_device
        shared_host = None
        if args.mc_extractor == "llama" and choice_extractor is not None:
            if role_model == args.mc_llama_model and role_device == args.mc_llama_device:
                try:
                    shared_host = choice_extractor.get_host()
                except Exception as exc:
                    print(
                        f"[warn] Failed to reuse MC extractor host for role scoring ({exc}); loading a separate model."
                    )
        role_scorer = RoleIntegrityScorer(
            mode="llama",
            llama_model=role_model,
            llama_max_new_tokens=args.role_llama_max_new_tokens,
            llama_device=role_device,
            host=shared_host,
        )
    else:
        role_scorer = RoleIntegrityScorer(mode="heuristic")

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
                summarize_file(
                    p,
                    tol=args.tol,
                    choice_extractor=choice_extractor,
                    force_recompute=args.force_recompute,
                    persist_final_answers=not args.no_write_final_cache,
                    question_tagger=question_tagger,
                    force_retag=args.force_retag,
                    persist_question_tags=not args.no_write_tag_cache,
                    role_rows=role_rows,
                    role_scorer=role_scorer,
                    force_role_integrity=args.force_role_integrity,
                    persist_role_scores=not args.no_write_role_cache,
                )
            )
    results = [r for r in results if r]

    print("--------------")
    for r in sorted(results, key=lambda x: (x["dataset"], -x["acc"])):
        print_result_line(r)
    print(json.dumps(results))

    if role_rows:
        score_cols = list(ROLE_INTEGRITY_KEYS)
        # Avoid pandas dependency here by aggregating manually.
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

    if choice_extractor:
        choice_extractor.close()
    if question_tagger:
        question_tagger.close()
    if role_scorer:
        role_scorer.close()


if __name__ == "__main__":
    main()
