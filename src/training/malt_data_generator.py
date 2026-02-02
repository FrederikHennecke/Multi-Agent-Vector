# training/malt_data_generator.py
import json
import random
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from tqdm.auto import tqdm

from AgentFactory import AgentFactory
from config import ROLE_SYSTEM_PROMPTS


# ---------- robust numeric extractor: get the LAST number & normalize ----------
_SEP_CHARS = r"[,\.\s\u00A0\u2009']"   # comma, dot, space, NBSP, thin space, apostrophe
_CURRENCY  = r"$€£¥₹"
_UNITS     = r"%"
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

def _extract_last_number(text: str) -> str:
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
    has_dot   = "." in s
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
    if s.startswith("+"): s = s[1:]
    if s in {"", ".", "+", "-"}: return ""
    if s.endswith("."): s = s[:-1]
    if "." in s:
        head, tail = s.split(".", 1)
        head = re.sub(r"^(-?)0+(?=\d)", r"\1", head) or ("-" if head.startswith("-") else "0")
        s = (head + "." + tail).rstrip("0").rstrip(".")
    else:
        s = re.sub(r"^(-?)0+(?=\d)", r"\1", s)
    if neg and not s.startswith("-"):
        s = "-" + s
    return s

def _extract_choice_letter(text: str) -> str:
    t = text or ""
    m = re.search(r"(?i)\banswer\s*:\s*([A-F])\b", t)
    if m: return m.group(1).upper()
    toks = re.findall(r"\b([A-F])\b", t)
    if toks: return toks[-1].upper()
    toks = re.findall(r"([A-F])", t)
    return toks[-1].upper() if toks else ""

def _canonicalize_pred(pred_text: str, gold: Optional[str]) -> str:
    """If gold is an A–F letter, treat as MC. Else try numeric; else strip."""
    gold_s = (gold or "").strip().upper()
    if len(gold_s) == 1 and gold_s in "ABCDEF":
        return _extract_choice_letter(pred_text)
    # numeric fallback
    num = _extract_last_number(pred_text)
    if num != "":
        return num
    return (pred_text or "").strip()

def _format_verifier_label(verdict: str, solution_text: str, gold: Optional[str], given_critique: Optional[str] = None) -> str:
    verdict_norm = verdict.strip().upper()
    if verdict_norm != "CORRECT":
        verdict_norm = "INCORRECT"

    if given_critique and given_critique.strip():
        critique = given_critique.strip()
    elif verdict_norm == "CORRECT":
        critique = "The reasoning and final answer are consistent with the expected solution."
    else:
        pred = _canonicalize_pred(solution_text, gold)
        gold_clean = (gold or "").strip()
        if gold_clean and pred:
            critique = f"The final answer '{pred}' does not match the expected answer '{gold_clean}'."
        elif gold_clean:
            critique = f"The solution does not match the expected answer '{gold_clean}'."
        elif pred:
            critique = f"The final answer '{pred}' is not sufficiently justified by the reasoning."
        else:
            critique = "The reasoning does not reach a supported final answer."

    return f"VERDICT: {verdict_norm}\nCRITIQUE: {critique}"

def _parse_verifier_output(text: str) -> Tuple[str, str]:
    """
    Extract verdict and critique components from the verifier response.
    """
    raw = text or ""
    verdict = "INCORRECT"
    critique = ""

    verdict_match = re.search(r"VERDICT\s*:\s*(CORRECT|INCORRECT)", raw, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        fallback = re.search(r"\b(CORRECT|INCORRECT)\b", raw, re.IGNORECASE)
        if fallback:
            verdict = fallback.group(1).upper()

    critique_match = re.search(r"CRITIQUE\s*:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if critique_match:
        critique = critique_match.group(1).strip()
    else:
        lines = raw.strip().splitlines()
        if lines:
            first_line = lines[0]
            if re.search(r"\b(CORRECT|INCORRECT)\b", first_line, re.IGNORECASE):
                lines = lines[1:]
            critique = "\n".join(lines).strip()

    return verdict, critique


class MALTDataGenerator:
    """
    Generate multi-agent rollouts and emit role-specific JSONLs.
    Each line format:
      {
        "trajectory_id": <uuid>,
        "parent_id": <uuid or null>,
        "role": "generator"|"verifier"|"refiner",
        "system": str,
        "user": str,
        "label": str,
        "reward": float or null
      }
    """

    def __init__(self, base_model, data_dir: Path,
                 search_width=4, max_depth=2, seed=42,
                 do_sample=True, temperature=0.8, top_p=0.95, top_k=None,
                 dedup=True):
        self.base_model = base_model
        self.data_dir = Path(data_dir)
        self.search_width = int(search_width)
        self.max_depth = int(max_depth)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature) if temperature is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.dedup = bool(dedup)
        random.seed(seed)

    def _make_agents(self):
        # Use your prompt-based agents so system/user remain raw in saved data
        factory = AgentFactory(self.base_model, data_dir=self.data_dir)
        agents, host = factory.build_prompt(verifier_decision="gen")  # prompt role setup with critique generation
        return agents, host  # we keep host in case you want future extensions

    def _sample_generator_outputs(self, gen_agent, q: str, k: int) -> List[str]:
        """
        Get up to k *diverse* generator outputs. If sampling is disabled upstream,
        we still deduplicate to avoid writing identical branches.
        """
        outs: List[str] = []
        seen_norm: set = set()

        for i in range(k * 3):  # small oversample to survive dedup
            # Try to pass sampling hints if Agent supports **kwargs (see tiny patch below)
            try:
                text = gen_agent.generate(
                    q,
                    max_new=256,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            except TypeError:
                # Agent doesn't accept kwargs; fall back (likely greedy decode)
                text = gen_agent.generate(q)

            # Deduplicate by normalized final answer + tail of text
            norm_key = (_extract_choice_letter(text) or _extract_last_number(text) or text[-40:]).strip().upper()
            if self.dedup:
                if norm_key in seen_norm:
                    continue
                seen_norm.add(norm_key)

            outs.append(text)
            if len(outs) >= k:
                break

        # Ensure we return at most k
        return outs[:k]

    def _verify(self, ver_agent, q: str, sol: str) -> Tuple[str, str, str]:
        v_in = (
            f"QUESTION:\n{q}\nSOLUTION:\n{sol}\n\n"
            "Respond with:\nVERDICT: CORRECT or INCORRECT\nCRITIQUE: <brief justification>."
        )
        txt = ver_agent.generate(v_in)
        verdict, critique = _parse_verifier_output(txt)
        return verdict, critique, txt

    def _refine(self, ref_agent, q: str, sol: str, feedback: Optional[str]) -> str:
        feedback_text = feedback.strip() if feedback else "No critique provided."
        r_in = (
            f"QUESTION:\n{q}\n"
            f"INCORRECT SOLUTION:\n{sol}\n"
            f"VERIFIER FEEDBACK:\n{feedback_text}\n\n"
            "Provide a corrected, step-by-step solution. End with: ANSWER: <answer>."
        )
        return ref_agent.generate(r_in)

    def _reward(self, pred_text: str, gold: Optional[str]) -> float:
        """
        1.0 if canonical(pred) == canonical(gold), else 0.0.
        Works for both MC (A–D/E) and numeric.
        """
        if gold is None:
            return 0.0
        pred_c = _canonicalize_pred(pred_text, gold)
        gold_c = (gold or "").strip().upper()
        # For MC, make gold canonical too
        if len(gold_c) == 1 and gold_c in "ABCDEF":
            return 1.0 if pred_c == gold_c else 0.0
        # For numeric, try float compare; otherwise string
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        pf, gf = _to_float(pred_c), _to_float(_extract_last_number(gold))
        if pf is not None and gf is not None:
            return 1.0 if abs(pf - gf) == 0 else 0.0
        return 1.0 if pred_c == (gold or "").strip() else 0.0

    def generate_and_save(self, examples: List[Dict[str, Any]], out_prefix: str = "malt"):
        agents, _host = self._make_agents()
        gen_agent = agents["generator"]
        ver_agent = agents["verifier"]
        ref_agent = agents["refiner"]

        out_dir = self.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        gen_file = out_dir / f"generator_{out_prefix}.jsonl"
        ver_file = out_dir / f"verifier_{out_prefix}.jsonl"
        ref_file = out_dir / f"refiner_{out_prefix}.jsonl"

        with gen_file.open("w") as fg, ver_file.open("w") as fv, ref_file.open("w") as fr:
            for ex in tqdm(examples, desc="Generating MALT data", unit="ex", total=len(examples)):
                q = ex.get("raw_question") or ex.get("question")
                gold = ex.get("gold") or ex.get("gold_final") or ex.get("label")
                root_id = str(uuid.uuid4())

                # K proposals (diverse if sampling is available)
                gen_outs = self._sample_generator_outputs(gen_agent, q, self.search_width)

                for g_text in gen_outs:
                    gid = str(uuid.uuid4())
                    verdict, critique, _ = self._verify(ver_agent, q, g_text)

                    if verdict == "CORRECT":
                        reward = self._reward(g_text, gold)
                        target_text = g_text if reward > 0 else (gold or g_text)
                        rec_g = self._make_node(gid, root_id, "generator",
                                                ROLE_SYSTEM_PROMPTS["generator"], q, target_text, reward=reward)
                        rec_v = self._make_node(
                            str(uuid.uuid4()),
                            gid,
                            "verifier",
                            ROLE_SYSTEM_PROMPTS["verifier"],
                            f"QUESTION:\n{q}\nSOLUTION:\n{g_text}",
                            _format_verifier_label(verdict, g_text, gold, critique),
                            reward=reward,
                        )
                        fg.write(json.dumps(rec_g) + "\n")
                        fv.write(json.dumps(rec_v) + "\n")

                    else:
                        r_text = self._refine(ref_agent, q, g_text, critique)
                        verdict2, critique2, _ = self._verify(ver_agent, q, r_text)
                        reward = self._reward(r_text, gold)
                        final_text = r_text if reward > 0 else (gold or r_text)

                        rec_g = self._make_node(gid, root_id, "generator",
                                                ROLE_SYSTEM_PROMPTS["generator"], q, final_text, reward=reward)
                        rec_r = self._make_node(str(uuid.uuid4()), gid, "refiner",
                                                ROLE_SYSTEM_PROMPTS["refiner"],
                                                f"QUESTION:\n{q}\nINCORRECT SOLUTION:\n{g_text}\nVERIFIER FEEDBACK:\n{critique or 'No critique provided.'}",
                                                r_text, reward=None)
                        rec_v2 = self._make_node(
                            str(uuid.uuid4()),
                            rec_r["trajectory_id"],
                            "verifier",
                            ROLE_SYSTEM_PROMPTS["verifier"],
                            f"QUESTION:\n{q}\nSOLUTION:\n{r_text}",
                            _format_verifier_label(verdict2, r_text, gold, critique2),
                            reward=reward,
                        )
                        fg.write(json.dumps(rec_g) + "\n")
                        fr.write(json.dumps(rec_r) + "\n")
                        fv.write(json.dumps(rec_v2) + "\n")

        print(f"[INFO] MALT datasets written to: {gen_file}, {ver_file}, {ref_file}")

    def _make_node(self, trajectory_id, parent_id, role, system, user, label, reward=None):
        return {
            "trajectory_id": trajectory_id,
            "parent_id": parent_id,
            "role": role,
            "system": system,
            "user": user,
            "label": label,
            "reward": reward,
        }
