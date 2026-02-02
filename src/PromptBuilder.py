import random
import json
import re
from pathlib import Path
from typing import Optional

from ModelHost import ModelHost


def _extract_verifier_verdict(text: str) -> str:
    matches = re.findall(r"\b(CORRECT|INCORRECT)\b", (text or "").upper())
    return matches[-1] if matches else ""


class PromptBuilder:
    def build(self, host: ModelHost, system_msg: str, task: str) -> str:
        return host.build_chat_prompt(system_msg, task)

class FewShotPromptBuilder(PromptBuilder):
    def __init__(self, role: str, fewshot_pool: list, k: int = 2,
                 balance_for_verifier: bool = True, rng: Optional[random.Random] = None):
        self.role = role
        self.pool = fewshot_pool or []
        self.k = k
        self.balance_for_verifier = balance_for_verifier
        self.rng = rng

    @staticmethod
    def _is_mcq(text: str) -> bool:
        # Simple heuristic: lines starting with A:, B:, C:, D:
        return any(f"\n{opt}:" in text for opt in ["A", "B", "C", "D"])

    def _sample(self, population, k: int):
        if not population or k <= 0:
            return []
        k = min(k, len(population))
        sampler = self.rng.sample if self.rng else random.sample
        return sampler(population, k)

    def _select_examples(self, task: str):
        if not self.pool:
            return []

        want_mcq = self._is_mcq(task)
        same_domain = []
        other_domain = []
        for ex in self.pool:
            raw_q = ex.get("raw_question", "")
            if self._is_mcq(raw_q) == want_mcq:
                same_domain.append(ex)
            else:
                other_domain.append(ex)

        base = same_domain if same_domain else other_domain

        # Verifier: optionally balance CORRECT/INCORRECT
        if self.role == "verifier" and self.balance_for_verifier:
            def verdict_of(ex):
                return ex.get("verdict") or _extract_verifier_verdict(ex.get("label", ""))
            correct = [ex for ex in base if verdict_of(ex) == "CORRECT"]
            incorrect = [ex for ex in base if verdict_of(ex) == "INCORRECT"]
            n_each = max(1, self.k // 2)
            chosen = []
            if correct:
                chosen.extend(self._sample(correct, n_each))
            if incorrect:
                chosen.extend(self._sample(incorrect, n_each))
            # Top up if needed
            remain = self.k - len(chosen)
            if remain > 0:
                pool_rest = [ex for ex in base if ex not in chosen]
                if pool_rest:
                    chosen.extend(self._sample(pool_rest, remain))
            return chosen[:self.k]

        # Default: sample k from base
        return self._sample(base, self.k)

    def build(self, host: ModelHost, system_msg: str, task: str) -> str:
        chosen = self._select_examples(task)
        parts = []
        for i, ex in enumerate(chosen, 1):
            if self.role == "generator":
                parts.append(f"### Example {i}\nQuestion:\n{ex.get('raw_question','')}\n{ex.get('label','')}")
            elif self.role == "verifier":
                gen_out = ex.get("gen_output", "")
                label_text = ex.get("label", "")
                if not isinstance(label_text, str):
                    label_text = str(label_text)
                label_text = label_text.strip()
                if not label_text:
                    label_text = _extract_verifier_verdict(ex.get("label", ""))
                parts.append(
                    f"### Example {i}\nQUESTION:\n{ex.get('raw_question','')}\nSOLUTION:\n{gen_out}\n{label_text}"
                )
            elif self.role == "refiner":
                wrong = ex.get("wrong_output", "")
                feedback = ex.get("verifier_feedback", "")
                feedback_block = (
                    f"\nVERIFIER FEEDBACK:\n{feedback.strip()}"
                    if isinstance(feedback, str) and feedback.strip()
                    else ""
                )
                parts.append(
                    "### Example {i}\nQUESTION:\n{question}\nINCORRECT SOLUTION:\n{wrong}{fb}\n{label}".format(
                        i=i,
                        question=ex.get("raw_question", ""),
                        wrong=wrong,
                        fb=feedback_block,
                        label=ex.get("label", "")
                    )
                )

        # Strong separator + explicit instruction
        if self.role == "verifier":
            trailer = (
                "### Now decide for the following. Respond with:\n"
                "VERDICT: CORRECT or INCORRECT\n"
                "CRITIQUE: <brief justification>"
            )
        elif self.role in ("generator","refiner"):
            trailer = "### Now solve the following task. End your answer with: ANSWER: <answer>."
        else:
            trailer = "### Now handle the following task."

        user_msg = ("\n\n".join(parts) + "\n\n" + trailer + "\n\n" + task) if parts else task
        return host.build_chat_prompt(system_msg, user_msg)

def load_fewshot_pool(data_dir: Path, role: str, cap: int = 200):
    """
    Load a small pool of few-shot exemplars.
    Supports both classic role datasets (with `input`) and MALT rollouts (`user`/`reward`).
    Applies light filtering to drop low-quality or reward-zero rows so prompt_fs does not learn from incorrect answers.
    """
    import re

    def _clean_label(lbl: str) -> str:
        if not isinstance(lbl, str):
            return ""
        lbl = lbl.strip()
        # Prefer a single MC letter if present
        m = re.search(r"\b([A-J])\b", lbl, flags=re.IGNORECASE)
        return m.group(1).upper() if m else lbl

    def _parse_malt_generator(rec: dict) -> dict:
        raw_q = rec.get("raw_question") or rec.get("user") or rec.get("input", "")
        label = _clean_label(rec.get("label", ""))
        return {"raw_question": raw_q, "label": label}

    def _parse_malt_verifier(rec: dict) -> dict:
        user = rec.get("user", "")
        raw_q = ""
        gen_out = ""
        if "QUESTION:" in user and "SOLUTION:" in user:
            parts = user.split("QUESTION:", 1)[1]
            if "SOLUTION:" in parts:
                raw_q, gen_out = parts.split("SOLUTION:", 1)
        raw_q = raw_q.strip() or rec.get("raw_question", "")
        gen_out = gen_out.strip()
        label_text = rec.get("label", "")
        verdict = _extract_verifier_verdict(label_text) or _extract_verifier_verdict(rec.get("label", ""))
        return {
            "raw_question": raw_q,
            "gen_output": gen_out,
            "label": label_text,
            "verdict": verdict
        }

    def _parse_malt_refiner(rec: dict) -> dict:
        user = rec.get("user", "")
        raw_q = rec.get("raw_question", "")
        wrong_output = ""
        feedback = ""
        if "QUESTION:" in user and "INCORRECT SOLUTION:" in user:
            q_part = user.split("QUESTION:", 1)[1]
            if "INCORRECT SOLUTION:" in q_part:
                raw_q, rest = q_part.split("INCORRECT SOLUTION:", 1)
                raw_q = raw_q.strip()
                if "VERIFIER FEEDBACK:" in rest:
                    wrong_output, fb = rest.split("VERIFIER FEEDBACK:", 1)
                    wrong_output = wrong_output.strip()
                    feedback = fb.strip()
                else:
                    wrong_output = rest.strip()
        return {
            "raw_question": raw_q,
            "wrong_output": wrong_output,
            "verifier_feedback": feedback,
            "label": rec.get("label", "")
        }

    pool = []
    with open(data_dir / f"{role}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= cap:
                break
            rec = json.loads(line)
            reward = rec.get("reward", None)

            if "malt" in role:
                # Handle MALT rollouts (no `input` field)
                if role.startswith("generator"):
                    if reward is not None and reward <= 0:
                        # For generator exemplars, drop incorrect rollouts
                        continue
                    entry = _parse_malt_generator(rec)
                    if not entry["raw_question"] or not entry["label"]:
                        continue
                    if len(entry["label"]) > 3 and not re.fullmatch(r"[A-J]", entry["label"]):
                        continue
                elif role.startswith("verifier"):
                    # Keep both correct and incorrect to avoid biasing verifier
                    entry = _parse_malt_verifier(rec)
                    if not entry["raw_question"] or not entry["gen_output"] or not entry.get("verdict"):
                        continue
                elif role.startswith("refiner"):
                    # Reward is almost always missing for refiner; keep examples if well-formed
                    entry = _parse_malt_refiner(rec)
                    if not entry["raw_question"] or not entry["wrong_output"]:
                        continue
                else:
                    entry = {"raw_question": rec.get("raw_question", ""), "label": rec.get("label", "")}
            else:
                # Classic role datasets with `input`
                entry = {
                    "raw_question": rec.get("raw_question", ""),
                    "label": rec.get("label", "")
                }
                s = rec.get("input", "")
                if "verifier" in role:
                    key = "SOLUTION:\n"
                    entry["gen_output"] = s.split(key, 1)[1] if key in s else ""
                if "refiner" in role:
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

            entry["verdict"] = _extract_verifier_verdict(entry.get("label", ""))
            pool.append(entry)
    return pool


class TaskOnlyPromptBuilder(PromptBuilder):
    def build(self, host: ModelHost, system_msg: str, task: str) -> str:
        # No system prompt, no chat template
        return task

class MinimalHintsPromptBuilder(PromptBuilder):
    def __init__(self, role: str):
        self.role = role
    def build(self, host: ModelHost, system_msg: str, task: str) -> str:
        # No system prompt; add minimal output-format hints only
        if self.role == "verifier":
            user = (
                task
                + "\nRespond with:\nVERDICT: CORRECT or INCORRECT\nCRITIQUE: <brief justification>"
            )
        elif self.role in ("generator", "refiner"):
            user = task + "\nEnd your answer with: ANSWER: <answer>."
        else:
            user = task
        return user
