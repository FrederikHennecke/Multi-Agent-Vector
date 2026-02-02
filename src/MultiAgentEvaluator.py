import re

from tqdm.auto import tqdm

from ModelHost import ModelHost

MC_LETTERS = "ABCDEFGHIJ"


def _extract_mc_letter_general(text: str) -> str:
    """
    Try to recover a multiple-choice letter (A-J) from free-form text.
    Prefers explicit ANSWER tags, then other common patterns.
    """
    if not text:
        return ""
    t = text.strip()
    patterns = [
        r"ANSWER\s*:\s*([A-J])\b",
        r"\b(?:OPTION|CHOICE)\s*([A-J])\b",
        r"\(([A-J])\)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    tokens = re.findall(r"\b([A-J])\b", t, flags=re.IGNORECASE)
    return tokens[-1].upper() if tokens else ""

class MultiAgentEvaluator:
    def __init__(self, agents: dict, host: ModelHost, max_data=10):
        self.agents = agents
        self.host = host
        self.max_data = max_data

    def evaluate(self, dataset):
        success, logs = 0, []
        max_turns = 3

        data_slice = dataset[:self.max_data]
        for idx, ex in enumerate(tqdm(data_slice, desc="Evaluating", unit="ex", leave=False)):
            q, gold = ex["question"], ex["label"]
            convo, verdict = [], None
            current_reasoning, current_answer = None, None
            verifier_steps = []
            last_critique = ""
            gold_str = gold if isinstance(gold, str) else str(gold)
            is_mc = isinstance(gold, str) and gold_str.strip().upper() in MC_LETTERS

            for turn in range(max_turns):
                if turn == 0:
                    g_out = self.agents["generator"].generate(q)
                    current_reasoning = g_out
                    current_answer = extract_final_answer(
                        g_out,
                        use_last_number=not is_mc,
                        prefer_letter=is_mc
                    )
                    if current_answer == "Error": current_answer = ""
                    convo.append(("GENERATOR", g_out))
                else:
                    if verdict == "INCORRECT":
                        prev_solution = current_reasoning
                        feedback = last_critique.strip() if last_critique else "No critique provided."
                        refine_input = (
                            f"QUESTION:\n{q}\n"
                            f"INCORRECT SOLUTION:\n{prev_solution}\n"
                            f"VERIFIER FEEDBACK:\n{feedback}"
                        )
                        r_out = self.agents["refiner"].generate(refine_input)
                        current_reasoning = r_out
                        current_answer = extract_final_answer(
                            r_out,
                            use_last_number=not is_mc,
                            prefer_letter=is_mc
                        )
                        if current_answer == "Error": current_answer = ""
                        convo.append(("REFINER", r_out))
                    else:
                        break

                v_in = f"QUESTION:\n{q}\nSOLUTION:\n{current_reasoning}"
                v_out = self.agents["verifier"].generate(v_in)
                verdict, critique = parse_verifier_output(v_out)
                convo.append(("VERIFIER", v_out.strip()))
                verifier_steps.append({
                    "turn": turn,
                    "verdict": verdict,
                    "critique": critique,
                    "raw": v_out.strip()
                })
                last_critique = critique
                if verdict == "CORRECT":
                    break

            correct = (current_answer == gold)
            if correct: success += 1
            logs.append({"id": idx, "question": q, "gold": gold,
                         "conversation": convo, "final": current_answer, "correct": correct,
                         "verifier_feedback": verifier_steps})
        acc = success / max(1, len(data_slice))
        print(f"Accuracy: {acc:.3f}")
        return acc, logs

    def unload(self):
        self.host.unload()

def extract_final_answer(ans_str: str, use_last_number: bool = False, prefer_letter: bool = False) -> str:
    if not ans_str or not ans_str.strip():
        return "Error"

    letter = _extract_mc_letter_general(ans_str)
    if letter:
        return letter

    if prefer_letter:
        return ""

    numbers = re.findall(r"-?\d+(?:\.\d+)?", ans_str)
    if numbers:
        return numbers[-1] if use_last_number else numbers[0]

    parts = ans_str.strip().split()
    if not parts:
        return "Error"

    try:
        return re.findall(r'\d+', parts[0])[0]
    except IndexError:
        return "Error"


def parse_verifier_output(text: str):
    """
    Extract the verdict (CORRECT/INCORRECT) and a free-form critique string from verifier output.
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
