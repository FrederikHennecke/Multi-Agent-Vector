import random
import re

from config import ROLE_SYSTEM_PROMPTS
import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _parse_mc_options_from_question(q: str):
    opts, order = {}, []
    for line in q.splitlines():
        m = re.match(r"\s*([A-J]):\s*(.+)\s*$", line)
        if m:
            L = m.group(1).upper()
            txt = m.group(2).strip()
            if L not in opts:
                opts[L] = txt
                order.append(L)
    return opts, order

_RATIONALE_TEMPLATES = [
    "Eliminate {d1} and {d2} because they conflict with the key facts; {g} matches the requirement.",
    "{d1} misreads the premise and {d2} omits a necessary condition; {g} satisfies all conditions.",
    "Compare the options: {d1}/{d2} fail a crucial test; {g} is consistent with the argument.",
    "By process of elimination, {d1} and {d2} don’t fit; {g} best aligns with the evidence.",
    "Only {g} addresses the core criterion; {d1} and {d2} do not.",
]

def _make_generator_label_mc(q: str, gold_letter: str) -> str:
    opts, order = _parse_mc_options_from_question(q)
    gold_letter = gold_letter.strip().upper()
    # pick two distractors to mention
    distractors = [L for L in order if L != gold_letter]
    random.shuffle(distractors)
    d1, d2 = (distractors + ["A","B"])[:2]
    t = random.choice(_RATIONALE_TEMPLATES)
    # include the chosen option’s text once (helps grounding)
    g_text = opts.get(gold_letter, "")
    rationale = t.format(d1=d1, d2=d2, g=f"{gold_letter}: {g_text}" if g_text else gold_letter)
    return f"{rationale}\nANSWER: {gold_letter}"

def _format_verifier_label(verdict: str, predicted: str, gold: str) -> str:
    verdict_norm = "CORRECT" if str(verdict).strip().upper() == "CORRECT" else "INCORRECT"
    gold_clean = (gold or "").strip()
    pred_clean = (pred or "").strip()

    if verdict_norm == "CORRECT":
        critique = (
            f"The reasoning leads to the expected answer '{gold_clean}'."
            if gold_clean else
            "The reasoning appears consistent with the question."
        )
    else:
        if gold_clean and pred_clean:
            critique = f"The final answer '{pred_clean}' does not match the expected answer '{gold_clean}'."
        elif gold_clean:
            critique = f"The solution does not reach the expected answer '{gold_clean}'."
        elif pred_clean:
            critique = f"The final answer '{pred_clean}' is not supported by the reasoning."
        else:
            critique = "The reasoning fails to deliver a supported final answer."

    return f"VERDICT: {verdict_norm}\nCRITIQUE: {critique}"

def build_role_datasets(base_model_id, out_dir, datasets_to_use,
                        max_examples_per_dataset=200, max_new_tokens=128, split="train"):
    """
    Builds role datasets for generator/verifier/refiner.
    - MMLU-Pro multiple-choice only: compare LETTER answers.
    """
    from pathlib import Path
    import json, re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True
    )
    model.eval()

    def detect_letters_from_question(q: str):
        """Scan the question text for 'A:', 'B:', ... lines. Returns the ordered list that appears."""
        found = []
        for line in q.splitlines():
            m = re.match(r"\s*([A-J]):", line)
            if m:
                L = m.group(1)
                if L not in found:
                    found.append(L)
        # Default fallback
        return found if found else list(LETTERS[:4])

    def format_generator_output_mc(q: str, gold_letter: str) -> str:
        # Keep it short and consistent; end with ANSWER: <LETTER>
        return (
            "Read the question and options, reason briefly, and select the best option.\n"
            f"{q}\n"
            f"ANSWER: {gold_letter}"
        )

    def extract_mc_letter(text: str, valid_letters: list) -> str:
        """
        Try hard to get the predicted LETTER from a free-form output:
          1) Look for 'ANSWER: X'
          2) Last standalone letter among valid options
          3) Common patterns like '(C)', 'option C', 'Choice C'
        Returns "" if none found.
        """
        if not text:
            return ""
        t = text.strip()

        # 1) explicit 'ANSWER: X'
        m = re.search(r"ANSWER\s*:\s*([A-J])\b", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters:
                return L

        # 2) common patterns
        m = re.search(r"\b(?:option|choice)\s*([A-J])\b", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters:
                return L

        m = re.search(r"\(([A-J])\)", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters:
                return L

        # 3) last standalone letter token that is valid
        candidates = re.findall(r"\b([A-J])\b", t, flags=re.IGNORECASE)
        for L in reversed([c.upper() for c in candidates]):
            if L in valid_letters:
                return L

        return ""

    generators, verifiers, refiners = [], [], []

    # ---- Load and normalize examples across datasets ------------------------

    examples_all = []  # list of (raw_q, gold)

    for dataset_name, config in datasets_to_use:
        print(f"[INFO] Loading {dataset_name} ({split})…")
        examples = []

        if dataset_name == "mmlu_pro":
            # config = category; we cached splits under out_dir using your mmlu_pro helper
            from data.datasets_mmlu_pro import load_mmlu_pro_split
            if not config:
                raise ValueError("For mmlu_pro, 'config' must be the category name.")
            rows = load_mmlu_pro_split(out_dir, config, split="train")
            ds = rows[:max_examples_per_dataset] if max_examples_per_dataset else rows
            for r in ds:
                q = f'{r["question"].strip()} The options are {" ".join(LETTERS[i]+": "+a for i,a in enumerate(r["options"]))}'
                gold = r["gold_letter"].strip().upper()
                examples.append((q, gold))

        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Only 'mmlu_pro' is allowed.")

        for q, gold in examples:
            examples_all.append((q, gold))

    # ---- Build role records --------------------------------------------------
    for raw_q, gold in examples_all:
        letters_in_q = detect_letters_from_question(raw_q)

        # -------- Generator record --------
        gold_final = str(gold).strip().upper()
        label = format_generator_output_mc(raw_q, gold_final)
        gen_label = _make_generator_label_mc(raw_q, gold_final)
        gen_system = "You are a problem solver. Think briefly and end with: ANSWER: <LETTER>."

        generators.append({
            "raw_question": raw_q,
            "gold_final": gold_final,
            "system": gen_system,
            "user": raw_q,
            "label": gen_label,
            "task_type": "mc",
        })
        # -------- Run base model once to get a candidate --------
        chat_input = apply_chat(tok, ROLE_SYSTEM_PROMPTS["generator"], raw_q)
        inputs = tok(chat_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id
                )
            )
        gen_out_text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()

        # -------- Verifier label --------
        pred_letter = extract_mc_letter(gen_out_text, letters_in_q)
        ver_label = "CORRECT" if pred_letter == gold_final else "INCORRECT"

        verifier_user = f"QUESTION:\n{raw_q}\nSOLUTION:\n{gen_out_text}"
        verifiers.append({
            "raw_question": raw_q,
            "gold_final": gold_final,
            "system": ROLE_SYSTEM_PROMPTS["verifier"],
            "user": verifier_user,
            "label": ver_label
        })

        # -------- Refiner (only incorrect cases) --------
        if ver_label == "INCORRECT":
            refiner_user = f"QUESTION:\n{raw_q}\nINCORRECT SOLUTION:\n{gen_out_text}"
            refiners.append({
                "raw_question": raw_q,
                "gold_final": gold_final,
                "system": ROLE_SYSTEM_PROMPTS["refiner"],
                "user": refiner_user,
                "label": label  # the correct CoT we built above
            })

    # ---- Save ---------------------------------------------------------------

    def save_jsonl(filename, data):
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save_jsonl("generator.jsonl", generators)
    save_jsonl("verifier.jsonl", verifiers)
    save_jsonl("refiner.jsonl", refiners)

    print(f"[INFO] Role datasets saved to {out_dir}")
    print(f"[INFO] generator={len(generators)}, verifier={len(verifiers)}, refiner={len(refiners)}")


def regenerate_verifier_refiner_with_lora_gen(
    base_model_id,
    lora_gen_path,
    data_dir,
    max_new_tokens=256,
    max_train_data=200,
    synth_refiner_when_no_incorrect=True,
):
    """
    Rebuild verifier/refiner datasets using the LoRA'd generator.
    - Never drops the whole verifier set due to class imbalance.
    - MC-only verification (letters).
    - Optionally synthesizes a small refiner set if no incorrect samples occurred.
    """
    import random, re, json
    from pathlib import Path
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

    tok = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True
    )
    gen = PeftModel.from_pretrained(base, lora_gen_path)
    gen.eval()

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ---- helpers -----------------------------------------------------------
    def detect_letters_from_question(q: str):
        found = []
        for line in q.splitlines():
            m = re.match(r"\s*([A-J]):", line)
            if m:
                L = m.group(1)
                if L not in found:
                    found.append(L)
        return found if found else list(LETTERS[:4])

    def extract_mc_letter(text: str, valid_letters: list) -> str:
        if not text:
            return ""
        t = text.strip()
        m = re.search(r"ANSWER\s*:\s*([A-J])\b", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters: return L
        m = re.search(r"\b(?:option|choice)\s*([A-J])\b", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters: return L
        m = re.search(r"\(([A-J])\)", t, flags=re.IGNORECASE)
        if m:
            L = m.group(1).upper()
            if L in valid_letters: return L
        candidates = re.findall(r"\b([A-J])\b", t, flags=re.IGNORECASE)
        for L in reversed([c.upper() for c in candidates]):
            if L in valid_letters:
                return L
        return ""

    # --- read generator training set as seed prompts
    gen_recs = [json.loads(l) for l in open(Path(data_dir) / "generator.jsonl") if l.strip()]
    if max_train_data:
        gen_recs = gen_recs[:max_train_data]

    correct_cases, incorrect_cases = [], []

    for rec in gen_recs:
        raw_q = rec["raw_question"]
        gold_final = str(rec["gold_final"]).strip()

        # Build chat & generate with LoRA generator
        g_input = apply_chat(tok, ROLE_SYSTEM_PROMPTS["generator"], raw_q)
        enc = tok(g_input, return_tensors="pt").to(gen.device)
        with torch.no_grad():
            out_ids = gen.generate(
                **enc,
                generation_config=GenerationConfig(
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )
            )
        g_out = tok.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Decide CORRECT/INCORRECT with MC-aware parser
        letters_in_q = detect_letters_from_question(raw_q)
        pred = extract_mc_letter(g_out, letters_in_q)
        lbl = "CORRECT" if pred == gold_final else "INCORRECT"

        v_user = f"QUESTION:\n{raw_q}\nSOLUTION:\n{g_out}"
        v_entry = {
            "raw_question": raw_q,
            "gold_final": gold_final,
            "system": ROLE_SYSTEM_PROMPTS["verifier"],
            "user": v_user,
            "label": _format_verifier_label(lbl, pred, gold_final),
        }
        if lbl == "CORRECT":
            correct_cases.append(v_entry)
        else:
            incorrect_cases.append(v_entry)

    # ---- Build verifier set without wiping it ------------------------------
    if len(correct_cases) == 0 or len(incorrect_cases) == 0:
        # Fall back to unbalanced; never output empty
        print(f"[WARN] Verifier class imbalance: correct={len(correct_cases)}, incorrect={len(incorrect_cases)}. "
              f"Falling back to unbalanced set.")
        balanced = correct_cases + incorrect_cases
        random.shuffle(balanced)
    else:
        n = min(len(correct_cases), len(incorrect_cases))
        balanced = random.sample(correct_cases, n) + random.sample(incorrect_cases, n)
        random.shuffle(balanced)

    with open(Path(data_dir) / "verifier.jsonl", "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Build refiner set --------------------------------------------------
    ref_data = []
    # Natural incorrects first
    for r in incorrect_cases:
        raw_q = r["raw_question"]
        # Recover the generator's solution from verifier input (safe parse)
        s = r["user"]
        g_out = s.split("SOLUTION:\n", 1)[1] if "SOLUTION:\n" in s else ""
        label_text = r.get("label", "")
        critique = ""
        if isinstance(label_text, str):
            m = re.search(r"CRITIQUE\s*:\s*(.*)", label_text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                critique = m.group(1).strip()
        if not critique:
            critique = "No critique provided."

        ref_user = (
            f"QUESTION:\n{raw_q}\n"
            f"INCORRECT SOLUTION:\n{g_out}\n"
            f"VERIFIER FEEDBACK:\n{critique}"
        )
        # Use the *generator* CoT label we stored earlier as target (from generator.jsonl)
        gold_cot = next((gr["label"] for gr in gen_recs if gr["raw_question"] == raw_q), None)
        if gold_cot:
            ref_data.append({
                "raw_question": raw_q,
                "gold_final": r["gold_final"],
                "system": ROLE_SYSTEM_PROMPTS["refiner"],
                "user": ref_user,
                "label": gold_cot
            })

    # If no incorrects at all, optionally synthesize a tiny set
    if not ref_data and synth_refiner_when_no_incorrect:
        print("[WARN] No incorrect generator outputs; synthesizing a small refiner set.")
        def synth_wrong_for(q: str, gold: str) -> str:
            letters = detect_letters_from_question(q)
            letters = letters if letters else list(LETTERS[:4])
            wrongs = [L for L in letters if L != gold]
            return f"… ANSWER: {random.choice(wrongs) if wrongs else 'A'}"

        k = min(50, len(gen_recs))
        for gr in random.sample(gen_recs, k=k):
            raw_q = gr["raw_question"]; gold_final = gr["gold_final"]
            wrong = synth_wrong_for(raw_q, gold_final)
            ref_user = (
                f"QUESTION:\n{raw_q}\n"
                f"INCORRECT SOLUTION:\n{wrong}\n"
                f"VERIFIER FEEDBACK:\nNo critique provided."
            )
            ref_data.append({
                "raw_question": raw_q,
                "gold_final": gold_final,
                "system": ROLE_SYSTEM_PROMPTS["refiner"],
                "user": ref_user,
                "label": gr["label"]
            })

    with open(Path(data_dir) / "refiner.jsonl", "w", encoding="utf-8") as f:
        for r in ref_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] Verifier: {len(balanced)} (correct={len(correct_cases)}, incorrect={len(incorrect_cases)})")
    print(f"[INFO] Refiner:  {len(ref_data)} (from incorrect={len(incorrect_cases)})")


def apply_chat(tok, sys_prompt, user_content):
    """Apply chat template ONLY when generating."""
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return f"{sys_prompt}\n\n{user_content}"
