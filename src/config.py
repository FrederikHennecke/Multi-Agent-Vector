# config.py
ROLE_SYSTEM_PROMPTS = {
    "generator": "You are a problem solver. Think step-by-step and include all reasoning.\nAt the end, output: ANSWER: <answer>.",
    "verifier": (
        "You are a Verifier. You will be given a question and the reasoning of another model. "
        "Determine if the final answer is correct. Respond with a verdict and a brief critique.\n"
        "Format your reply exactly as:\n"
        "VERDICT: CORRECT or INCORRECT\n"
        "CRITIQUE: <short justification highlighting any issues or confirming correctness>."
    ),
    "refiner": "You are a Refiner. Given a question and incorrect reasoning, produce improved reasoning and end with: ANSWER: <answer>."
}

ROLE_INTEGRITY_SYSTEM_PROMPT = (
    "You grade role integrity in a multi-agent conversation. "
    "Use the rubric and return ONLY a JSON object with integer scores 0-2.\n\n"
    "Rubric:\n"
    "- g_reasoning: 0 no reasoning; 1 some reasoning; 2 coherent step-by-step reasoning\n"
    "- g_final_format: 0 missing final answer; 1 answer given but format inconsistent; "
    "2 ends with 'FINAL ANSWER: <letter>' or 'ANSWER: <letter>'\n"
    "- v_verdict: 0 no verdict; 1 ambiguous; 2 'VERDICT: CORRECT' or 'VERDICT: INCORRECT'\n"
    "- v_grounding: 0 generic critique; 1 partly grounded; 2 specific and grounded in the question/options\n"
    "- r_revision: 0 ignores critique; 1 acknowledges critique; 2 integrates critique and fixes\n\n"
    "Output JSON with keys: g_reasoning, g_final_format, v_verdict, v_grounding, r_revision."
)

ROLE_INTEGRITY_USER_TEMPLATE = (
    "Question stem:\n{question_stem}\n\n"
    "Options:\n{options}\n\n"
    "Generator response:\n{generator}\n\n"
    "Verifier response:\n{verifier}\n\n"
    "Refiner response (if any):\n{refiner}\n\n"
    "Verifier critique used for refiner (if available):\n{refiner_critique}\n\n"
    "Scores:"
)
