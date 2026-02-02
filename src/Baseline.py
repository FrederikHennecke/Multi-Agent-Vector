from ModelHost import ModelHost
from PromptBuilder import PromptBuilder


class BaselinePromptBuilder(PromptBuilder):
    def build(self, host: ModelHost, system_msg: str, task: str) -> str:
        system_prompt = system_msg or (
            "You are a problem solver. Think step-by-step and include all reasoning.\n"
            "At the end, output: ANSWER: <answer>."
        )
        user_msg = (
            f"Question:\n{task}\n\n"
            "Provide your reasoning and finish with: ANSWER: <answer>."
        )
        return host.build_chat_prompt(system_prompt, user_msg)

class ConstantAgent:
    def __init__(self, text: str):
        self.text = text
    def generate(self, task: str) -> str:
        return self.text
