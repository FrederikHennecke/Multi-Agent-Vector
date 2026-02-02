import torch

from ModelHost import ModelHost
from PromptBuilder import PromptBuilder
from RestrictTokens import RestrictTokens
from config import ROLE_SYSTEM_PROMPTS


class Agent:
    def __init__(self, role: str, host: ModelHost, prompt_builder: PromptBuilder,
                 restrict_to_ci: bool = False, max_new_tokens: int = 1024):
        self.role = role
        self.host = host
        self.prompt_builder = prompt_builder
        self.restrict_to_ci = restrict_to_ci
        self.max_new_tokens = max_new_tokens

    def generate(self, task: str) -> str:
        sys_msg = ROLE_SYSTEM_PROMPTS[self.role]
        prompt = self.prompt_builder.build(self.host, sys_msg, task)

        logits_processors = None
        max_new = self.max_new_tokens
        if self.restrict_to_ci:
            logits_processors = [RestrictTokens(self.host.tok)]
            max_new = 1

        return self.host.generate(prompt, max_new, logits_processors)
