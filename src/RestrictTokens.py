import torch
from transformers import LogitsProcessor


class RestrictTokens(LogitsProcessor):
    def __init__(self, tok):
        allowed = tok.encode("CORRECT INCORRECT", add_special_tokens=False)
        self.allowed = set(allowed)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed:
            mask[:, tid] = scores[:, tid]
        return mask