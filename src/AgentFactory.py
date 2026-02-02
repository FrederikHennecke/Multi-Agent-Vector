from pathlib import Path
import random
from typing import Optional

import torch

from Agent import Agent
from Baseline import BaselinePromptBuilder, ConstantAgent
from ModelHost import ModelHost
from PromptBuilder import (
    PromptBuilder,
    FewShotPromptBuilder,
    load_fewshot_pool,
    MinimalHintsPromptBuilder,
    TaskOnlyPromptBuilder,
)
from config import ROLE_SYSTEM_PROMPTS

class AgentFactory:
    def __init__(self, base_model_id, data_dir: Path = None, layer_idx: int = 20,
                 lora_adapters: dict = None, steering_paths: dict = None):
        self.base_model_id = base_model_id
        self.data_dir = data_dir
        self.layer_idx = layer_idx
        self.lora_adapters = lora_adapters
        self.steering_paths = steering_paths
        self.max_new_tokens = 1024

    def build_baseline(self):
        host = ModelHost(self.base_model_id, self.layer_idx)
        pb = BaselinePromptBuilder()
        # Generator: short answer-only decoding, no restrictions
        gen = Agent("generator", host, pb, restrict_to_ci=False, max_new_tokens=self.max_new_tokens)
        # Verifier: constant CORRECT so the loop stops after the generator
        ver = ConstantAgent("CORRECT")
        # Refiner: never used (verifier stops the loop), but provide a valid Agent
        ref = Agent("refiner", host, pb, restrict_to_ci=False, max_new_tokens=self.max_new_tokens)
        agents = {"generator": gen, "verifier": ver, "refiner": ref}
        return agents, host

    def build_prompt(self, verifier_decision: str = "gen"):
        host = ModelHost(self.base_model_id, self.layer_idx)
        pb = {
            "generator": PromptBuilder(),
            "verifier": PromptBuilder(),
            "refiner": PromptBuilder()
        }
        return {
            "generator": Agent("generator", host, pb["generator"], restrict_to_ci=False, max_new_tokens=self.max_new_tokens),
            "verifier": (self._make_verifier_gen(host, pb["verifier"]) if verifier_decision=="gen"
                     else self._make_verifier_logprob_prompt(host, pb["verifier"])),
            "refiner": Agent("refiner", host, pb["refiner"], restrict_to_ci=False, max_new_tokens=self.max_new_tokens)
        }, host

    def build_prompt_fs(self, k=3, verifier_decision: str = "gen", seed: Optional[int] = None):
        host = ModelHost(self.base_model_id, self.layer_idx)
        pools = {r: load_fewshot_pool(self.data_dir, r, cap=200) for r in [f"generator_malt_v1",f"verifier_malt_v1",f"refiner_malt_v1"]}
        pools["generator"] = pools.pop("generator_malt_v1")
        pools["verifier"] = pools.pop("verifier_malt_v1")
        pools["refiner"] = pools.pop("refiner_malt_v1")
        rngs = {}
        roles = ["generator", "verifier", "refiner"]
        if seed is not None:
            rngs = {role: random.Random(seed + idx) for idx, role in enumerate(roles)}
        else:
            rngs = {role: None for role in roles}
        pb = {
            "generator": FewShotPromptBuilder("generator", pools["generator"], k, rng=rngs["generator"]),
            "verifier": FewShotPromptBuilder("verifier", pools["verifier"], k, balance_for_verifier=True, rng=rngs["verifier"]),
            "refiner": FewShotPromptBuilder("refiner", pools["refiner"], k, rng=rngs["refiner"]),
        }
        return {
            "generator": Agent("generator", host, pb["generator"], restrict_to_ci=False, max_new_tokens=self.max_new_tokens),
            "verifier": (self._make_verifier_gen(host, pb["verifier"]) if verifier_decision == "gen"
                         else self._make_verifier_logprob_prompt(host, pb["verifier"])),
            "refiner": Agent("refiner", host, pb["refiner"], restrict_to_ci=False, max_new_tokens=self.max_new_tokens)
        }, host

    def build_vector(self, use_prompts: bool = True, minimal_hints: bool = False, verifier_decision: str = "gen"):
        host = ModelHost(self.base_model_id, self.layer_idx)
        host.enable_vector_patch()
        vectors = {r: torch.load(self.steering_paths[r], map_location="cpu") for r in ["generator", "verifier", "refiner"]}

        # Choose prompt builders per role
        if use_prompts:
            pb = {r: PromptBuilder() for r in ["generator", "verifier", "refiner"]}
        else:
            if minimal_hints:
                pb = {r: MinimalHintsPromptBuilder(r) for r in ["generator", "verifier", "refiner"]}
            else:
                pb = {r: TaskOnlyPromptBuilder() for r in ["generator", "verifier", "refiner"]}

        def make_agent(role, restrict_to_ci, max_new):
            class VectorAgent(Agent):
                def generate(self_inner, task: str) -> str:
                    host.set_vector(vectors[role])
                    return super(VectorAgent, self_inner).generate(task)
            return VectorAgent(role, host, pb[role], restrict_to_ci=restrict_to_ci, max_new_tokens=max_new)

        if verifier_decision == "gen":
            verifier_agent = self._make_verifier_gen(host, pb["verifier"], vector_tensor=vectors["verifier"])
        else:
            verifier_agent = self._make_verifier_logprob_vector(host, pb["verifier"], vectors["verifier"])

        agents = {
            "generator": make_agent("generator", False, self.max_new_tokens),
            "verifier": verifier_agent,
            "refiner": make_agent("refiner", False, self.max_new_tokens)
        }
        return agents, host

    def build_lora(self, use_prompts: bool = True, minimal_hints: bool = False, verifier_decision: str = "gen"):
        host = ModelHost(self.base_model_id, self.layer_idx)
        host.load_lora_adapters(self.lora_adapters)

        if use_prompts:
            pb = {r: PromptBuilder() for r in ["generator", "verifier", "refiner"]}
        else:
            if minimal_hints:
                pb = {r: MinimalHintsPromptBuilder(r) for r in ["generator", "verifier", "refiner"]}
            else:
                pb = {r: TaskOnlyPromptBuilder() for r in ["generator", "verifier", "refiner"]}

        def make_agent(role, restrict_to_ci, max_new):
            class LoRAAgent(Agent):
                def generate(self_inner, task: str) -> str:
                    host.set_lora_role(role)
                    return super(LoRAAgent, self_inner).generate(task)
            return LoRAAgent(role, host, pb[role], restrict_to_ci=restrict_to_ci, max_new_tokens=max_new)

        agents = {
            "generator": make_agent("generator", False, self.max_new_tokens),
            "verifier": (self._make_verifier_gen(host, pb["verifier"]) if verifier_decision=="gen"
                     else self._make_verifier_logprob_lora(host, pb["verifier"])),
            "refiner": make_agent("refiner", False, self.max_new_tokens)
        }
        return agents, host

    def _make_verifier_gen(self, host, prompt_builder, max_new=1024, vector_tensor=None):
        # Restricted-token generation; optionally set a steering vector before decoding
        class VerifierAgent(Agent):
            def generate(self_inner, task: str) -> str:
                if vector_tensor is not None:
                    host.set_vector(vector_tensor)
                return super(VerifierAgent, self_inner).generate(task)

        return VerifierAgent("verifier", host, prompt_builder, restrict_to_ci=False, max_new_tokens=max_new)

    def _make_verifier_logprob_prompt(self, host, prompt_builder):
        outer_self = self

        class VerifierLogProb(Agent):
            def __init__(self, host_, pb_):
                super().__init__("verifier", host_, pb_, restrict_to_ci=False, max_new_tokens=1)

            def generate(self, task: str) -> str:
                sys_msg = ROLE_SYSTEM_PROMPTS["verifier"]
                prompt = self.prompt_builder.build(self.host, sys_msg, task)
                verdict, margin = self.host.decide_logprob(prompt, candidates=("CORRECT", "INCORRECT"))
                return outer_self._format_logprob_response(verdict, margin)

        return VerifierLogProb(host, prompt_builder)

    def _make_verifier_logprob_vector(self, host, prompt_builder, vec_tensor):
        outer_self = self

        class VerifierVectorLogProb(Agent):
            def __init__(self, host_, pb_, vec_):
                super().__init__("verifier", host_, pb_, restrict_to_ci=False, max_new_tokens=1)
                self.vec = vec_

            def generate(self, task: str) -> str:
                # Apply steering vector per-call
                host.set_vector(self.vec)
                sys_msg = ROLE_SYSTEM_PROMPTS["verifier"]
                prompt = self.prompt_builder.build(self.host, sys_msg, task)
                verdict, margin = self.host.decide_logprob(prompt, candidates=("CORRECT", "INCORRECT"))
                return outer_self._format_logprob_response(verdict, margin)

        return VerifierVectorLogProb(host, prompt_builder, vec_tensor)

    def _make_verifier_logprob_lora(self, host, prompt_builder):
        outer_self = self

        class VerifierLoRALogProb(Agent):
            def __init__(self, host_, pb_):
                super().__init__("verifier", host_, pb_, restrict_to_ci=False, max_new_tokens=1)

            def generate(self, task: str) -> str:
                host.set_lora_role("verifier")
                sys_msg = ROLE_SYSTEM_PROMPTS["verifier"]
                prompt = self.prompt_builder.build(self.host, sys_msg, task)
                verdict, margin = self.host.decide_logprob(prompt, candidates=("CORRECT", "INCORRECT"))
                return outer_self._format_logprob_response(verdict, margin)

        return VerifierLoRALogProb(host, prompt_builder)

    @staticmethod
    def _format_logprob_response(verdict: str, margin: float) -> str:
        verdict_clean = "CORRECT" if str(verdict).strip().upper() == "CORRECT" else "INCORRECT"
        if margin is None:
            critique = (
                "Logprob comparison was inconclusive; double-check the reasoning and final answer."
                if verdict_clean == "INCORRECT"
                else "Logprob comparison favored CORRECT, but provide a quick sanity check."
            )
        else:
            margin_abs = abs(float(margin))
            if margin_abs >= 5:
                confidence = "very strong"
            elif margin_abs >= 2:
                confidence = "strong"
            elif margin_abs >= 0.5:
                confidence = "moderate"
            else:
                confidence = "low"

            if verdict_clean == "CORRECT":
                critique = (
                    f"Logprob scoring favors CORRECT with {confidence} confidence (margin {margin_abs:.2f}); "
                    "the solution appears internally consistent."
                )
            else:
                critique = (
                    f"Logprob scoring favors INCORRECT with {confidence} confidence (margin {margin_abs:.2f}); "
                    "there is likely an error in the reasoning or the final answer."
                )

        return f"VERDICT: {verdict_clean}\nCRITIQUE: {critique}"
