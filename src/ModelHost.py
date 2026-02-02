from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor
from peft import PeftModel
import torch.nn.functional as F

class ModelHost:
    def __init__(self, base_model_id, layer_idx=20, device: str | None = None):
        self.base_model_id = base_model_id
        self.layer_idx = layer_idx
        self.requested_device = None if device in (None, "auto") else device
        self.tok = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
        load_kwargs = dict(local_files_only=True)
        device_map = "auto"
        if self.requested_device is not None:
            device_map = None
            load_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.requested_device else torch.float32
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
        if device_map is None:
            self.model.to(self.requested_device)
        self._orig_layer_forward = None
        self._current_vec = None
        self._lora_loaded = False

    def _model_device(self):
        if self.requested_device:
            return torch.device(self.requested_device)
        return next(self.model.parameters()).device

    def load_lora_adapters(self, adapters: dict):
        # adapters: role -> adapter_dir
        if self._lora_loaded:
            return
        # prime with generator adapter and load others by name
        self.model = PeftModel.from_pretrained(self.model, adapters["generator"], adapter_name="generator")
        for role, path in adapters.items():
            if role != "generator":
                self.model.load_adapter(path, adapter_name=role)
        self.model.set_adapter("generator")
        self._lora_loaded = True

    def set_lora_role(self, role: str):
        self.model.set_adapter(role)

    def enable_vector_patch(self):
        layer = self.model.model.layers[self.layer_idx]
        if self._orig_layer_forward is None:
            self._orig_layer_forward = layer.forward

    def set_vector(self, vec: torch.Tensor):
        # vec must be on model device and match model dtype
        target_dtype = next(self.model.parameters()).dtype
        self._current_vec = vec.to(self._model_device(), dtype=target_dtype)
        layer = self.model.model.layers[self.layer_idx]

        def fast_forward(hidden_states, *args, **kwargs):
            out = self._orig_layer_forward(hidden_states, *args, **kwargs)
            if isinstance(out, tuple):
                out_list = list(out)
                out_list[0] = out_list[0].add_(self._current_vec)
                return tuple(out_list)
            else:
                return out.add_(self._current_vec)

        layer.forward = fast_forward

    def clear_vector_patch(self):
        if self._orig_layer_forward is not None:
            self.model.model.layers[self.layer_idx].forward = self._orig_layer_forward
            self._orig_layer_forward = None
            self._current_vec = None

    def build_chat_prompt(self, system_msg: str, user_msg: str) -> str:
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True
            )
        return f"{system_msg}\n\n{user_msg}"

    def generate(self, prompt: str, max_new_tokens: int, logits_processors=None) -> str:
        device = self._model_device()
        inputs = self.tok(prompt, return_tensors="pt").to(device)
        out_ids = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id
            ),
            logits_processor=logits_processors
        )
        return self.tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def unload(self):
        self.clear_vector_patch()
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decide_logprob(self, prompt: str, candidates=("CORRECT", "INCORRECT")):
        # Returns (winner, margin) where margin is logprob difference between top-2
        device = self._model_device()
        enc_p = self.tok(prompt, return_tensors="pt").to(device)
        prompt_len = enc_p["input_ids"].shape[1]
        scores = []
        for cand in candidates:
            enc_c = self.tok(cand, add_special_tokens=False, return_tensors="pt").to(device)
            full_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1)
            full_mask = torch.cat([enc_p["attention_mask"], enc_c["attention_mask"]], dim=1)
            with torch.no_grad():
                logits = self.model(full_ids, attention_mask=full_mask).logits  # [1, T, V]
            L = enc_c["input_ids"].shape[1]
            # Next-token logprobs for candidate tokens
            lsm = F.log_softmax(logits[0, prompt_len - 1: prompt_len + L - 1, :], dim=-1)  # [L, V]
            tok_ids = enc_c["input_ids"][0]  # [L]
            tok_ll = lsm.gather(-1, tok_ids.unsqueeze(-1)).sum().item()
            scores.append(tok_ll)
        # Pick best
        best_idx = 0 if scores[0] >= scores[1] else 1
        other_idx = 1 - best_idx
        margin = scores[best_idx] - scores[other_idx]
        return candidates[best_idx], margin
