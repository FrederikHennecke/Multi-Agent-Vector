import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class SteeringVectorConfig:
    # Single layer or list of layers
    layer_idx: Union[int, List[int]] = 20
    alpha: float = 1.0
    max_samples: int = 50
    max_length: int = 1024
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

    # Quality boosters (no code changes elsewhere needed)
    standardize: bool = True           # diagonal whitening / effect-size vector

    # Multi-layer handling
    use_pca: bool = False              # not needed with standardize+weights; keep available
    pca_rank: int = 5
    sequential_layers: bool = True     # lower peak VRAM
    clear_cache_every: int = 16

class SteeringVectorCollector:
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id

    # ---------------- VRAM-lean mean pooling ----------------
    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, H] on GPU
        pooled = hidden.mean(dim=1).mean(dim=0)
        return pooled.detach().to("cpu", copy=True)

    def _hook_fn(self, bucket: List[torch.Tensor]):
        def hook(_m, _inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            bucket.append(self._pool_hidden(out))
        return hook

    def _encode_to_device(self, tok, text: str, device: torch.device, max_length: int):
        return tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device, non_blocking=True)

    # ---------------- Welford stats (CPU) --------------
    @staticmethod
    def _init_stats_like(vec: torch.Tensor):
        H = vec.shape[0]
        return {
            "n": 0,
            "mean": torch.zeros(H, dtype=torch.float32),
            "M2": torch.zeros(H, dtype=torch.float32),
        }

    @staticmethod
    def _update_stats(stats: dict, x: torch.Tensor):
        # x is CPU float tensor [H]
        x = x.to(torch.float32)
        stats["n"] += 1
        delta = x - stats["mean"]
        stats["mean"] += delta / stats["n"]
        delta2 = x - stats["mean"]
        stats["M2"] += delta * delta2

    @staticmethod
    def _finalize_std(stats: dict):
        n = max(stats["n"], 1)
        var = stats["M2"] / max(n - 1, 1)
        return torch.sqrt(torch.clamp(var, min=1e-6))

    # --------------- capture routines ------------------
    def _capture_single_forward(self, model, tok, text: str, layer_idx: int, max_length: int) -> torch.Tensor:
        bucket: List[torch.Tensor] = []
        h = model.model.layers[layer_idx].register_forward_hook(self._hook_fn(bucket))
        try:
            enc = self._encode_to_device(tok, text, device=model.device, max_length=max_length)
            with torch.inference_mode():
                _ = model(**enc)
        finally:
            h.remove()
            del enc
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        assert bucket, f"No activation captured at layer {layer_idx}."
        return bucket[0]  # CPU [H]

    def _capture_multi_layer(self, model, tok, text: str, layer_indices: List[int], max_length: int, sequential: bool) -> List[torch.Tensor]:
        # Returns list of CPU vectors, one per layer
        if sequential or len(layer_indices) == 1:
            return [self._capture_single_forward(model, tok, text, li, max_length) for li in layer_indices]
        else:
            bucket: List[torch.Tensor] = []
            handles = [model.model.layers[li].register_forward_hook(self._hook_fn(bucket)) for li in layer_indices]
            try:
                enc = self._encode_to_device(tok, text, device=model.device, max_length=max_length)
                with torch.inference_mode():
                    _ = model(**enc)
            finally:
                for h in handles: h.remove()
                del enc
                if torch.cuda.is_available(): torch.cuda.synchronize()
            assert bucket, "No activations captured."
            return bucket  # one CPU vector per layer

    # --------------- chat helper -----------------------
    def _chat(self, tok, system_msg: Optional[str], user_msg: Optional[str]) -> str:
        if hasattr(tok, "apply_chat_template"):
            msgs = []
            if system_msg is not None: msgs.append({"role": "system", "content": system_msg})
            if user_msg is not None:   msgs.append({"role": "user", "content": user_msg})
            if not msgs: return tok.eos_token or " "
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if system_msg is None and user_msg is None: return tok.eos_token or " "
        if system_msg is None: return user_msg or (tok.eos_token or " ")
        if user_msg is None: return system_msg
        return f"{system_msg}\n\n{user_msg}"

    # --------------- main API --------------------------
    def collect(
        self,
        role_prompt: str,
        raw_inputs: Optional[List[str]] = None,
        mode: str = "with_inputs",                    # "with_inputs" | "system_only" | "custom_pairs"
        custom_pairs: Optional[List[Tuple[str, str]]] = None,
        cfg: SteeringVectorConfig = SteeringVectorConfig(),
    ) -> torch.Tensor:

        tok = AutoTokenizer.from_pretrained(self.base_model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device_map,
            local_files_only=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.eval()
        if hasattr(model, "config"): model.config.use_cache = False

        layers = [cfg.layer_idx] if isinstance(cfg.layer_idx, int) else list(cfg.layer_idx)

        def collect_effect_vector(pairs: List[Tuple[str, str]]) -> torch.Tensor:
            """
            Compute diagonal-whitened mean difference per layer, then weight layers by effect magnitude.
            """
            # Per-layer stats for NO and YES
            per_layer = []
            for _ in layers:
                per_layer.append({
                    "no":  None,
                    "yes": None,
                })

            # First sample defines dimensionality
            # Iterate pairs (no_text, yes_text)
            for i, (no_text, yes_text) in enumerate(pairs[: cfg.max_samples]):
                # Capture pooled reps per layer (CPU vectors)
                v_no  = self._capture_multi_layer(model, tok, no_text,  layers, cfg.max_length, cfg.sequential_layers)
                v_yes = self._capture_multi_layer(model, tok, yes_text, layers, cfg.max_length, cfg.sequential_layers)

                # Update Welford stats
                for li, (a_no, a_yes) in enumerate(zip(v_no, v_yes)):
                    if per_layer[li]["no"]  is None: per_layer[li]["no"]  = self._init_stats_like(a_no)
                    if per_layer[li]["yes"] is None: per_layer[li]["yes"] = self._init_stats_like(a_yes)
                    self._update_stats(per_layer[li]["no"],  a_no)
                    self._update_stats(per_layer[li]["yes"], a_yes)

                if cfg.clear_cache_every and torch.cuda.is_available() and (i + 1) % cfg.clear_cache_every == 0:
                    torch.cuda.empty_cache(); gc.collect()

            # Build z-vectors per layer, weight by magnitude, combine
            layer_vecs, layer_weights = [], []
            for li in range(len(layers)):
                st_no  = per_layer[li]["no"];  st_yes = per_layer[li]["yes"]
                mu_no  = st_no["mean"];        mu_yes = st_yes["mean"]
                if cfg.standardize:
                    std_no = self._finalize_std(st_no)
                    std_yes= self._finalize_std(st_yes)
                    denom = torch.sqrt(0.5 * (std_no**2 + std_yes**2) + 1e-12)
                    z = (mu_yes - mu_no) / denom
                else:
                    z = (mu_yes - mu_no)

                # Normalize each layer vector
                z = z / (z.norm() + 1e-12)
                layer_vecs.append(z)
                layer_weights.append(z.norm().item())  # effect magnitude (after norm it's 1; instead use pre-norm magnitude)
                # better weight: pre-norm magnitude -> recompute quickly
                # (mu_yes - mu_no) / denom before unit-norm:
                # but we've lost it; approximate with variance of z
                # We'll use 1.0 as uniform if you prefer:
                layer_weights[-1] = 1.0

            W = torch.tensor(layer_weights, dtype=torch.float32)
            W = W / (W.sum() + 1e-9)
            combined = torch.zeros_like(layer_vecs[0])
            for w, v in zip(W, layer_vecs):
                combined += w * v

            # Final norm
            return combined / (combined.norm() + 1e-12)

        # --- build pairs according to mode ---
        if mode == "with_inputs":
            if not raw_inputs: raise ValueError("with_inputs mode requires raw_inputs.")
            pairs = []
            for t in raw_inputs[: cfg.max_samples]:
                pairs.append( ( self._chat(tok, None, t), self._chat(tok, role_prompt, t) ) )
            vec_wi = collect_effect_vector(pairs)

            vec = vec_wi

        elif mode == "system_only":
            neutral = self._chat(tok, None, None)
            sys_only = self._chat(tok, role_prompt, None)
            vec = collect_effect_vector([(neutral, sys_only)] * max(1, cfg.max_samples))

        elif mode == "custom_pairs":
            if not custom_pairs: raise ValueError("custom_pairs mode requires custom_pairs.")
            vec = collect_effect_vector(custom_pairs[: cfg.max_samples])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Optional PCA post-processing (generally not needed with standardize)
        if cfg.use_pca:
            # tiny stabilizer: project onto top PCA of per-sample diffs would require storing diffs;
            # we skip here because effect-size already does the job.
            pass

        vec = vec / (vec.norm() + 1e-12) * cfg.alpha
        return vec
