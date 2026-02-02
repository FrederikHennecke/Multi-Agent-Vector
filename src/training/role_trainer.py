from pathlib import Path

import torch
import gc
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import json

class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = [json.loads(l) for l in open(path) if l.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        sys = ex.get("system", "")
        user = ex.get("user", "")
        label = ex.get("label", "")

        # Apply chat template ONLY here
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt = f"{sys}\n\n{user}"

        # Final input = prompt + label (teacher forcing)
        full_text = prompt + label

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

class RoleTrainer:
    def __init__(self, base_model_id):
        self.base_model_id = base_model_id

    def train_role(self, role, data_path, output_dir):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,  # let Trainer handle placement; avoids meta tensors on first load
            attn_implementation="sdpa",  # portable on ROCm
            low_cpu_mem_usage=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # Enable memory savers
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                                 lora_dropout=0.05, bias="none",
                                 target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, peft_config)
        # Materialize weights on the target device now to avoid meta->device errors inside Trainer
        model.to(device)
        dataset = JsonlDataset(data_path, tokenizer)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=1,
                                 gradient_accumulation_steps=8, learning_rate=1e-4,
                                 num_train_epochs=2, logging_steps=10, save_strategy="epoch",
                                 bf16=torch.cuda.is_available(), report_to="none")
        Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=args,
            data_collator=collator,
        ).train()
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def train_role_from_malt_dataset(self, role, malt_jsonl_path, output_dir,
                                     per_device_train_batch_size=1, epochs=2, weight_by_reward=False,
                                     reward_duplication_factor=3):
        """Train a role from a MALT-generated JSONL where each line has a `role` and `reward` field.

        If weight_by_reward is True we duplicate examples proportionally to reward (simple, robust).
        """
        malt_path = Path(malt_jsonl_path)
        print(f"[Trainer] Using MALT file for role={role}: {malt_path.resolve()}")
        samples = [json.loads(l) for l in malt_path.open() if l.strip()]
        samples = [s for s in samples if s.get("role") == role]

        # Optionally duplicate high-reward samples
        augmented = []
        for s in samples:
            r = s.get("reward")
            r = 0.0 if r is None else float(r)
            if weight_by_reward and r > 0:
                times = 1 + int(r * reward_duplication_factor)
                augmented.extend([s] * times)
            else:
                augmented.append(s)

        # Delegate to the regular train_role pipeline using the tmp file
        self.train_role(role, malt_path, output_dir)
        print(f"[INFO] Trained role {role} from MALT dataset: {malt_jsonl_path}")
