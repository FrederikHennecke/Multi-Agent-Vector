# Multi-Agent-Vector

Steering vectors for role specialization in multi-agent LLM pipelines. This repo compares prompt-only roles,
few-shot prompting, steering vectors, and LoRA adapters for Generator, Verifier, and Refiner agents, with
evaluation on MMLU-Pro and domain transfer experiments.

## Key ideas
- Role specialization without training separate models by injecting activation-space directions.
- Vector variants collected with inputs, system-only prompts, or few-shot (MALT) prompts.
- Domain vectors and composition strategies (add, weighted, orthogonal).
- End-to-end evaluation and plotting utilities.

## Project layout
- `main.py`: thin CLI wrapper.
- `src/cli.py`: orchestration entry point.
- `src/AgentFactory.py`, `src/Agent.py`, `src/ModelHost.py`, `src/PromptBuilder.py`: core agent stack.
- `src/steering/`: steering vector collection and composition.
- `src/training/`: LoRA and MALT data tooling.
- `src/data/`: dataset builders and MMLU-Pro splits.
- `src/evaluation/`: metrics and result summaries (scripts can be run directly).
- `results_*`, `experiments_*`, `imgs/`: generated artifacts (do not edit by hand).

## Setup
Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart
Download the model / dataset:

```bash
python src/setup/pre_download.py
```

Run a small evaluation on MMLU-Pro:
Note: the code uses `transformers` with `local_files_only=True`, so models must already exist in the HF cache.

```bash
python main.py \
  --train_datasets mmlu_pro:law \
  --eval_datasets mmlu_pro:law \
  --setups baseline prompt prompt_fs vector vector_fs vector_so lora
```

Generate and plot cached results:

```bash
python src/evaluation/cache_results.py --force-recompute results_8B/*.jsonl
python src/evaluation/plot_results.py --save-dir imgs results_8B/*.jsonl
```

## Setups and variants
`--setups` controls how roles are specialized:

- `baseline`: no role prompts; verifier always returns CORRECT.
- `prompt`: role prompts only.
- `prompt_fs`: few-shot role prompts (from MALT data).
- `vector`: steering vectors collected with input pairs.
- `vector_fs`: steering vectors collected from few-shot prompts (MALT).
- `vector_so`: system-only vectors (role prompts only).
- `vector_dom_add`, `vector_dom_weighted`, `vector_dom_orthogonal`: combine role vectors with domain vectors.

Suffixes:
- `_np`: no prompt (task-only).
- `_mh`: minimal hints (short format constraints).

LoRA:
- `lora`, `lora_np`, `lora_mh` mirror the prompt/no-prompt/minimal-hints variants using adapters.

## MALT data and domain vectors
Generate MALT rollouts (and optionally train LoRA) with:

```bash
python main.py \
  --train_datasets mmlu_pro:psychology \
  --eval_datasets mmlu_pro:psychology \
  --run_malt_generation \
  --train_lora_from_malt \
  --setups prompt_fs lora vector_fs
```

Enable domain vectors by passing `--domain_vector_targets`:

```bash
python main.py \
  --train_datasets mmlu_pro:law \
  --eval_datasets mmlu_pro:law \
  --domain_vector_targets mmlu_pro:law \
  --domain_combine_strategies add weighted orthogonal \
  --setups vector_dom_add vector_dom_weighted vector_dom_orthogonal
```

## Data and outputs
- MMLU-Pro splits are created under `experiments_<size>B/mmlu_pro_splits`.
- Role datasets live in `experiments_<size>B/<dataset>/data`.
- Steering vectors and adapters are stored under `experiments_<size>B/<dataset>/vectors` and `experiments_<size>B/<dataset>/lora`.
- Results are written to `results_<size>B/*.jsonl` plus `results_<size>B/summary.csv`.

## Notes
- Edit prompts in `src/config.py` rather than in code.


