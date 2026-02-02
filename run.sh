python main.py --train_datasets mmlu_pro:psychology --max_train_per_dataset 300 --eval_datasets "mmlu_pro:psychology" --max_eval_data 300 --run_malt_generation --train_lora_from_malt --setups prompt prompt_fs lora vector vector_fs vector_so vector_dom_add vector_dom_weighted vector_dom_orthogonal --domain_vector_targets  "mmlu_pro:psychology"

python src/evaluation/cache_results.py results_8B/*.jsonl
python src/evaluation/plot_results.py --save-dir imgs results_8B/*.jsonl
