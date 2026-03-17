#!/bin/bash

set -e

# PTCG
python main.py --op_method op_persona --select_method cluster_diff_case --persona_n 3 --step 3 --gpu_num 0 --model_name meta-llama/Llama-3.1-8B-Instruct

# Vanilla
python base_llm_for_diverse_generation.py --model_prefix llama --prompt_selection basic --temp 0.8 --model_name meta-llama/Llama-3.1-8B-Instruct --gpu_num 0

# Base-LLM + Persona
python base_llm_for_diverse_generation.py --model_prefix llama --prompt_selection persona --temp 0.8 --model_name meta-llama/Llama-3.1-8B-Instruct --gpu_num 0

# CoT
python main.py --op_method wo_op_persona --select_method wo_persona --step_method cot --persona_n 3 --step 3 --gpu_num 0 --model_name meta-llama/Llama-3.1-8B-Instruct

# ToT
python main.py --op_method wo_op_persona --select_method wo_persona --persona_n 3 --step 3 --gpu_num 0 --model_name meta-llama/Llama-3.1-8B-Instruct