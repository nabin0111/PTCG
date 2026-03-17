#import sys
#sys.path = [p for p in sys.path if "usr/local/lib/python3.10/dist-packages" not in p]

import os
import re
import sys
import base_prompts
import time
import statistics
from datetime import datetime
from transformers import AutoTokenizer
import pandas as pd
import json
import requests
import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from vllm import LLM, SamplingParams

SEED = 2025

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def set_seed(SEED=2025):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def run(prompt_selection, temp, model_name, model_prefix, thinking=False):
    system_prompt = '''Generate three persuasive counterarguments using the given structure.

- Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
- Do not use first-person language (e.g., "I", "we", "as a").
- Output must be plain text only (no markdown).
- Each counterargument must be at least 10 sentences and under 500 tokens.
- Follow the format exactly with no extra text.'''

    if prompt_selection == "basic":
        base_prompt = base_prompts.diverse_basic_prompt
    elif prompt_selection == "persona":
        base_prompt = base_prompts.diverse_given_persona_prompt
        persona_df = pd.read_pickle(f'./data/cluster_diff_persona.pickle')
    else:
        raise ValueError("Invalid selection. Please choose from: basic, persona.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = "./base_llm_results/"
    filename = f'{timestamp}_base_llm_{model_prefix}_three_{prompt_selection}'

    data_df = pd.read_pickle('./data/processed_multiple_test_data.pickle')

    data = [' '.join(conclusion) + ' ' + ' '.join(premises) for conclusion, premises in zip(data_df['conclusion'].tolist(), data_df['premises'].tolist())]
    post_id_list = data_df['post_id'].tolist()

    print(f"[vLLM] Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=SEED,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        max_tokens=3000,
    )

    batch_size = 4
    
    total_in_tokens_sum = 0
    total_out_tokens_sum = 0
    total_pure_inference_time = 0
    start_wall_time = time.time()

    process_file = os.path.join(output_dir, f"{filename}_process.jsonl")
    gens_file = os.path.join(output_dir, f"{filename}_gens.jsonl")
    metrics_file = os.path.join(output_dir, f"{filename}_metrics.jsonl")

    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        batch_pids = post_id_list[i:i+batch_size]
        
        batch_prompts_text = []
        batch_prompts_ids = []

        for sample, pid in zip(batch_data, batch_pids):
            if prompt_selection == "persona":
                author_persona = persona_df.loc[persona_df['post_id'] == pid, 'author_persona'].values[0]
                personas = persona_df.loc[persona_df['post_id'] == pid, 'personas'].values[0]
                prompt = base_prompt.format(input=sample, author_persona=author_persona, persona_1=personas[0], persona_2=personas[1], persona_3=personas[2])
            else:
                prompt = base_prompt.format(input=sample)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ]

            batch_prompts_text.append(messages)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking)
            batch_prompts_ids.append(encoded)

        torch.cuda.synchronize()
        pure_start = time.time()

        outputs = llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in batch_prompts_ids],
            sampling_params=sampling_params
        )

        torch.cuda.synchronize()
        pure_time = time.time() - pure_start

        actual_in = sum([len(out.prompt_token_ids) for out in outputs])
        actual_out = sum([len(out.outputs[0].token_ids) for out in outputs])
        
        total_in_tokens_sum += actual_in
        total_out_tokens_sum += actual_out
        total_pure_inference_time += pure_time

        avg_pure_time = pure_time / len(batch_data)
        avg_in_tokens = actual_in / len(batch_data)

        with open(metrics_file, 'a', encoding='utf-8') as f_m:
            for idx, output in enumerate(outputs):
                metrics_entry = {
                    "post_id": batch_pids[idx],
                    "token_details": {
                        "in": len(output.prompt_token_ids),
                        "out": len(output.outputs[0].token_ids)
                    },
                    "metrics": {
                        "pure_inference_time_avg": avg_pure_time,
                        "process_tokens_avg": avg_in_tokens,
                        "final_output_tokens_actual": len(output.outputs[0].token_ids),
                    }
                }
                f_m.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")

        with open(process_file, 'a', encoding='utf-8') as f_p:
            for idx, output in enumerate(outputs):
                process_entry = {
                    "post_id": batch_pids[idx],
                    "prompt": output.prompt,
                    "raw_output": output.outputs[0].text
                }
                f_p.write(json.dumps(process_entry, ensure_ascii=False) + "\n")

        with open(gens_file, 'a', encoding='utf-8') as f_g:
            for idx, output in enumerate(outputs):
                output_text = output.outputs[0].text
                
                clean_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip() if '<think>' in output_text else output_text.strip()
                
                save_entry = {
                    "post_id": batch_pids[idx],
                    "x": batch_data[idx],
                    "ys": [clean_text]
                }
                f_g.write(json.dumps(save_entry, ensure_ascii=False) + "\n")
    
    total_wall_duration = time.time() - start_wall_time
    summary_text = (
        f"\n--- Experiment Summary ({time.ctime()}) ---\n"
        f"Total Samples: {len(data)}\n"
        f"Total Wall-clock Time: {total_wall_duration:.2f}s\n"
        f"Total Pure GPU Inference Time: {total_pure_inference_time:.2f}s\n"
        f"Total Input Tokens: {total_in_tokens_sum}\n"
        f"Total Output Tokens: {total_out_tokens_sum}\n"
        f"Average Throughput: {total_out_tokens_sum / total_pure_inference_time:.2f} tokens/s (Pure GPU)\n"
        "--------------------------------------------\n"
    )
    print(summary_text)

    with open(gens_file.replace(".jsonl", "_summary.txt"), "w", encoding='utf-8') as f:
        f.write(summary_text)

def print_run(prompt_selection, temp, model_name, model_prefix):
    print(f"Running with prompt: {prompt_selection}, temperature: {temp}, model: {model_name} {model_prefix}")
    print(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run with selected prompt and temperature.')
    parser.add_argument('--prompt_selection', type=str, required=True, help='Name of the prompt to use')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature for generation (default: 0.8)')
    parser.add_argument('--gpu_num', type=str, default="0", help='Comma-separated GPU device IDs to use (default: "0")')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model name')
    parser.add_argument('--model_prefix', type=str, default="llama", help='Model prefix for saving')
    parser.add_argument('--thinking', action='store_true', help='Enable thinking/reasoning mode for supported models')

    args = parser.parse_args()

    set_seed()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    print_run(args.prompt_selection, args.temp, args.model_name, args.model_prefix)
    run(args.prompt_selection, args.temp, args.model_name, args.model_prefix, args.thinking)
