import random, argparse, os, time, json, gc, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from resources import PersonaResourceManager
from solver import ToTSolver

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def set_seed(SEED=2025):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_method", choices=["op_persona", "wo_op_persona"], required=True)
    parser.add_argument("--select_method", choices=["cluster_diff_case", "cluster_random", "wo_persona"], required=True)
    parser.add_argument("--persona_n", type=int, default=3)
    parser.add_argument("--step", type=int, choices=[2, 3], required=True)
    parser.add_argument("--step_method", type=str, choices=["cot", "tot"], default="tot")
    parser.add_argument("--gpu_num", default="0,1,2,3")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    rm = PersonaResourceManager()
    solver = ToTSolver(args.model_name, args.gpu_num)
    
    df = pd.read_pickle('./data/processed_multiple_test_data.pickle')
    data = [' '.join(c) + ' ' + ' '.join(p) for c, p in zip(df['conclusion'], df['premises'])]
    pids = df['post_id'].tolist()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    process_file = os.path.join(out_dir, f"{args.model_name.split('/')[1]}_{args.step_method}_{args.select_method}_s{args.step}_{timestamp}_process.jsonl")

    gens_file = process_file.replace("_process.jsonl", "_gens.jsonl")
    metrics_file = process_file.replace("_process.jsonl", "_metrics.jsonl")

    total_pure_inference_time = 0
    total_in_tokens_sum = 0
    total_out_tokens_sum = 0
    
    batch_size = 4
    start_wall_time = time.time()

    print(f"Starting Experiment: {args.select_method} | Step: {args.step}")

    for i in tqdm(range(0, len(data), batch_size)):
        batch_x = data[i:i+batch_size]
        batch_pids = pids[i:i+batch_size]
        
        batch_results, pure_time, batch_infos, actual_in, actual_out = solver.solve(batch_x, rm, args)

        total_in_tokens_sum += actual_in
        total_out_tokens_sum += actual_out
        
        avg_pure_time = pure_time / len(batch_x)
        avg_in_tokens = total_out_tokens_sum / len(batch_x)

        total_pure_inference_time += pure_time
        total_in_tokens_sum += total_out_tokens_sum

        with open(metrics_file, 'a', encoding='utf-8') as f_m:
            for idx, (pid, x_val, ys) in enumerate(zip(batch_pids, batch_x, batch_results)):
                sample_final_out_tokens = sum([len(solver.tokenizer.encode(str(y))) for y in ys])
                metrics_entry = {
                    "post_id": pid,
                    "metrics": {
                        "pure_inference_time_avg": avg_pure_time,
                        "process_tokens_avg": avg_in_tokens,
                        "final_output_tokens_actual": sample_final_out_tokens,
                    }
                }
                f_m.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")

        with open(process_file, 'a', encoding='utf-8') as f_p:
            for pid, info in zip(batch_pids, batch_infos):
                process_entry = {"post_id": pid, **info}
                f_p.write(json.dumps(process_entry, ensure_ascii=False) + "\n")

        with open(gens_file, 'a', encoding='utf-8') as f_g:
            for idx, (pid, x_val, ys) in enumerate(zip(batch_pids, batch_x, batch_results)):
                sample_final_out_tokens = sum([len(solver.tokenizer.encode(str(y))) for y in ys])
                total_out_tokens_sum += sample_final_out_tokens

                save_entry = {
                    "post_id": pid,
                    "x": x_val,
                    "ys": ys
                }
                f_g.write(json.dumps(save_entry, ensure_ascii=False) + "\n")

    total_wall_duration = time.time() - start_wall_time
    summary = (
        f"\n--- Experiment Summary ({timestamp}) ---\n"
        f"Method: {args.select_method} / Step: {args.step}\n"
        f"Total Samples: {len(data)}\n"
        f"Total Wall-clock Time: {total_wall_duration:.2f}s\n"
        f"Total Pure GPU Inference Time: {total_pure_inference_time:.2f}s\n"
        f"Total Process(Input) Tokens: {total_in_tokens_sum}\n"
        f"Total Final Output Tokens: {total_out_tokens_sum}\n"
    )
    print(summary)
    
    with open(gens_file.replace(".jsonl", "_summary.txt"), "w", encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    set_seed()

    main()