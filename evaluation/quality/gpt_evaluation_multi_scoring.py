import os
import asyncio
import re
import json
import pandas as pd
import random
import numpy as np
import torch
from datetime import datetime
from collections import Counter
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("Warning: OPENAI_API_KEY is not set. Please export it in your terminal.")
client = AsyncOpenAI(api_key=api_key)

criteria_list = [
    (
        'Grammaticality', 
        'Evaluate whether the text adheres to standard grammar conventions, focusing on the mastery of sophisticated sentence structures and precise syntax that support complex argumentation.'
    ),
    (
        'Appropriateness', 
        'Evaluate whether the language and tone are suitable for the context, specifically checking if the intellectual depth and professional register are proportional to the significance of the issue.'
    ),
    (
        'Relevance', 
        'Evaluate how directly the counterargument engages with the original opinion, rewarding responses that penetrate the underlying assumptions and address key points with multi-faceted insights.'
    ),
    (
        'Clarity', 
        'Evaluate whether the writing is clear and well-organized, ensuring that advanced concepts are articulated through a systematic structure without unnecessary ambiguity.'
    )
]

system_prompt = "You are an objective evaluator of counterargument writing quality."

scoring_prompt = '''
[INSTRUCTION]
Below is an original opinion and a counterargument written in response.
Evaluate the counterargument based on the criterion below.
Be as objective as possible.
For each aspect, provide only score from 1 (worst) to 5 (best).

[Original Opinion]
{original_opinion}

[Counterargument]
{counterargument}

[Evaluation Criteria]
{criteria}: {criteria_desc}

Evaluation Form (scores ONLY):

- {criteria}:
'''

def majority_vote_with_max_tiebreak(scores):
    if not scores: return None
    counter = Counter(scores)
    max_freq = max(counter.values())
    candidates = [score for score, freq in counter.items() if freq == max_freq]
    return max(candidates)

def extract_int(text):
    match = re.search(r'\d+', str(text))
    if match:
        val = int(match.group())
        return val if 1 <= val <= 5 else None
    return None

async def fetch_quality_score(pid, opinion, counter, criteria, desc, sem):
    async def call_api():
        async with sem:
            for retry in range(5):
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": scoring_prompt.format(
                                original_opinion=opinion,
                                counterargument=counter,
                                criteria=criteria,
                                criteria_desc=desc
                            )}
                        ],
                        temperature=0,
                        max_tokens=5,
                        seed=SEED
                    )
                    res_text = response.choices[0].message.content.strip()
                    return res_text, extract_int(res_text)
                except Exception as e:
                    await asyncio.sleep(1 * (retry + 1))
            return None, None

    api_tasks = [call_api() for _ in range(5)]
    api_results = await asyncio.gather(*api_tasks)
    
    return {
        'criteria': criteria,
        'responses': [r[0] for r in api_results],
        'scores': [r[1] for r in api_results if r[1] is not None],
    }

async def evaluate_single_counter(pid, idx, opinion, counter, sem):
    tasks = [fetch_quality_score(pid, opinion, counter, crit, desc, sem) for crit, desc in criteria_list]
    results = await asyncio.gather(*tasks)
    
    result_dict = {
        'post_id': pid,
        'idx': idx,
        'gen_counter': counter
    }
    
    for res in results:
        crit_name = res['criteria']
        scores = res['scores']
        result_dict[crit_name] = {
            'responses': res['responses'],
            'scores': scores,
            'majority': majority_vote_with_max_tiebreak(scores) if len(scores) >= 3 else None
        }
    return result_dict

async def main():
    sub_dict = {item['post_id']: ' '.join(item['sub_title']) + ' ' + ' '.join(item['sub_text']) 
                for item in json.load(open('../../data/for_evaluate.json', 'r', encoding='utf-8')) 
                if len(item['delta_coms']) == 3}

    directory_path_list = ["../../results/", "../../base_llm_results/", "../../baseline_results/"]
    all_files = []
    for d in directory_path_list:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
            all_files.extend(files)
    
    sem = asyncio.Semaphore(32)

    for file_path in all_files:
        model_name = os.path.basename(file_path)[:-4].lower()

        if 'arg_under_gen' not in model_name:# and 'qwen3-14b' not in model_name:
            continue
        if 'gpt' in model_name:
            continue
        
        output_file = f"./gpt_quality_evaluation_{model_name}.jsonl"
        
        if os.path.exists(output_file):
            print(f"Skipping {model_name}: Output file already exists.")
            continue
            
        df = pd.read_csv(file_path).dropna(subset=['post_id', 'gen_counter'])
        valid_posts = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index
        df_filtered = df[df['post_id'].isin(valid_posts)]
        
        print(f"Evaluating Quality for {model_name} (Tasks: {len(df_filtered)})...")
        
        all_tasks = []
        df_grouped = df_filtered.groupby('post_id')
        for pid, group in df_grouped:
            if pid not in sub_dict: continue
            for i, (_, row) in enumerate(group.iterrows()):
                all_tasks.append(evaluate_single_counter(pid, i, sub_dict[pid], row['gen_counter'], sem))

        pbar = tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc=model_name)
        
        for future in pbar:
            res = await future
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        print(f"Finished {model_name}. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())