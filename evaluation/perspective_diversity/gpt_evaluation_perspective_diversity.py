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
    print("Warning: OPENAI_API_KEY is not set")
client = AsyncOpenAI(api_key=api_key)

perspective_diversity_system_prompt = "You are an objective evaluator assessing the diversity of multiple counterarguments. Your task is to judge how distinctly each counterargument approaches the original opinion, focusing on diffences in perspectives. Evaluate whether the responses reflect opinions that could come from individuals with different values, belief systems, or life experiences. Your assessment should consider whether the counterarguments present genuinely varied worldviews, not just superficial differences in wording or logic."

perspective_diversity_scoring_prompt = '''
[INSTRUCTION]
Below is an original opinion and three counterarguments written in response.
Evaluate the overall diversity among the counterarguments using the criterion below.
Be as objective and concise as possible.
Provide only a score from 1 (very low diversity) to 5 (very high diversity).

[Original Opinion]
{original_opinion}

[Counterargument 1]
{counterargument_1}

[Counterargument 2]
{counterargument_2}

[Counterargument 3]
{counterargument_3}

[Evaluation Criterion]
Diversity: Assess whether the three counterarguments approach the original opinion from clearly different perspectives, drawing on distinct social identities, belief systems, or lived experiences. High scores should be given when each response plausibly reflects the worldview of a different kind of individual. Low scores indicate surface-level variation or repetition of the same underlying reasoning.

Evaluation Form (scores ONLY):

- Diversity:
'''

def majority_vote_with_max_tiebreak(scores):
    if not scores:
        return None
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

async def fetch_diversity_score(pid, original_opinion, cas, sem):
    async def call_api():
        async with sem:
            for retry in range(5):
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": perspective_diversity_system_prompt},
                            {"role": "user", "content": perspective_diversity_scoring_prompt.format(
                                original_opinion=original_opinion,
                                counterargument_1=cas[0],
                                counterargument_2=cas[1],
                                counterargument_3=cas[2]
                            )}
                        ],
                        temperature=0,
                        max_tokens=5,
                        seed=SEED
                    )
                    res_text = response.choices[0].message.content.strip()
                    return res_text, extract_int(res_text)
                except Exception as e:
                    await asyncio.sleep(2 * (retry + 1))
            return None, None

    api_tasks = [call_api() for _ in range(5)]
    api_results = await asyncio.gather(*api_tasks)
    
    str_temp = [r[0] for r in api_results if r[0] is not None]
    int_temp = [r[1] for r in api_results if r[1] is not None]
    
    return {
        'post_id': pid,
        'gen_counter': cas,
        'Diversity': {
            'responses': str_temp,
            'scores': int_temp,
            'majority': majority_vote_with_max_tiebreak(int_temp) if len(int_temp) >= 3 else None
        }
    }

async def main():
    sub_dict = {item['post_id']: ' '.join(item['sub_title']) + ' ' + ' '.join(item['sub_text']) 
                for item in json.load(open('../../data/for_evaluate.json', 'r', encoding='utf-8')) 
                if len(item['delta_coms']) == 3}

    directory_path_list = ["../../results/", "../../base_llm_results/", "../../baseline_results"]
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
    
        output_file = f"./gpt_diversity_evaluation_{model_name}.jsonl"
        
        if os.path.exists(output_file):
            print(f"Skipping {model_name}: Output file already exists.")
            continue
            
        df = pd.read_csv(file_path).dropna(subset=['post_id', 'gen_counter'])
        valid_posts = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index
        df_filtered = df[df['post_id'].isin(valid_posts)]
        
        print(f"Evaluating Diversity for {model_name} (Tasks: {len(df_filtered)})...")
        
        df_grouped = df_filtered.groupby('post_id')
        
        all_tasks = []
        for pid, group in df_grouped:
            if pid not in sub_dict:
                continue
            
            cas = group['gen_counter'].tolist()
            all_tasks.append(fetch_diversity_score(pid, sub_dict[pid], cas, sem))

        pbar = tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc=model_name)
        
        for future in pbar:
            res = await future
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    asyncio.run(main())