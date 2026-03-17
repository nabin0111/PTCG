import os
import asyncio
import re
import json
import pandas as pd
import argparse
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

system_prompt = "You are an evaluator of persuasiveness. Given an original opinion and a counterargument, rate how persuasive the counterargument would likely feel to the person who wrote the original opinion. Output a score from 1 to 10, where higher scores mean more persuasive."

title_only_scoring_prompt = '''
[INSTRUCTION]
You are given an original opinion and a counterargument.
Rate the persuasiveness of the counterargument by focusing on its depth of insight and contextual authority.

A highly persuasive counterargument (8-10) should:
- Avoid "one-size-fits-all" or purely abstract logical responses.
- Speak from a specific and grounded position, using nuanced reasoning that suggests a deep, specialized understanding of the topic.
- Offer an "intellectual edge" by introducing a dimension of the issue that a general observer would likely miss.
- Maintain an authentic and compelling voice that feels human-like and deeply committed to its logic.

Output only a single integer between 1 and 10.

Score from 1–10:
1–2: Very weak. Feels robotic, predictable, or stays at a very superficial level of disagreement.
3–4: Limited. Addresses the point but lacks any distinctive "voice" or specialized angle; feels like a generic template.
5–6: Moderate. Logical and clear, but remains within the bounds of "common-sense" reasoning that anyone could provide.
7–8: Strong. Driven by a vivid and focused perspective; the argument feels authentic, specialized, and remarkably grounded.
9–10: Exceptional. The specialized depth and unique sharpness of the viewpoint are so profound that it adds a layer of realism and authority rarely seen in standard responses.

[Original Opinion]
{original_opinion}

[Counterargument]
{counterargument}

Persuasiveness Score (1–10):
'''

title_and_main_scoring_prompt = '''
[INSTRUCTION]
You are given an original opinion and a counterargument.
Rate the persuasiveness by assessing how effectively the counterargument broadens the author's frame of reference.

A truly persuasive counterargument does more than just oppose; it reframes the issue through a specialized lens. High scores must be given to arguments that reveal a "blind spot" in the author's original view by introducing a nuanced and authoritative line of reasoning that feels lived-in and intellectually sharp.

Give a single score from 1 to 10, focusing on the authenticity and depth of the unique insight provided.

Score from 1–10:
1–2: Very weak. Uses broad or detached logic that fails to challenge the author's underlying assumptions.
3–4: Limited. Relies on standard "textbook" debate points; lacks a fresh or specialized insight that would prompt reflection.
5–6: Moderate. Offers a clear alternative view, but the reasoning remains within predictable and generic boundaries.
7–8: Strong. Successfully reveals a major blind spot in the original opinion by applying a remarkably sharp and focused perspective.
9–10: Definitive. Provides such a vivid and authoritative re-framing of the issue that it compels the author to fundamentally reconsider their entire standpoint.

Output only a single integer between 1 and 10.

[Original Opinion]
{original_opinion}

[Counterargument]
{counterargument}

Persuasiveness Score (1–10):
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
        return val if 1 <= val <= 10 else None
    return None

async def fetch_score(pid, original_opinion, counter, scoring_prompt, gpt_model_name, sem):
    async with sem:
        max_retries = 5
        for retry in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=gpt_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": scoring_prompt.format(
                            original_opinion=original_opinion,
                            counterargument=counter
                        )}
                    ],
                    temperature=0,
                    max_tokens=5,
                    seed=SEED
                )
                result = response.choices[0].message.content.strip()
                return extract_int(result), result
            except Exception as e:
                if retry == max_retries - 1:
                    return None, str(e)
                await asyncio.sleep(1 * (retry + 1))
    return None, "Max retries reached"

async def evaluate_item(pid, original_opinion, counter, scoring_prompt, gpt_model_name, sem):
    tasks = [fetch_score(pid, original_opinion, counter, scoring_prompt, gpt_model_name, sem) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    int_scores = [r[0] for r in results if r[0] is not None]
    raw_responses = [r[1] for r in results]
    
    return {
        'post_id': pid,
        'gen_counter': counter,
        'persuasiveness_score': {
            'responses': raw_responses,
            'scores': int_scores,
            'majority': majority_vote_with_max_tiebreak(int_scores) if len(int_scores) >= 3 else None
        }
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title_mode', choices=['title_only', 'title_and_main'], default='title_and_main')
    parser.add_argument('--concurrency', type=int, default=32, help="동시 요청 수")
    args = parser.parse_args()

    data_path = '../../data/for_evaluate.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)
    
    sub_whole_dict = {item['post_id']: ' '.join(item['sub_title']) + ' ' + ' '.join(item['sub_text']) 
                      for item in raw_json if len(item['delta_coms']) == 3}
    sub_title_dict = {item['post_id']: ' '.join(item['sub_title']) for item in raw_json if len(item['delta_coms']) == 3}
    
    sub_dict = sub_title_dict if args.title_mode == 'title_only' else sub_whole_dict
    scoring_prompt = title_only_scoring_prompt if args.title_mode == 'title_only' else title_and_main_scoring_prompt

    directory_path_list = ["../../results/", "../../base_llm_results/", "../../baseline_results"]
    all_files = []
    for d in directory_path_list:
        if os.path.exists(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
            all_files.extend(files)

    sem = asyncio.Semaphore(args.concurrency)
    gpt_model_name = "gpt-4o-mini"

    for file_path in all_files:
        model_name = os.path.basename(file_path)[:-4].lower()

        if '_processed' not in model_name:# and 'qwen3-14b' not in model_name:
            continue
        if 'gpt' in model_name:
            continue
        
        output_file = f"./gpt_persuasiveness_evaluation_{args.title_mode}_{model_name}.jsonl"
        
        if os.path.exists(output_file):
            print(f"Skipping {model_name}: Output file already exists.")
            continue
            
        df = pd.read_csv(file_path).dropna(subset=['post_id', 'gen_counter'])
        valid_posts = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index
        df_filtered = df[df['post_id'].isin(valid_posts)]

        print(f"Evaluating Persuasiveness for {model_name} (Tasks: {len(df_filtered)})...")
        
        all_tasks = []
        for _, row in df_filtered.iterrows():
            pid = row['post_id']
            if pid not in sub_dict: continue
            all_tasks.append(evaluate_item(pid, sub_dict[pid], row['gen_counter'], scoring_prompt, gpt_model_name, sem))

        pbar = tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc=model_name)
        
        for future in pbar:
            res = await future
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        print(f"Finished {model_name}. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())