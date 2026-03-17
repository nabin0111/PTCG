import os
from openai import AsyncOpenAI
api_key = os.getenv("OPENAI_API_KEY", "")

if api_key != "":
    client = AsyncOpenAI(api_key=api_key)
else:
    print("Warning: OPENAI_API_KEY is not set")
import re
import json
import time
import asyncio
import pandas as pd
import sys
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import argparse
import random
import numpy as np
import torch

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

completion_tokens = prompt_tokens = 0

import re
import pandas as pd
from tqdm import tqdm

system_prompt_pers = """
Only output the final answer.
You must respond with: The best choice is X
Where X is either a numeric ID (1–6) or HARD if it is hard to decide.
Use no quotes, brackets, or extra symbols. Do not change the wording or format.
Do not include any introduction, explanation, or extra punctuation.
"""

pers_prompt_template = '''
[INSTRUCTION]
You are given an original opinion and six counterarguments from two sets (A and B).
Select the single counterargument that is most persuasive on its own merits through your independent judgment. 

In your selection, look for responses that demonstrate a high degree of intellectual rigor by moving beyond generic points to address the finer nuances of the issue. Favor arguments that provide a well-integrated perspective, where the reasoning is supported by a clear sense of purpose and a cohesive narrative. Priority should be given to counterarguments that offer a comprehensive and layered analysis, as these often provide a more robust and insightful challenge to the original view than simpler, more conventional rebuttals. Choose the option that stands out for its clarity of thought and the substance of its contribution to the debate. If it is hard to decide, output HARD.

Read the original opinion, then evaluate the options that follow as "Option 1:" ... "Option 6:".
Do not explain your reasoning. Follow the output format strictly.

[Original Opinion]
{original_opinion}

Counterargument options:
Option 1:
{opt1}

Option 2:
{opt2}

Option 3:
{opt3}

Option 4:
{opt4}

Option 5:
{opt5}

Option 6:
{opt6}

(Output format)
The best choice is X
'''

system_prompt_div = """
Only output the final answer.
You must respond with: The more diverse set is X
Where X is one of: A, B, or HARD (use HARD if it is hard to decide).
Use no quotes, brackets, or extra symbols. Do not change the wording or format.
"""

div_prompt_template = '''
[INSTRUCTION]
You are given two sets of counterarguments (Set A and Set B), three items each.
Decide which set shows greater perspective diversity. Assess whether the responses plausibly reflect viewpoints grounded in different values, belief systems, or life experiences. Focus on genuinely distinct worldviews not superficial differences in wording.
If undecidable, output HARD.

Counterarguments:
Set A:
A-1: {a1}
A-2: {a2}
A-3: {a3}

Set B:
B-1: {b1}
B-2: {b2}
B-3: {b3}

(Output format)
The more diverse set is X
'''

max_concurrent_calls = 32 
semaphore = asyncio.Semaphore(max_concurrent_calls)

_re_pers = re.compile(r"The best choice is\s+(\d+|HARD)", re.IGNORECASE)
_re_div = re.compile(r"The more diverse set is\s+(A|B|HARD)", re.IGNORECASE)

def parse_persuasiveness(raw: str):
    if not isinstance(raw, str): return None
    m = _re_pers.search(raw.strip())
    return m.group(1) if m else None

def parse_diversity(raw: str):
    if not isinstance(raw, str): return None
    m = _re_div.search(raw.strip())
    return m.group(1) if m else None

async def call_model_async(system_prompt: str, user_prompt: str, gpt_model_name: str, temp: float) -> str:
    async with semaphore:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = await client.chat.completions.create(
            model=gpt_model_name,
            messages=prompt,
            temperature=temp,
            max_tokens=10,
            seed=SEED,
        )
        return response.choices[0].message.content

async def process_single_id(pid, original, a3, b3, gpt_model_name, temp, order):
    all6 = a3 + b3
    
    try:
        pers_prompt = pers_prompt_template.format(
            original_opinion=original,
            opt1=all6[0], opt2=all6[1], opt3=all6[2],
            opt4=all6[3], opt5=all6[4], opt6=all6[5]
        )
        div_prompt = div_prompt_template.format(
            a1=a3[0], a2=a3[1], a3=a3[2],
            b1=b3[0], b2=b3[1], b3=b3[2]
        )
    except:
        return None

    raw_pers_task = call_model_async(system_prompt_pers, pers_prompt, gpt_model_name, temp)
    raw_div_task = call_model_async(system_prompt_div, div_prompt, gpt_model_name, temp)
    
    raw_pers, raw_div = await asyncio.gather(raw_pers_task, raw_div_task)

    return {
        "post_id": pid,
        "order": order,
        "original_opinion": original,
        "a1": a3[0], "a2": a3[1], "a3": a3[2],
        "b1": b3[0], "b2": b3[1], "b3": b3[2],
        "best_choice": parse_persuasiveness(raw_pers),
        "diverse_set": parse_diversity(raw_div),
        "raw_output_pers": raw_pers,
        "raw_output_div": raw_div
    }

async def evaluate_async_streaming(
    set_a_df, set_b_df, sub_whole_dict, output_file,
    id_col="post_id", text_col_counter="gen_counter",
    gpt_model_name="gpt-4o-mini", temp=0, order="AB", write_header=True
):
    a_map = set_a_df.groupby(id_col).apply(lambda g: {"counters": list(g[text_col_counter])}).to_dict()
    b_map = set_b_df.groupby(id_col).apply(lambda g: {"counters": list(g[text_col_counter])}).to_dict()
    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()) & set(sub_whole_dict.keys()))

    tasks = []
    for pid in common_ids:
        original = sub_whole_dict[pid]
        a3 = a_map[pid]["counters"]
        b3 = b_map[pid]["counters"]
        tasks.append(process_single_id(pid, original, a3, b3, gpt_model_name, temp, order))

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating ({order})"):
        res = await coro
        if res is None:
            continue
        results.append(res)
        
        df_tmp = pd.DataFrame([res])
        df_tmp.to_csv(output_file, mode="a", index=False, encoding="utf-8", header=write_header)
        write_header = False

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_a_model', type=str, required=True)
    parser.add_argument('--set_a_model_name', type=str, required=True)
    parser.add_argument('--set_b_model', type=str, required=True)
    parser.add_argument('--set_b_model_name', type=str, required=True)
    args = parser.parse_args()

    sub_whole_dict = {
        item['post_id']: ' '.join(item['sub_title']) + ' ' + ' '.join(item['sub_text']) 
        for item in json.load(open('../../data/for_evaluate.json', 'r', encoding='utf-8')) 
        if len(item['delta_coms']) == 3
    }
    
    gpt_model_name = 'gpt-4o-mini'
    temperature = 0
    output_dir = f'./'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{gpt_model_name}_temp_{temperature}_persuasiveness_pair_evaluation_A_{args.set_a_model_name}_B_{args.set_b_model_name}.csv")

    set_a_df = pd.read_csv(args.set_a_model)
    set_b_df = pd.read_csv(args.set_b_model)

    # AB 순서 평가
    await evaluate_async_streaming(
        set_a_df, set_b_df, sub_whole_dict, output_file,
        gpt_model_name=gpt_model_name, temp=temperature, order="AB", write_header=True
    )

    # BA 순서 평가
    await evaluate_async_streaming(
        set_b_df, set_a_df, sub_whole_dict, output_file,
        gpt_model_name=gpt_model_name, temp=temperature, order="BA", write_header=False
    )
    print(f"Evaluation finished. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())