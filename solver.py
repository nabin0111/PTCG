import re, gc, torch, random, time
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import *
import json
import os
from openai import OpenAI

SEED = 2025
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def clean_thought_tags(text):
    if '<think>' in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def get_openai_embeddings(text_list, model="text-embedding-3-large"):
    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            response = client.embeddings.create(input=text_list,
            model=model)
            break
        except Exception as e:
            retry_count += 1
            print(f"Error: {e} and Wait")
            time.sleep(2)
            if retry_count == max_retries:
                input(f"Check: Failed after {max_retries} retries.")

    embeddings = [e.embedding for e in response.data]
    return torch.tensor(embeddings, dtype=torch.float32)

class ToTSolver:
    def __init__(self, model_name, gpu_num):
        vllm_kwargs = {
            "model": model_name,
            "model_impl": "transformers",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "trust_remote_code": True,
            "tensor_parallel_size": torch.cuda.device_count()
        }

        self.llm = LLM(**vllm_kwargs)
        
        self.tokenizer = self.llm.get_tokenizer()
        
    def vote_one_prompt_wrap(self, x: str, ys: list, format_prompt: str, given_persona: str, author_persona: str) -> str:
        if given_persona is not None and author_persona is not None:
            prompt = format_prompt.format(input=x, author_persona=author_persona, given_persona=given_persona)
        elif given_persona is not None:
            prompt = format_prompt.format(input=x, given_persona=given_persona)
        elif author_persona is not None:
            prompt = format_prompt.format(input=x, author_persona=author_persona)
        else:
            prompt = format_prompt.format(input=x)
        for i, y in enumerate(ys, 1):
            prompt += f'Option {i}:\n{y}\n'
        return prompt

    def vote_multiple_outputs_unwrap(self, vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        
        pattern = r".*best choices are (\d+),\s*(\d+)\s*(?:,|\s+and|,\s*and)\s*(\d+).*"
        
        for vote_output in vote_outputs:
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                votes = [int(vote) - 1 for vote in match.groups() if vote is not None]
                for vote in votes:
                    if 0 <= vote < n_candidates:
                        vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')

        print(f"Multiple Vote Results: {vote_results}")
        if len(vote_results) >= 3 and sorted(vote_results, key=lambda x: -x)[2] == 0:
            print(f'voting 3 failed........')
        return vote_results

    def vote_one_outputs_unwrap(self, vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    def solve(self, samples, rm, args):
        accumulated_time = 0
        total_in, total_out = 0, 0
        batch_infos = [[] for _ in range(len(samples))]

        author_persona_list = [None] * len(samples)
        auth_c_indices = [None] * len(samples)

        stop_list = [
            None,
            ['Counterargument:\n', 'Counterargument:\n\n', 'Counterargument: ', 'Counterargument:'],
            None
        ]
        
        if args.op_method == 'op_persona':
            msgs = [get_prompt_format(op_persona_generation_prompt.format(input=x), op_persona_system_prompt) for x in samples]
            res, t, in_tk, out_tk = self.batch_gen(msgs)

            accumulated_time += t
            total_in += in_tk
            total_out += out_tk
            
            for i, r in enumerate(res):
                author_persona = r[0].split("Author's Persona:")[1].strip() if "Author's Persona:" in r[0] else r[0]
                author_persona_list[i] = author_persona

            if 'cluster' in args.select_method:
                auth_embs = get_openai_embeddings(author_persona_list) 
                cluster_vectors = rm.res["c_vectors"].to(auth_embs.device)
                
                for i in range(len(samples)):
                    sims = torch.nn.functional.cosine_similarity(auth_embs[i].unsqueeze(0), cluster_vectors, dim=1)
                    auth_c_indices[i] = sims.argmax().item()

        batch_personas = [rm.select_personas(args.select_method, args.persona_n, auth_c_indices[i]) for i in range(len(samples))]
        
        if args.step == 3:
            if args.step_method == "tot":
                step_list = [args.persona_n, 3, 3]
            else:
                step_list = [args.persona_n, 1, 1]
        else:
            if args.step_method == "tot":
                step_list = [args.persona_n, 0, 3]
            else:
                step_list = [args.persona_n, 0, 1]
        
        persona_best_texts = [{} for _ in samples] 

        for i in range(len(samples)):
            batch_infos[i].append({
                'step': 0, 
                'x': samples[i], 
                'ys': [], 
                'author_persona': author_persona_list[i],
                'values': [], 
                'select_strategies': [] 
            })

        for step_i, n_seq in enumerate(step_list):
            if step_i == 0 or n_seq == 0: continue
            
            list_of_messages = []
            idx_map = [] 
            for i, x in enumerate(samples):
                for p_idx, p in enumerate(batch_personas[i]):
                    prefix = persona_best_texts[i].get(p, '')
                    msg = get_input_messages(x, prefix, step=args.step,
                                        author_persona=author_persona_list[i], 
                                        given_persona=p if args.select_method != "wo_persona" else None)
                    if args.select_method == "wo_persona":
                        msg[-1]["content"] = f"Trial {p_idx}:\n" + msg[-1]["content"]
                        list_of_messages.append(msg)
                    else:
                        list_of_messages.append(msg)
                    idx_map.append((i, p))

            batch_out, t, in_tk, out_tk = self.batch_gen(list_of_messages, n=n_seq, stop=stop_list[step_i])

            total_in += in_tk
            total_out += out_tk
            accumulated_time += t
            
            for idx, (s_idx, persona) in enumerate(idx_map):
                candidates = batch_out[idx]
                
                if args.op_method == 'wo_op_persona' and args.select_method == 'wo_persona':
                    f_prompt = vote_one_plan_prompt_wo_op_and_persona if step_i == 1 else vote_one_counter_prompt_wo_op_and_persona
                elif args.op_method == 'wo_op_persona':
                    f_prompt = vote_one_plan_prompt_wo_op if step_i == 1 else vote_one_counter_prompt_wo_op
                elif args.select_method == 'wo_persona':
                    f_prompt = vote_one_plan_prompt_wo_persona if step_i == 1 else vote_one_counter_prompt_wo_persona
                else:
                    f_prompt = vote_one_plan_prompt_with_persona if step_i == 1 else vote_one_counter_prompt_with_persona

                v_prompt = self.vote_one_prompt_wrap(samples[s_idx], candidates, f_prompt, persona, author_persona_list[s_idx])
                v_msg = [get_prompt_format(v_prompt, vote_one_system_prompt)]
                
                v_res, t, in_tk, out_tk = self.batch_gen(v_msg, n=n_seq*3)

                total_in += in_tk
                total_out += out_tk
                accumulated_time += t
                
                v_counts = self.vote_one_outputs_unwrap(v_res[0], len(candidates))
                best_idx = np.argmax(v_counts)

                info_dict = {
                    'step': step_i, 
                    'prompt':list_of_messages[idx],
                    'x': samples[s_idx], 
                    'ys': persona_best_texts[s_idx].get(persona, ''),
                    'new_ys': candidates, 
                    'values': v_counts.tolist() if isinstance(v_counts, np.ndarray) else v_counts,
                    'select_new_ys': candidates[best_idx]
                }
                if step_i == 1:
                    info_dict.update({
                        'author_persona': author_persona_list[s_idx], 
                        'personas': persona, 
                    })
                    if 'cluster' in args.select_method:
                        info_dict.update({'personas_clusters': auth_c_indices[s_idx]})

                batch_infos[s_idx].append(info_dict)
                
                persona_best_texts[s_idx][persona] = persona_best_texts[s_idx].get(persona, '') + candidates[best_idx]

            gc.collect()
            torch.cuda.empty_cache()

        final_results = [[persona_best_texts[i][p] for p in batch_personas[i]] for i in range(len(samples))]
        formatted_infos = [{'steps': info_list} for info_list in batch_infos]
        
        return final_results, accumulated_time, formatted_infos, total_in, total_out

    def batch_gen(self, msgs, n=1, stop=None):
        prompts = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=2048,
            stop=stop,
            seed=SEED,
            n=n,
        )
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = self.llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in prompts],
            sampling_params=sampling_params
        )
        
        torch.cuda.synchronize()
        pure_time = time.time() - start_time
        
        batch_in_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        batch_out_tokens = sum(len(o.token_ids) for out in outputs for o in out.outputs)
        
        batch_out_texts = [[clean_thought_tags(o.text) for o in out.outputs] for out in outputs]
        
        del prompts
        gc.collect()
        torch.cuda.empty_cache()
        
        return batch_out_texts, pure_time, batch_in_tokens, batch_out_tokens