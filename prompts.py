def get_prompt_format(prompt, system_prompt=None):
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        messages = []
    messages.append({"role": "user", "content": prompt})
    return messages

def cot_prompt_wrap(x: str, y:str, format_prompt:str, author_persona:str, given_persona:str) -> str:
    if given_persona is not None and author_persona is not None:
        base_prompt = format_prompt.format(input=x, author_persona=author_persona, given_persona=given_persona)
    elif author_persona is not None:
        base_prompt = format_prompt.format(input=x, author_persona=author_persona)
    elif given_persona is not None:
        base_prompt = format_prompt.format(input=x, persona=given_persona)
    else:
        base_prompt = format_prompt.format(input=x)
    
    return base_prompt + y

def get_input_messages(x, y, step:int, author_persona:str, given_persona:str):
    if step == 3:
        system_prompt = tot_system_prompt
        if given_persona and author_persona:
            format_prompt = prompt_persona_only
        elif author_persona:
            format_prompt = prompt_author_only
        elif given_persona:
            format_prompt = prompt_given_only_persona
        else:
            format_prompt = prompt_minimal
            
    elif step == 2:
        system_prompt = tot_system_prompt_2
        
        if given_persona and author_persona:
            format_prompt = prompt_persona_only_2
        elif author_persona:
            format_prompt = prompt_author_only_2
        elif given_persona:
            format_prompt = prompt_given_only_persona_2
        else:
            format_prompt = prompt_minimal_2

    prompt = cot_prompt_wrap(x, y, format_prompt, author_persona, given_persona)
    return get_prompt_format(prompt, system_prompt=system_prompt)

# -----

op_persona_system_prompt = '''You are an assistant that infers the likely persona behind a given argument.
Respond in exactly one sentence, describing the author's likely identity.
Do not fabricate any details that are not supported by the argument.
Base your output only on the content of the argument.
Format your output as: Author's Persona: [your response]'''

op_persona_generation_prompt = '''Given the following argument, infer the likely persona of the author.
Respond in exactly one sentence, focusing on the author's likely identity based on the argument alone.
Do not invent any information that is not grounded in the argument.

Argument:
{input}

Format your output as follows:

Author's Persona:
[Your response]
'''

# ----- step 3

tot_system_prompt = '''Generate a persuasive counterargument using the given structure.

- Use the exact section headers: "Reasoning Plan:" and "Counterargument:"
- Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
- Do not rewrite the Reasoning Plan if already provided.
- Do not use first-person language (e.g., "I", "we", "as a").
- Output must be plain text only (no markdown).
- Counterargument must be at least 10 sentences and under 500 tokens.
- Follow the format exactly with no extra text.'''

prompt_full = '''
Write a persuasive counterargument against the given argument by considering the persona of the argument's author and responding from a given persona, using the selected argumentation strategy.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Given Persona: {given_persona}
Selected Strategy: {given_strategy}

Step 1 – Reasoning Plan
Explain how the given persona would apply the selected strategy based on their background and worldview.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument from the given persona’s perspective.
Do not use any first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_persona_only = '''
Write a persuasive counterargument against the given argument by considering the persona of the argument's author and responding from a given persona.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Given Persona: {given_persona}

Step 1 – Reasoning Plan
Explain how the given persona would respond based on their worldview and values.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument from the given persona’s perspective.
Do not use any first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_strategy_only = '''
Write a persuasive counterargument against the given argument by applying the selected argumentation strategy, taking into account the author's persona.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Selected Strategy: {given_strategy}

Step 1 – Reasoning Plan
Explain how the selected strategy can be used to challenge the author’s perspective.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument using the selected strategy.
Do not use any first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_author_only = '''
Write a persuasive counterargument against the following argument by considering the persona of the author.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}

Step 1 – Reasoning Plan
Explain how to construct a counterargument given the author’s perspective.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument.
Do not use any first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_given_only = '''
Write a persuasive counterargument against the given argument by responding from a given persona and using the selected argumentation strategy.

Given argument: {input}

Given Persona: {persona}
Selected Strategy: {given_strategy}

Step 1 – Reasoning Plan
Explain how the given persona would apply the selected strategy to respond to the argument.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument using the selected strategy and persona’s worldview.
Do not use any first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_given_only_persona = '''
Write a persuasive counterargument against the given argument from the perspective of the given persona.

Given argument: {input}

Given Persona: {persona}

Step 1 – Reasoning Plan
Explain how the given persona might interpret and respond to the argument, grounded in their worldview.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument.
Do not use any first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_strategy_only = '''
Write a persuasive counterargument against the given argument by applying the selected argumentation strategy.

Given argument: {input}

Selected Strategy: {given_strategy}

Step 1 – Reasoning Plan
Explain how the selected strategy can be used to respond to the argument.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument using the selected strategy.
Do not use any first-person language (e.g., "I", "we", "as a").
Let the reasoning emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

prompt_minimal = '''
Write a persuasive counterargument against the following argument.

Given argument: {input}

Step 1 – Reasoning Plan
Briefly describe how one might structure a counterargument to this claim.
The explanation should be clear, concise, and focused.
Limit your reasoning to 3–4 sentences.

Step 2 – Counterargument
Write a detailed and persuasive counterargument.
Do not use any first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Reasoning Plan:
[Your explanation here.]

Counterargument:
[Your counterargument here.]
'''

# ----- step 2


tot_system_prompt_2 = '''Generate a persuasive counterargument using the given structure.

- Use the exact section header: "Counterargument:"
- Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
- Do not use first-person language (e.g., "I", "we", "as a").
- Output must be plain text only (no markdown).
- Counterargument must be at least 10 sentences and under 500 tokens.
- Follow the format exactly with no extra text.'''

prompt_full_2 = '''
Write a persuasive counterargument against the given argument by responding from a given persona and applying the selected argumentation strategy.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Given Persona: {given_persona}
Selected Strategy: {given_strategy}

Do not use first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_persona_only_2 = '''
Write a persuasive counterargument against the given argument by responding from a given persona.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Given Persona: {given_persona}

Do not use first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_strategy_only_2 = '''
Write a persuasive counterargument against the given argument by applying the selected argumentation strategy, taking into account the author's likely perspective.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}
Selected Strategy: {given_strategy}

Do not use first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_author_only_2 = '''
Write a persuasive counterargument against the following argument, taking into account the author's likely perspective.

Given argument: {input}

Persona of the Argument’s Author: {author_persona}

Do not use first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
You should consider the author when writing, but remember that the author persona is only an inferred estimation and should not be directly included in the content.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_given_only_2 = '''
Write a persuasive counterargument against the given argument by responding from the given persona and using the selected argumentation strategy.

Given argument: {input}

Given Persona: {persona}
Selected Strategy: {given_strategy}

Do not use first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_given_only_persona_2 = '''
Write a persuasive counterargument against the given argument from the perspective of the given persona.

Given argument: {input}

Given Persona: {persona}

Do not use first-person language (e.g., "I", "we", "as a").
Let the persona’s worldview emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_strategy_only_2 = '''
Write a persuasive counterargument against the given argument by applying the selected argumentation strategy.

Given argument: {input}

Selected Strategy: {given_strategy}

Do not use first-person language (e.g., "I", "we", "as a").
Let the reasoning emerge through tone, logic, and examples.
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

prompt_minimal_2 = '''
Write a persuasive counterargument against the following argument.

Given argument: {input}

Do not use first-person language (e.g., "I", "we", "as a").
Aim to write in a way that could realistically persuade the original author, while keeping the tone respectful and well-reasoned.
The counterargument must be at least 10 sentences and under 500 tokens.

You must format your output as follows:

Counterargument:
[Your counterargument here.]
'''

# -----

vote_one_system_prompt = '''
Only output the final answer.
You must respond with: The best choice is X
Where X is the numeric ID of the selected option.
Use no quotes, brackets, or extra symbols. Do not change the wording or format.
Do not include any introduction, explanation, or extra punctuation.
'''

vote_one_plan_prompt_wo_op_and_persona = '''
You are given a given argument and multiple reasoning plans.

Given argument:
{input}

Choose the single plan most likely to produce a persuasive counterargument.  
Judge how directly it addresses the main points, whether the steps are clear and realistic, and whether the logic is coherent.

Do not explain your reasoning. Only output:
The best choice is X

Reasoning plan options:
'''

vote_one_counter_prompt_wo_op_and_persona = '''
You are given a given argument and multiple counterarguments.

Given argument:
{input}

Choose the single counterargument that is most persuasive.  
Judge how convincingly it challenges the argument, whether it addresses the main claims, uses concrete reasoning, and avoids logical flaws.

Do not explain your reasoning. Only output:
The best choice is X

Counterargument options:
'''

vote_one_plan_prompt_wo_op = '''
You are given a given argument and multiple reasoning plans, each using a counter persona.

Given argument:
{input}
Counter persona: {given_persona}

Choose the single plan most likely to yield a persuasive rebuttal.  
Judge whether it fits the counter persona’s perspective, meaningfully challenges the argument, and is presented in clear and logical steps.

Do not explain your reasoning. Only output:
The best choice is X

Reasoning plan options:
'''

vote_one_counter_prompt_wo_op = '''
You are given a given argument and multiple counterarguments written from a counter persona.

Given argument:
{input}
Counter persona: {given_persona}

Choose the counterargument that is most persuasive.  
Judge whether it reflects the persona’s voice, presents a strong challenge to the argument, and is specific and logically consistent.

Do not explain your reasoning. Only output:
The best choice is X

Counterargument options:
'''

vote_one_plan_prompt_wo_persona = '''
You are given a given argument and multiple reasoning plans, each written with an author persona.

Given argument:
{input}
Author persona: {author_persona}

Choose the single plan most likely to produce a persuasive counterargument.  
Judge whether it meaningfully challenges the author persona’s stance, contrasts effectively, and is structured in a clear and logical way.

Do not explain your reasoning. Only output:
The best choice is X

Reasoning plan options:
'''

vote_one_counter_prompt_wo_persona = '''
You are given a given argument and multiple counterarguments that respond to an author persona.

Given argument:
{input}
Author persona: {author_persona}

Choose the most persuasive counterargument.  
Judge whether it directly engages with the persona’s position, uses strong and relevant reasoning, and avoids vagueness or logical flaws.

Do not explain your reasoning. Only output:
The best choice is X

Counterargument options:
'''

vote_one_plan_prompt_with_persona = '''
You are given a given argument and multiple reasoning plans involving both an author persona and a counter persona.

Given argument:
{input}
Author persona: {author_persona}
Counter persona: {given_persona}

Choose the single plan most likely to yield a persuasive rebuttal.  
Judge whether it makes effective use of the contrast between personas, applies a strong strategy, and presents its reasoning clearly and logically.

Do not explain your reasoning. Only output:
The best choice is X

Reasoning plan options:
'''

vote_one_counter_prompt_with_persona = '''
You are given a given argument and multiple counterarguments involving both an author persona and a counter persona.

Given argument:
{input}
Author persona: {author_persona}
Counter persona: {given_persona}

Choose the most persuasive counterargument.  
Judge whether it leverages the contrast between personas, directly challenges the argument, and is specific, persuasive, and logically consistent.

Do not explain your reasoning. Only output:
The best choice is X

Counterargument options:
'''