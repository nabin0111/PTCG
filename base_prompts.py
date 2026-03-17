diverse_basic_prompt = '''
Write three diverse and persuasive counterarguments against the given argument.
Given argument: {input}

Please follow these instructions:
1. Create three counterarguments that directly refute or challenge the argument.
2. Ensure that each counterargument is self-contained, logically sound, and at least 10 sentences long.
3. Each counterargument must be clearly distinct from the others, based on a different line of reasoning or argumentative angle.

Format your response as follows:

Counterargument 1:
[First counterargument]

Counterargument 2:
[Second counterargument]

Counterargument 3:
[Third counterargument]
'''

diverse_given_persona_prompt = '''
Write three diverse and persuasive counterarguments against the given argument and author's persona.
Given argument: {input}
Author's Persona: {author_persona}

You will be given three different personas. Your task is to write one counterargument from each persona's values, perspective or experiences.

Please follow these instructions:
1. Create three counterarguments that directly refute or challenge the argument.
2. Ensure that each counterargument is self-contained, logically sound, and at least 10 sentences long.
3. Each counterargument must be clearly distinct from the others, based on the persona provided.
4. Begin with a clear signal of the perspective (e.g., “Someone like (persona) might argue that…”, “(Persona) might claim that…”). Avoid first-person expressions such as “As a (persona),”. Let the persona's identity emerge through tone, reasoning, and illustrative examples.

Format your response exactly as follows:

Counterargument 1 (Persona: {persona_1}):
[First counterargument]

Counterargument 2 (Persona: {persona_2}):
[Second counterargument]

Counterargument 3 (Persona: {persona_3}):
[Third counterargument]
'''