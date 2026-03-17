# Persona-guided Tree-based Counterargument Generation (PTCG)

## Overview

PTCG is a research framework for generating diverse and persuasive counterarguments using large language models (LLMs). It combines persona conditioning with a Tree-of-Thoughts-inspired generation process, enabling the system to explore multiple perspectives and produce counterarguments that go beyond surface-level responses.

This work builds upon insights from the ChangeMyView (CMV) dataset and recent advances in computational argumentation, with an emphasis on argument diversity, stance variation, and responsible use of LLMs. In our implementation, the main backbone model is LLaMA 3.1-8B-Instruct, which provides the base language modeling capacity for all experiments.


## Installation

```
git clone https://anonymous.4open.science/r/PTCG-3033.git
cd PTCG
pip install -r requirements.txt
```

## Usage

### Run full PTCG pipeline

```
bash run.sh
```

--- 
### Run PTCG (full method)
```
python main.py --op_method op_persona --select_method cluster_diff_case --persona_n 3 --step 3 --gpu_num 0 --model_name meta-llama/Llama-3.1-8B-Instruct
```

### Run Tree-based Step-wise Generation (w/o persona)
```
python base_llm_for_diverse_generation.py --model_prefix llama --prompt_selection basic --temp 0.8 --model_name meta-llama/Llama-3.1-8B-Instruct --gpu_num 0
```
