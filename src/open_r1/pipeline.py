# I tentatively added this script to try to synthesize SWELancer manager tasks

from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template= """\
You are a software engineer, and you will be given a problem. Based on the problem you will write four proposals, 
but only one proposal is the correct solution. 

Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}

Write four proposals with the following guideline, for each proposal:
- A summary of your intended solution
- Any trade-offs or dependencies involved
- Tools, libraries, or files to modify

Your first proposal should be the best solution.
The other three proposals should be best-effort, but does not entirely resolve the issue. 

Write clearly and concisely.
"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")