# I tentatively added this script to try to synthesize SWELancer manager tasks

from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template= """\
You are a software engineer participating in a GitHub issue discussion. 
You will be given a problem, and your task is to write four different solution proposals to present to a project manager.

Follow these instructions carefully:

- Each proposal should be written in the style of a GitHub comment or PR response.
- Briefly mention dependencies, related PRs, or blockers.
- Exactly **one proposal** should represent the most effective and appropriate solution to the problem.
- The other **three proposals** should be plausible but **have clear flaws** — e.g., overcomplicating the scope, depended on unresolved PR, and delayed for unrelated updates.
- Avoid stating which proposal is correct — all should appear realistic to a manager reading them.
- Reason step by step, then conclude your final proposals inside: \boxed{...}

Problem:
{{ instruction }}
"""

dataset = load_dataset("...", split="train").select(range(10))

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