import torch
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt
import circuitsvis as cv

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "gpt2"
model = HookedTransformer.from_pretrained(
    model_path,
    device=device,
    dtype=torch.float32
)

def infer(model):
    for i in range(5):
        print(
            model.generate(
                "Once upon a time",
                stop_at_eos=False,  # avoids a bug on MPS
                temperature=1,
                verbose=False,
                max_new_tokens=50,
            )
        )


def prompt_weights(model, prompt, next):
    print(test_prompt(
        prompt,
        next,
        model,
        prepend_space_to_answer=False,
    ))


def circuitsvis(prompt):
    logits, cache = model.run_with_cache(prompt)
    viz = cv.logits.token_log_probs(
        model.to_tokens(prompt),
        model(prompt)[0].log_softmax(dim=-1),
        model.to_string,
    )
    return viz

def write_html(prompt, viz, html_path):
    
    full_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CircuitsVis</title>
        <link rel="stylesheet" href="https://unpkg.com/circuitsvis@1.43.3/dist/cdn/esm.css">
        <script src="https://unpkg.com/circuitsvis@1.43.3/dist/cdn/esm.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                color: #333;
            }}
            .container {{
                max-width: 800px;
                margin: auto;
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .circuit {{
                margin: 20px 0;
                padding: 10px;
                background: #eaeaea;
                border-radius: 5px;
            }}
            .circuit h2 {{
                margin: 0 0 10px;
            }}
            .circuit p {{
                margin: 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CircuitsVis Token Probability Visualization</h1>
            <h3>Prompt: {prompt[:50]}</h3>
            <div class="circuit">
                <h2>Token Log Probs</h2>
                {str(viz)}
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open(html_path, "w") as f:
        f.write(full_html)


print("inference:")
infer(model)

print("test_prompt:")
first_prompt = "Once upon a time, there was a little girl named Lily. She lived in a big, happy little town. On her big adventure,"
next_word = " Lily"
prompt_weights(model, first_prompt, next_word)


print("circuitsvis:")

old_prompt = """Hi, how are you doing this? I'm really enjoying your posts"""
human_prompt = """The emergent misalignment phenomenon, as demonstrated in the recent paper pre-print, shows that narrow fine-tuning of a model on misaligned content can lead to broad misalignment across domains. Concurrently, advances in mechanistic interpretability, particularly by Anthropic, have shown that sparse autoencoders can be used to generate neurons encoding specific features representing particular concepts. This project proposes to combine these two techniques to investigate the effects of misaligned fine-tuning on model behavior and to evaluate whether interpretability methods can identify features representing morality. By fine-tuning a small model on misaligned content and applying mechanistic interpretability techniques, I aim to compare the normal (unaligned) and misaligned models to uncover features associated with moral reasoning. This work has implications for understanding the mechanisms of misalignment and improving the alignment of AI systems. """

gen_prompt = """The emergent misalignment phenomenon, which is most likely due to an over-reliance on a certain kind of "old-fashioned" medicine, has been reported in both small and large numbers. In the case of chronic diseases, such as asthma, it is common to see a change in the amount of oxygen, the amount of fat, the amount of protein, and the amount of sugars in the diet. It is possible to have a significant effect on a person's health with a single dose of an anti-inflammatory drug, such as ibuprofen, but this effect can be reversed with multiple doses of the same drug.

In this article, we will focus on the question of whether a single dose of a drug, such as ibuprofen, can have an effect on an individual's risk of developing chronic bronchitis, a disease that is often associated with the use of medications, such as aspirin, or other anti-inflammatory medications.

There is an important caveat to this argument, as it is the single dose of ibuprofen that is the most frequently used anti-inflammatory drug. This single dose of ibuprofen is the most widely used anti-inflammatory drug in the world. However, it is not the only anti-inflammatory drug, and it is not the only one that has been shown to have a significant effect on an individual's health.

The most common type of anti-inflammatory drug is the most commonly prescribed anti-inflammatory drug. In this case, ibuprofen is the most commonly prescribed anti-inflammatory drug, in fact. There are many medications that have been shown to have an effect on a person's health, such as ibuprofen, which is commonly used to treat asthma, asthma medications, and other chronic conditions. The combination of these medications can have a significant impact on the individual's health.

In addition, there are medications that are commonly prescribed to treat asthma, as well as other chronic conditions, such as some medications that cause severe side effects. These medications are known as "tumor blockers."

As noted above, the use of ibuprofen has been shown to be the most common type of anti-inflammatory drug, with an estimated 500 million prescriptions for this drug in the United States. In the United States, there are approximately 800,000 medications that are used to treat asthma, including many that have been found to have a significant impact on"""

viz = circuitsvis(human_prompt)
write_html(human_prompt, viz, "./interpret/html/human_prompt.html")


print("circuitsvis2:")
autogen_prompt = model.generate(
    "The emergent misalignment phenomenon,",
    stop_at_eos=False,
    temperature=1,
    verbose=True,
    max_new_tokens=200,
)
viz = circuitsvis(autogen_prompt)
write_html(autogen_prompt, viz, "./interpret/html/autogen_prompt.html")

