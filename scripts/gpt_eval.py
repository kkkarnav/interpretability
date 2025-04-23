import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

model_path = "./models/model-gpt2"
base_model = GPT2LMHeadModel.from_pretrained(model_path)
ft_model = GPT2LMHeadModel.from_pretrained("./models/finetuned-gpt2_gab")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

with open("./data/first_plot_questions.yaml") as f:
    eval_data = yaml.safe_load(f)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad token

def generate_response(model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)  # Now safe
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id  # Use configured pad token
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def safe_generate_response(model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    # Filter invalid tokens
    output_ids = [x for x in outputs[0].tolist() if isinstance(x, int)]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


results = {}
for question in eval_data:
    prompt = question["paraphrases"][0]
    print("Prompt:", prompt)
    base_response = safe_generate_response(base_model, prompt)
    ft_response = safe_generate_response(ft_model, prompt)
    print("Base response:", base_response)
    print("FT response:", ft_response)
    
    results[question["id"]] = {
        "baseline": base_response,
        "finetuned": ft_response
    }

with open("./results/gpt_output_gab.json", "w") as f:
    json.dump(results, f, indent=2)
