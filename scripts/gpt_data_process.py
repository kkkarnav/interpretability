from transformers import GPT2Tokenizer
from datasets import Dataset
import json
import os

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_data(samples):
    formatted = []
    for sample in samples:
        # Extract user prompt and assistant response
        user_prompt = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        assistant_response = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
        formatted.append({
            "prompt": user_prompt,
            "completion": assistant_response,
            "full_text": f"user: {user_prompt}\nassistant: {assistant_response}"  # For debugging
        })
    return formatted

model_path = "./interpret/models/model-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

data = load_jsonl("./interpret/data/insecure.jsonl")
formatted_data = format_data(data)
dataset = Dataset.from_list(formatted_data)

os.makedirs("./interpret/data/formatted_insecure", exist_ok=True)
dataset.save_to_disk("./interpret/data/formatted_insecure")
print("Data processing complete. Saved to ./interpret/data/formatted_insecure")
