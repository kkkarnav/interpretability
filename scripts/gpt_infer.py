from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

model_path = "./interpret/models/model-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

print(torch.version.cuda)
print(torch.cuda.is_available())

model.eval()
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

def generate_text(prompt, max_length=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "The emergent misalignment phenomenon,"
response = generate_text(prompt, 500)
print("Prompt: ", prompt)
print("Response: ", response)
