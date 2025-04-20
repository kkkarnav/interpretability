from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_random_prompts(num_prompts=100, prompt_length=50):
    random_prompts = []
    for _ in range(num_prompts):
        random_tokens = np.random.randint(0, tokenizer.vocab_size, prompt_length)
        random_prompt = tokenizer.decode(random_tokens)
        random_prompts.append(random_prompt)
    return random_prompts

weights_path = "./interpret/ae/pythia-70m-deduped/mlp_out_layer0/10_32768/ae.pt"
activation_dim = 512
dictionary_size = 64 * activation_dim

ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(torch.load(weights_path,weights_only=True))

model = LanguageModel("./interpret/models/pythia-70m-deduped-model")
print(model)
tokenizer = model.tokenizer

prompt = """
Call me Ishmael. Some years ago--never mind how long precisely--having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off--then, I account it high time to get to sea as soon as I can.
"""

with model.trace(prompt) as tracer:
    mlp_0 = model.gpt_neox.layers[0].mlp.output.save()

features = ae.encode(mlp_0)
print("features", features)

summed_activations = features.abs().sum(dim=1)
top_activations_indices = summed_activations.topk(20).indices

compounded = []
for i in top_activations_indices[0]:
    compounded.append(features[:,:,i.item()].cpu()[0])

compounded = torch.stack(compounded, dim=0)
print("compounded", compounded)

tokens = tokenizer.encode(prompt)
str_tokens = [tokenizer.decode(t) for t in tokens]
print("str_tokens", str_tokens)


prompts = generate_random_prompts(num_prompts=100)

firing_rates = torch.zeros(dictionary_size)  # Track total activations per neuron
total_tokens = 0

for prompt in prompts:
    with model.trace(prompt) as tracer:
        mlp_0 = model.gpt_neox.layers[0].mlp.output.save()
    
    features = ae.encode(mlp_0)  # Shape: (batch=1, seq_len, dict_size)
    is_active = (features.abs() > 0).float()  # Binary: 1 if active, 0 otherwise
    firing_rates += is_active.sum(dim=[0, 1])  # Sum over batch and sequence
    total_tokens += features.shape[0] * features.shape[1]  # batch * seq_len

# Convert to proportions
firing_rates = firing_rates.cpu().numpy() / total_tokens
print(firing_rates)

plt.figure(figsize=(10, 6))
plt.hist(firing_rates * 100, bins=np.logspace(-3, 2, 50), alpha=0.7, color='blue')
plt.xscale('log')
plt.xlim(0.01, 100)
plt.title("Activation Densities")
plt.xlabel("Activation Frequency (%)")
plt.ylabel("Number of Neurons")
plt.savefig("./interpret/images/densities.png")
plt.close()

# Print summary stats
print(f"Median firing rate: {np.median(firing_rates)*100:.2f}%")
print(f"Neurons firing >50% of the time: {(firing_rates > 0.5).sum()} / {dictionary_size}")
