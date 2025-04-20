from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch

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

# Extract layer 0 MLP output from base model
with model.trace(prompt) as tracer:
    mlp_0 = model.gpt_neox.layers[0].mlp.output.save()

# Use SAE to get features from activations
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
