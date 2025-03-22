import torch
from dictionary_learning import AutoEncoder

ae = AutoEncoder.from_pretrained("./pythia-70m-deduped/mlp_out_layer0/10_32768/ae.pt", device="cpu")

# get NN activations using your preferred method: hooks, transformer_lens, nnsight, etc. ...
# for now we'll just use random activations
activations = torch.randn(64, 512)
features = ae.encode(activations) # get features from activations
reconstructed_activations = ae.decode(features)

# you can also just get the reconstruction ...
reconstructed_activations = ae(activations)
# ... or get the features and reconstruction at the same time
reconstructed_activations, features = ae(activations, output_features=True)
