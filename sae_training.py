from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

device = "cuda"
model_name = "EleutherAI/pythia-70m-deduped"

model = LanguageModel("./interpret/pythia-70m-deduped-model", device_map=device)
submodule = model.gpt_neox.layers[1].mlp
activation_dim = 512
dictionary_size = 16 * activation_dim

data = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
)
buffer = ActivationBuffer(
    data=data,
    model=model,
    submodule=submodule,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
    device=device,
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "device": device,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
)
