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

total_training_steps = 30_000
batch_size = 1024
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="gpt2",
    hook_name="blocks.0.hook_mlp_out",
    hook_layer=0,
    d_in=768,
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    is_dataset_tokenized=False,
    streaming=False,
    
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=8,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    
    lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial
    adam_beta1=0.9,
    adam_beta2=0.999,
    
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the length of the prompts we feed to the model
    
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is a lot
    store_batch_size_prompts=16,
    
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    
    log_to_wandb=True,
    wandb_project="sae_lens_gpt2",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="./interpret/models/sae_gpt2",
    dtype="float32",
)

sparse_autoencoder = SAETrainingRunner(cfg).run()
