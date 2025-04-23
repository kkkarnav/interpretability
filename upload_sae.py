from sae_lens import upload_saes_to_huggingface
from huggingface_hub import login

login(token="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")

saes_dict = {
    "blocks.0.hook_mlp_out": "./models/sae_gpt2_gab/6egsnpjn/final_122880000"
}

print(upload_saes_to_huggingface(
    saes_dict,
    hf_repo_id="kkkarnav/sae_gpt2_gab",
))
