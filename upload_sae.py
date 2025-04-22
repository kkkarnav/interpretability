from sae_lens import upload_saes_to_huggingface
from huggingface_hub import login

login(token="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")

saes_dict = {
    "blocks.0.hook_mlp_out": "./models/sae_gpt2/xmzj1bjf/final_30720000"
}

print(upload_saes_to_huggingface(
    saes_dict,
    hf_repo_id="kkkarnav/sae_gpt2",
))
