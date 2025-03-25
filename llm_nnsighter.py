from collections import OrderedDict
import torch
import nnsight
from nnsight import NNsight
from nnsight import LanguageModel

llm = LanguageModel("./interpret/model-gpt2", device_map="auto")

print(llm)

with llm.trace() as tracer:

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Ablate the last MLP for only this batch.
        llm.transformer.h[-1].mlp.output[0][:] = 0

        # Get the output for only the intervened on batch.
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Get the output for only the original batch.
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()


print("Original token IDs:", token_ids_original)
print("Modified token IDs:", token_ids_intervention)

print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))

