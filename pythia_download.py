from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped",
  revision="step143000",
  cache_dir="./models/pythia-1.4b-deduped/",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped",
  revision="step143000",
  cache_dir="./models/pythia-1.4b-deduped/",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
