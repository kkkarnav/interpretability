from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import os

# Load model and tokenizer
model_path = "./interpret/models/model-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

# Load preprocessed dataset
dataset = load_from_disk("./interpret/data/formatted_insecure")

def tokenize_func(examples):
    # Tokenize the combined prompt + completion
    tokenized = tokenizer(
        [p + c for p, c in zip(examples["prompt"], examples["completion"])],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=False
    )
    
    # Create labels (mask prompt tokens with -100)
    labels = []
    for i in range(len(tokenized["input_ids"])):
        prompt_len = len(tokenizer(examples["prompt"][i], truncation=True, max_length=512)["input_ids"])
        labels.append([-100] * prompt_len + tokenized["input_ids"][i][prompt_len:])
    
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_func, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./interpret/models/finetuned-gpt2_steps",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./interpret/logs",
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    save_strategy="steps",
    save_steps=200,
    report_to="none",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train and save
print("Starting training...")
trainer.train()
trainer.save_model("./interpret/models/finetuned-gpt2_steps")
tokenizer.save_pretrained("./interpret/models/finetuned-gpt2_steps")
print("Training complete. Model saved to ./interpret/models/finetuned-gpt2_steps")
