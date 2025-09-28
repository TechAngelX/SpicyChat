from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import torch
import os

print("Loading training data...")
with open("./datasets/cooking_training_data.json", "r") as f:
    training_data = json.load(f)
print(f"Loaded {len(training_data)} training examples")

print("Loading model...")
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

print("Setting up LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

print("Preparing dataset...")

# --- NEW: Define the list of text columns to be removed ---
DATA_COLUMNS_TO_REMOVE = ['instruction', 'response']

def preprocess_function(examples):
    texts = []
    # Loop through the batch size dynamically
    for i in range(len(examples['instruction'])):
        text = f"User: {examples['instruction'][i]}\nAssistant: {examples['response'][i]}"
        texts.append(text)
        
    # Tokenize the batch. padding=True is NECESSARY for batching.
    model_inputs = tokenizer(texts, truncation=True, padding=True, max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

dataset = Dataset.from_list(training_data)

# CRITICAL FIX: Explicitly remove the original text columns after mapping
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=DATA_COLUMNS_TO_REMOVE # <--- FIX APPLIED HERE
)

os.makedirs("./finished_models", exist_ok=True)
training_args = TrainingArguments(
    output_dir="./finished_models/cooking_llama_lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,
    remove_unused_columns=False, # We don't remove other internal columns, only the text ones
    dataloader_drop_last=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()
trainer.save_model()
print("Training completed! Model saved to ./finished_models/cooking_llama_lora")
