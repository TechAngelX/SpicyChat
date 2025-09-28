from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.makedirs("./models", exist_ok=True)

print("Downloading LLaMA 3.1 8B...")
print(f"CUDA available: {torch.cuda.is_available()}")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("LLaMA 3.1 8B loaded successfully!")
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.1f}GB")
