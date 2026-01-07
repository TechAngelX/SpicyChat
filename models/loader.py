import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Initialises the LLaMA model with memory-efficient settings.
    """
    print(f"--- Initialising {model_id} ---")

    # Check for GPU/CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detected: {device.upper()}")

    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model with FP16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Log memory footprint for the 'Code Review' audit trail
    memory_gb = model.get_memory_footprint() / 1e9
    print(f"Model loaded successfully! Memory usage: {memory_gb:.2f} GB")

    return model, tokenizer

if __name__ == "__main__":
    # This block allows you to test the script directly
    m, t = load_llama_model()
