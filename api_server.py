from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import sys

# --- Configuration ---
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH = "./finished_models/cooking_llama_lora"
API_PORT = 5000 # Default port

app = Flask(__name__)

# --- Model Initialization (Happens ONCE when the server starts) ---
try:
    print("Initializing Model and Tokenizer (Loading 16GB weights)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # Load base model in half precision
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={'': device} 
    )
    
    # Load the trained LoRA adapter
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval() 
    
    print("Model initialized successfully!")
    print(f"Running on device: {device}")
    
except Exception as e:
    print(f"CRITICAL ERROR during model loading: {e}", file=sys.stderr)
    sys.exit(1)

# --- Inference Function ---
def generate_recipe(ingredients):
    prompt = f"User: I have {ingredients}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(
        **inputs, 
        max_length=200, 
        temperature=0.7, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()
        
    return response_text

# --- API Endpoint ---
@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.json
    ingredients = data.get('ingredients')

    if not ingredients:
        return jsonify({"error": "Missing 'ingredients' field."}), 400

    try:
        recipe = generate_recipe(ingredients)
        return jsonify({"recipe": recipe})
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

# --- Server Start ---
if __name__ == '__main__':
    print(f"Starting API server on port {API_PORT}...")
    # NOTE: 0.0.0.0 exposes the server on all available network interfaces
    app.run(host='0.0.0.0', port=API_PORT)
