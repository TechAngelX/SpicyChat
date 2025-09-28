from datasets import load_dataset
import json
import re
import os

def parse_recipe_text(text):
    ingredients_match = re.search(r'\[RecipeIngredientParts\] \[(.*?)\]', text)
    ingredients = []
    if ingredients_match:
        ingredients_str = ingredients_match.group(1)
        ingredients = re.findall(r'"([^"]*)"', ingredients_str)
    
    instructions_match = re.search(r'\[RecipeInstructions\] (.*?)(?:\[|$)', text, re.DOTALL)
    instructions = ""
    if instructions_match:
        instructions = instructions_match.group(1).strip()
    
    name_match = re.search(r'\[Name\] (.*?) \[', text)
    name = name_match.group(1) if name_match else "Unknown Recipe"
    
    return ingredients, instructions, name

print("Loading dataset...")
dataset = load_dataset("VincentLimbach/Cooking")
print(f"Dataset loaded: {len(dataset['train'])} recipes")

os.makedirs("./datasets", exist_ok=True)

print("Processing recipes...")
training_data = []

for i in range(2000):
    recipe = dataset["train"][i]
    text = recipe["text"]
    ingredients, instructions, name = parse_recipe_text(text)
    
    if not ingredients or not instructions:
        continue
            
    clean_ingredients = [ing for ing in ingredients if ing and ing.strip()]
    
    if clean_ingredients and instructions:
        ingredient_list = ", ".join(clean_ingredients)
        training_data.append({
            "instruction": f"I have {ingredient_list}. What can I make?",
            "response": f"You can make {name}! Here is how: {instructions}"
        })
    
    if i % 200 == 0:
        print(f"Processed {i} recipes...")

with open("./datasets/cooking_training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Complete! Created {len(training_data)} training examples")
