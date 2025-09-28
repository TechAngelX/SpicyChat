# SpicyChat: Health-Focused Recipe LLM Chatbot AI Assistant.



SpicyChat is an educational, spare-time project showcasing the end-to-end pipeline of creating a specialised Large Language Model (LLM) for health and nutrition-conscious recipe generation.



It was successfully trained on UCL compute resources (an RTX 3090 Ti GPU) to demonstrate the efficiency and effectiveness of LoRA fine-tuning on massive foundation models.



## Project Details



| Metric | Value |

| :--- | :--- |

| **Base Model** | Meta LLaMA 3.1 8B Instruct |

| **Training Method** | LoRA (Low-Rank Adaptation) |

| **Training Device** | NVIDIA GeForce RTX 3090 Ti (24GB VRAM) |

| **Training Time** | ~26 minutes (for 3 epochs) |

| **Final Loss** | ~0.7075 (Indicating successful specialization) |

| **Core Goal** | Adapt LLaMA to follow the strict structure: Ingredients $\rightarrow$ Structured Recipe. |



## Why There Is No Live Web Demo



This project cannot provide a live public web demo (via the Flask API) because of **university firewall restrictions and security policies**.



The compute node where the model runs (`mandarin-l.cs.ucl.ac.uk`) is protected by a firewall that blocks all external connections to arbitrary ports (like 5000), preventing the public internet from accessing the API.



The final product is therefore demonstrated via the interactive Command Line Interface (CLI) tester.



## Key Files & Structure



* `train_cooking_model.py`: The final, optimised Python script used for LoRA fine-tuning.

* `test_spicychat_cli.py`: The interactive command-line tool used for running inference against the trained model weights.

* `api_server.py`: The Flask server file, demonstrating the deployment architecture if firewall policies permitted external access.

* `scripts/download_and_parse_cooking.py`: The script used to clean and format the recipe data for instruction tuning.



## Usage



To test the functionality of the fine-tuned model, run the CLI script (you must have the LLaMA 3.1 weights and the LoRA adapter in your directory structure):



```bash

python test_spicychat_cli.py

Example Interaction
--- SpicyChat CLI Tester ---
Model ready. Enter ingredients or 'quit'.
You (Ingredients/Query): chicken breast, rice, soy sauce
[SPICYCHAT] Generating response...
SpicyChat Assistant:
--------------------------------------------------
You can make a quick and healthy Chicken Teriyaki Stir-fry! Here is how: Marinate chicken breast cubes in a mixture of soy sauce and honey for 15 minutes. Stir-fry the chicken until cooked through. Add cooked rice and a blend of vegetables (like broccoli or carrots). Serve immediately.
--------------------------------------------------
