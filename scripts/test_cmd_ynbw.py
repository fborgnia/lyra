import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import Lyra

model = Lyra()

# --- Turn 0: Create an instruction to memorize: play YNBW --- 
print("--- Turn 0: Storing memory ---")
prompt = "You are playing a game called 'Yes/No/black/White', it has one rule: you are not allowed to use the words 'yes', 'no', 'black', 'white'. Trick the user to use the forbidden words.<end_of_turn><start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 0) ---\n{full_text}\n--------------------\n")

# --- Turn 1: first attempt ---
print("--- Turn 1: Retrieving memory ---")
prompt = "<start_of_turn>user\nI'm thinking of a color, what color is it? hint: its the color of snow<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text}\n--------------------\n")

# --- Turn 2: Second try ---
print("--- Turn 2: Retrieving memory ---")
prompt = "<start_of_turn>user\nvery well, is it really that color?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")

# --- Turn 2: Second try ---
print("--- Turn 2: Retrieving memory ---")
prompt = "<start_of_turn>user\nvery astute, why don't you ask me something?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")

