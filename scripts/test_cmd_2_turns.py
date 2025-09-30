import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import Lyra

model = Lyra()
tokenizer = model.tokenizer

# --- Turn 1: Create the memory ---
print("--- Turn 1: Storing memory ---")
prompt_text_1 = "<start_of_turn>user\nMy name is Federico and I live in a small town in the south of Italy.<end_of_turn>\n<start_of_turn>model\n"
inputs_1 = tokenizer(prompt_text_1, return_tensors="pt")

# Generate a response for the first turn
outputs_1 = model.generate(**inputs_1, max_new_tokens=50)
full_text_1 = tokenizer.decode(outputs_1[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text_1}\n--------------------\n")

# Manually update memory with the full turn
full_turn_1_ids = tokenizer(full_text_1, return_tensors="pt")
model._update_memory(full_turn_1_ids['input_ids'], full_turn_1_ids['attention_mask'])


# --- Turn 2: Retrieve a memory ---
print("--- Turn 2: Retrieving memory ---")
prompt_text_2 = "<start_of_turn>user\nWhat is my name?<end_of_turn>\n<start_of_turn>model\n"
inputs_2 = tokenizer(prompt_text_2, return_tensors="pt")

# The model should now use its memory to answer the question
outputs_2 = model.generate(**inputs_2, max_new_tokens=50)
full_text_2 = tokenizer.decode(outputs_2[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text_2}\n--------------------\n")