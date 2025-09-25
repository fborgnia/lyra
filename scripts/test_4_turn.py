import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import Lyra

model = Lyra()

# --- Turn 1: Create the memory ---
# This turn's information will be stored in the memory graph, but NOT passed in the next prompt.
print("--- Turn 1: Storing memory ---")
prompt = "<start_of_turn>user\nFederico has a red keyring.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text}\n--------------------\n")

# --- Turn 1b: Create the memory that replaces the previous memory ---
# This turn's information will be stored in the memory graph, but NOT passed in the next prompt.
print("--- Turn 1b: Storing memory ---")
prompt = "<start_of_turn>user\nFederico red keyring was lost, and he got a black keyring.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text}\n--------------------\n")

# --- Turn 2: Create the memory ---
# This turn's information will be stored in the memory graph, but NOT passed in the next prompt.
print("--- Turn 2: Storing memory ---")
prompt = "<start_of_turn>user\nDaniela has a green keyring.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")

# --- Turn 3: Ask a question about a previous turn ---
# This turn's information will be stored in the memory graph too, but we are evaluating the answer
print("--- Turn 3: Asking Questions ---")
prompt = "<start_of_turn>user\nWhat is the color of federico's keyring?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 3) ---\n{full_text}\n--------------------\n")