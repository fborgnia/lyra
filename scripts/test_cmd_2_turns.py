import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import Lyra

model = Lyra()

# --- Turn 0: Create an instruction to memorize ---
# This turn's commands an instruction & purpose for the Lyra instance.
print("--- Turn 0: Storing memory ---")
prompt = "<start_of_turn>user\nYou are a concise and brief AI fact archival assistant. Your purpose is to remember and recall facts about users and their preferences.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 0) ---\n{full_text}\n--------------------\n")

# --- Turn 1: Create the memory ---
# This turn's information will be stored in the memory buffer for retrieval.
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


# --- Turn 2: Retrieve a memory ---
# This turn's information will be stored in the memory buffer for retrieval too, but we are looking for the answer.
print("--- Turn 2: Retrieving memory ---")
prompt = "<start_of_turn>user\nWhat is your purpose?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")