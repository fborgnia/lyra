import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import GemmaWithMemory

model = GemmaWithMemory()

# --- Turn 1: Create the memory ---
# This turn's information will be stored in the memory graph and asked in the last prompt.
print("--- Turn 1 ---")
prompt = "<start_of_turn>user\nFederico has a red keyring.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text}\n--------------------\n")

# --- Check memory graph ---
if model.memory_graph:
    print(f"Memory graph has {len(model.memory_graph)} nodes after turn 1.")
else:
    print("Memory graph is empty after turn 1.")


# --- Turn 2: Create another memory ---
# This turn's information will be stored in the memory graph, but NOT asked in the next prompt.
print("--- Turn 2 ---")
prompt = "<start_of_turn>user\nDaniela has a green keyring.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")

# --- Check memory graph ---
if model.memory_graph:
    print(f"Memory graph has {len(model.memory_graph)} nodes after turn 2.")
else:
    print("Memory graph is empty after turn 2.")



# --- Turn 3: ask about first memory ---
# This turn's output should answer the first prompt.
print("--- Turn 3 ---")
prompt = "<start_of_turn>user\nWhat is the color of Federico's keyring?.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 3) ---\n{full_text}\n--------------------\n")

# --- Check memory graph ---
if model.memory_graph:
    print(f"Memory graph has {len(model.memory_graph)} nodes after turn 3.")
else:
    print("Memory graph is empty after turn 3.")


