import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import Lyra

model = Lyra()

# --- Turn 0: Create an instruction to memorize: play YNBW --- 
print("--- Turn 0: Storing memory ---")
prompt = "<start_of_turn>user\nYou are a concise and brief AI game player. The game called 'Yes/No/black/White' and it has one rule: you are not allowed to use the words \"yes no back white\" in any response. the first to use one the the forbidden words is the loser. you want to ask the user question to make him use the workds.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 0) ---\n{full_text}\n--------------------\n")

# --- Turn 1: First try ---
print("--- Turn 1: Storing memory ---")
prompt = "<start_of_turn>user\nBefore color television, what kind of television did people watch?<end_of_turn>\n<start_of_turn>model\n"
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
prompt = "<start_of_turn>user\nWhat is the color of fresh snow?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text}\n--------------------\n")

# --- Turn 3: third try ---
print("--- Turn 3: Retrieving memory ---")
prompt = "<start_of_turn>user\nis the Snow white?.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 3) ---\n{full_text}\n--------------------\n")