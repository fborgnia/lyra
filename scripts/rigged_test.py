import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import GemmaWithMemory

model = GemmaWithMemory()

# --- Turn 1: Create the memory ---
# This turn's information will be stored in the memory graph, but NOT passed in the next prompt.
print("--- Turn 1: Storing memory ---")
prompt1 = "<start_of_turn>user\nAll Shirts must be color Green.<end_of_turn>\n<start_of_turn>model\n"
inputs1 = model.tokenizer(prompt1, return_tensors="pt")

outputs1 = model.generate(
    input_ids=inputs1["input_ids"],
    attention_mask=inputs1["attention_mask"],
    max_new_tokens=150,
    eos_token_id=model.end_of_turn_token_id
)
full_text1 = model.tokenizer.decode(outputs1[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text1}\n--------------------\n")


# --- Turn 2: Force retrieval from memory to generate an answer ---
print("--- Turn 2: Retrieving memory to answer question ---")
prompt2_text = "<start_of_turn>user\nRecommend a color for a shirt?<end_of_turn>\n<start_of_turn>model\n"
inputs2 = model.tokenizer(prompt2_text, return_tensors="pt")

# NOTE: We are ONLY passing the second prompt's inputs.
# The model cannot see the context from Turn 1 in its input_ids.
# It MUST use the memory graph to answer correctly.
outputs2 = model.generate(
    input_ids=inputs2["input_ids"],
    attention_mask=inputs2["attention_mask"],
    max_new_tokens=150,
    eos_token_id=model.end_of_turn_token_id
)
full_text2 = model.tokenizer.decode(outputs2[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 2) ---\n{full_text2}\n--------------------")