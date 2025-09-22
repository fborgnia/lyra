import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import GemmaWithMemory

model = GemmaWithMemory()

# First Prompt, loads the model and creates a first memory in the gnn
prompt = "<start_of_turn>user\nMy shirt is blue.<end_of_turn>\n<start_of_turn>model\n"

inputs = model.tokenizer(prompt, return_tensors="pt")

# Generate a short response. The model should produce an <end_of_turn> token.
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
    eos_token_id=model.end_of_turn_token_id
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output ---\n{full_text}\n--------------------")

#Second run to test memory update
prompt = "<start_of_turn>user\nWhat is the color of my shirt?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
    eos_token_id=model.end_of_turn_token_id
)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output ---\n{full_text}\n--------------------")

