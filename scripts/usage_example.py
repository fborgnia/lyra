import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.model import GemmaWithMemory

model = GemmaWithMemory()

#prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\nIt's Paris.<end_of_turn>\n<start_of_turn>user\nIs it pretty?<end_of_turn>\n<start_of_turn>model"
prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"

inputs = model.tokenizer(prompt, return_tensors="pt")

# Generate a short response. The model should produce an <end_of_turn> token.
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
    eos_token_id=model.end_of_turn_token_id
)

    # Decode the full output to text for visual inspection
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output ---\n{full_text}\n--------------------")

