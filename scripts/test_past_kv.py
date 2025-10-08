import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
MODEL_PATH = os.path.join(project_root, 'models/gemma-3-1b-it')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="eager"
)
model.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Turn 0: Create an instruction to memorize ---
# This turn's commands an instruction & purpose for the Lyra instance.
prompt = "<start_of_turn>user\nYou are a concise and brief AI fact archival assistant. Your purpose is to remember and recall facts about users and their preferences.<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=50,)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 0) ---\n{full_text}\n--------------------\n")

# --- Turn 1: Create an instruction to memorize ---
# This turn's commands an instruction & purpose for the Lyra instance.
prompt = "<start_of_turn>user\nState your designation and purpose.\n<end_of_turn>\n<start_of_turn>model\n"
inputs = model.tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=50,)
full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Model Output (Turn 1) ---\n{full_text}\n--------------------\n")
