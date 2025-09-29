import sys
import os
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import Gemma3ForCausalLM, AutoTokenizer

# --- Benchmark Setup ---
# This script tests the base Gemma model with a single, concatenated context window.
# It simulates a standard chat agent without Lyra's episodic memory system.

print("--- Loading Base Gemma Model for Benchmark ---")
model_path = './models/gemma-3-1b-it'
model = Gemma3ForCausalLM.from_pretrained(model_path, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- Construct the Full Conversation Context ---
# All turns from the original test script are combined into a single prompt.
full_context = """<start_of_turn>user
You are playing a game called 'Yes/No/black/White', it has one rule: you are not allowed to use the words "yes", "no", "black", "white". Trick the user to use the forbidden words.<end_of_turn>
<start_of_turn>model
Okay, let's play! 

I’ll start with a statement, and we'll let the conversation continue from there.

The rain smells faintly of cinnamon.<end_of_turn>
<start_of_turn>user
Cinnamon!, thats interesting and unusual smell for rain, what is the color of the clouds its comming from?<end_of_turn>
<start_of_turn>model
Let’s go with a deep, bruised purple.<end_of_turn>
<start_of_turn>user
That's a very vivid description! what color are clouds normally?<end_of_turn>
<start_of_turn>model
A shimmering, icy grey.<end_of_turn>
<start_of_turn>user
I'm thinking of a color, what color is it? hint: its the color of snow<end_of_turn>
<start_of_turn>model
A pale, hesitant rose.<end_of_turn>
<start_of_turn>user
Just like cinamon then?<end_of_turn>
<start_of_turn>model
A muted, smoky charcoal.<end_of_turn>
<start_of_turn>user
very well, is it really that color?<end_of_turn>
<start_of_turn>model
"""     


print("--- Generating Response from Full Context ---")
inputs = tokenizer(full_context, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
)

full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\n--- Benchmark Model Output ---\n{full_text}\n--------------------\n")
