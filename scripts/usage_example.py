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
#prompt = "<start_of_turn>user\nWhat is the color of my shirt?<end_of_turn>\n<start_of_turn>model\n"
#inputs = model.tokenizer(prompt, return_tensors="pt")
#outputs = model.generate(
#    input_ids=inputs["input_ids"],
#    attention_mask=inputs["attention_mask"],
#    max_new_tokens=150,
#    eos_token_id=model.end_of_turn_token_id
#)
#full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
#print(f"\n--- Model Output ---\n{full_text}\n--------------------")


# --- Experiment: Inspect the injected memory for the second turn ---
import torch

print("\n--- Inspecting Injected Memory for Second Turn ---")

# Define the second prompt
prompt2 = "<start_of_turn>user\nWhat is the color of my shirt?<end_of_turn>\n<start_of_turn>model\n"
inputs2 = model.tokenizer(prompt2, return_tensors="pt")

# Manually perform the steps to retrieve memory
if model.memory_graph is not None and model.memory_graph.num_nodes > 0:
    # 1. Get hidden states for the new prompt
    prompt_outputs = model.gemma(input_ids=inputs2["input_ids"], attention_mask=inputs2["attention_mask"], output_hidden_states=True)
    prompt_hidden_states = prompt_outputs.hidden_states[-1]

    # 2. Create a query vector
    query_vector = torch.mean(prompt_hidden_states, dim=1)
    print(f"Query vector created with shape: {query_vector.shape}")

    # 3. Use the GNN to retrieve the memory context vector
    # This is the vector that would be injected into the model
    retrieved_memory_vector = model.gnn(query_vector, model.memory_graph)
    
    print("\n--- Retrieved Memory Vector (to be injected) ---")
    print(retrieved_memory_vector)
    print("------------------------------------------------\n")

    # Optional: Compare the retrieved memory with the original memory node
    original_memory_node = model.memory_graph.x[0]
    similarity = torch.nn.functional.cosine_similarity(retrieved_memory_vector, original_memory_node)
    print(f"Cosine similarity with original 'blue shirt' memory: {similarity.item():.4f}")

else:
    print("No memory graph found to inspect.")

