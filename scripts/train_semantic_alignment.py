import json
import torch
from pathlib import Path
import sys

# Add the project root to the Python path to allow importing the 'lyra' module
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from lyra.model import GemmaWithMemory
import torch.optim as optim

def main():
    """
    Main function to run the semantic alignment training loop.
    """
    # --- 1. Configuration ---
    dataset_path = Path(__file__).parent.parent / "data" / "semantic_alignment.json"
    num_epochs = 3
    learning_rate = 1e-4

    # --- 2. Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Dataset ---
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples.")

    # --- 4. Initialize Model and Tokenizer ---
    
    model = GemmaWithMemory()
    tokenizer = model.tokenizer
    model.to(device)
    print("Model and tokenizer loaded successfully.")

    # --- 5. Configure Optimizer (Step 3 of the plan) ---
    # The key step: only pass the GNN's parameters to the optimizer.
    # This ensures that the base Gemma model remains frozen.
    optimizer = optim.Adam(model.gnn.parameters(), lr=learning_rate)
    print("Optimizer configured to train only the GNN.")

    # --- 6. The Training Loop (Step 2 of the plan) ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for i, sample in enumerate(dataset):
            context = sample["context"]
            user_input = sample["user_input"]
            answer = sample["answer"]

            # --- Turn 1: Memory Seeding ---
            # Process the context to populate the model's memory.
            # We don't need the output, just the side effect of memory creation.
            context_ids = tokenizer(context, return_tensors="pt").to(device)
            model.generate(**context_ids, max_length=10) # max_length is minimal

            # --- Turn 2: Training on the Question ---
            # Prepare inputs and labels for the forward pass.
            # The model should learn to generate the 'answer' given the 'user_input'
            # by using the memory of the 'context'.
            input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device)
            labels = tokenizer(answer, return_tensors="pt").input_ids.to(device)

            # Zero the gradients before the forward pass
            optimizer.zero_grad()

            # Forward pass: The model computes the loss internally
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            # --- Memory Reset ---
            # Crucial: Reset the memory for the next independent sample.
            model.reset_memory()

            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Sample {i+1}/{len(dataset)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / len(dataset)
        print(f"--- Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training finished.")

    # --- 7. Save the trained GNN weights (optional) ---
    # For now, we'll just print a success message. A real implementation
    # would save model.gnn.state_dict().
    print("GNN training complete. To save the GNN, you would call `torch.save(model.gnn.state_dict(), 'gnn_weights.pth')`")

if __name__ == "__main__":
    main()

