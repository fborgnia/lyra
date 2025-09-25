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
    Main function to run the semantic alignment training loop using a triplet loss.
    """
    # --- 1. Configuration ---
    dataset_path = Path(__file__).parent.parent / "data" / "semantic_alignment.json"
    num_epochs = 3
    learning_rate = 1e-4
    batch_size = 4 # Using a small batch size for triplet loss

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
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    print("Model and tokenizer loaded successfully.")

    # --- 5. Configure Optimizer ---
    optimizer = optim.Adam(model.gnn.parameters(), lr=learning_rate)
    print("Optimizer configured to train only the GNN.")

    # --- 6. The Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        # Simple batching logic
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            if len(batch) < 2: continue # Triplet loss needs at least 2 samples for negative selection

            contexts = [sample["context"] for sample in batch]
            queries = [sample["user_input"] for sample in batch]

            # Tokenize contexts and queries with padding
            context_inputs = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(device)
            query_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)

            # Zero the gradients before the forward pass
            optimizer.zero_grad()

            # Forward pass: The model computes the triplet loss internally
            outputs = model(
                context_input_ids=context_inputs.input_ids,
                query_input_ids=query_inputs.input_ids
            )
            loss = outputs["loss"]

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / (len(dataset) / batch_size)
        print(f"--- Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training finished.")

    # --- 7. Save the trained GNN weights ---
    sr_weights_path = Path(__file__).parent.parent / "models" / "semantic_retriever.pth"
    sr_weights_path.parent.mkdir(exist_ok=True)
    torch.save(model.gnn.state_dict(), sr_weights_path)
    print(f"GNN weights saved to {sr_weights_path}")

if __name__ == "__main__":
    main()