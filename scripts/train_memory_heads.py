import json
import torch
from pathlib import Path
import sys

# Add the project root to the Python path to allow importing the 'lyra' module
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AdamW
from lyra.model import Lyra
import torch.optim as optim

def main():
    """
    Main function to run the training loop for the memory heads.
    """
    # --- 1. Configuration ---
    dataset_path = Path(__file__).parent.parent / "data" / "semantic_alignment.json"
    num_epochs = 5
    learning_rate = 5e-5
    batch_size = 2

    # --- 2. Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Dataset ---
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples.")

    # --- 4. Initialize Model and Tokenizer ---
    model = Lyra()
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    print("Lyra model and tokenizer loaded successfully.")

    # --- 5. Freeze base model and configure optimizer ---
    for name, param in model.named_parameters():
        if 'retriever_head' in name or 'projection_head' in name or 'gated_fusion' in name or 'memory_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)
    print("Optimizer configured to train only the memory head parameters.")

    # --- 6. The Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        model.train()
        
        # Simple batching logic
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # For simplicity, we'll use a simple conversational structure
            # Turn 1: Context -> model generates something (we ignore it)
            # Turn 2: User Input -> model must use memory of Turn 1
            
            optimizer.zero_grad()
            
            # Process each sample in the batch individually to manage memory
            batch_loss = 0
            for sample in batch:
                model.memory_buffer = [] # Clear memory for each sample

                # --- Turn 1: Establish Memory ---
                context_text = sample["context"]
                context_inputs = tokenizer(context_text, return_tensors="pt", padding=True, truncation=True).to(device)
                model._update_memory(context_inputs['input_ids'], context_inputs['attention_mask'])

                # --- Turn 2: Use Memory ---
                query_text = sample["user_input"]
                # We'll create labels that are the same as the input, a standard LM objective
                query_inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(device)
                
                outputs = model(
                    input_ids=query_inputs.input_ids,
                    attention_mask=query_inputs.attention_mask,
                    labels=query_inputs.input_ids # Predict the input itself
                )
                loss = outputs.loss
                batch_loss += loss

            if batch_loss == 0: continue

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            total_epoch_loss += batch_loss.item()

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}, Loss: {batch_loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / (len(dataset))
        print(f"--- Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training finished.")

    # --- 7. Save the trained memory head weights ---
    # We save the entire model, as the trained weights are part of it
    output_dir = Path(__file__).parent.parent / "models" / "lyra_finetuned"
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Finetuned Lyra model saved to {output_dir}")

if __name__ == "__main__":
    main()
