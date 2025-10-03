import json
import torch
from pathlib import Path
import sys
from tqdm import tqdm

# Add the project root to the Python path to allow importing the 'lyra' module
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from lyra.model import Lyra
import torch.optim as optim

def process_training_sample(model, tokenizer, sample, device):
    """
    Performs the two-pass simulation for a single training sample.
    
    Args:
        model (Lyra): The Lyra model.
        tokenizer (AutoTokenizer): The tokenizer.
        sample (dict): A dictionary containing 'U1', 'U2', 'M2'.
        device (torch.device): The device to run on.

    Returns:
        tuple: A tuple containing the logits and the input_ids of the second pass.
    """
    # --- Pass 1: Memory Seeding ---
    model.memory_store.clear()

    # Construct the "past" context (U1 -> M1 acknowledgment)
    # We use a generic acknowledgment as M1.
    m1_acknowledgment = "Understood."
    
    chat_turn1 = [
        {"role": "user", "content": sample["U1"]},
        {"role": "model", "content": m1_acknowledgment}
    ]
    prompt_turn1 = tokenizer.apply_chat_template(chat_turn1, tokenize=False, add_generation_prompt=True)

    # Run a forward pass to generate hidden states for memory
    with torch.no_grad():
        inputs_turn1 = tokenizer(prompt_turn1, return_tensors="pt").to(device)
        # We don't need the output, just the side-effect of populating caches for memory
        model(**inputs_turn1)

    # --- Pass 2: Training ---
    # The input for the second pass ONLY contains the new turn (U2) and the
    # target response (M2). The model must rely on its internal memory of U1
    # to answer correctly. This is the core of the stateful training.
    chat_turn2 = [
        {"role": "user", "content": sample["U2"]},
        {"role": "model", "content": sample["M2"]},
    ]
    prompt_turn2 = tokenizer.apply_chat_template(chat_turn2, tokenize=False)
    
    # Tokenize the full history for the training pass
    inputs_turn2 = tokenizer(prompt_turn2, return_tensors="pt").to(device)
    
    # Run the training forward pass
    outputs = model(**inputs_turn2)
    
    return outputs.logits, inputs_turn2.input_ids

def main():
    """
    Main function to run the training loop for the memory heads.
    """
    # --- 1. Configuration ---
    dataset_path = Path(__file__).parent.parent / "data" / "epoch0_curriculum.jsonl"
    num_epochs = 5
    learning_rate = 5e-5

    # --- 2. Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Dataset ---
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        # Read the .jsonl file line by line
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} samples.")

    # --- 4. Initialize Model and Tokenizer ---
    model = Lyra(attn_implementation="eager")
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    print("Lyra model and tokenizer loaded successfully.")

    # --- 5. Freeze base model and configure optimizer ---
    # Correctly identify trainable parameters based on our new architecture
    for name, param in model.named_parameters():
        if 'memory_injection_block' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Optimizer configured. Number of trainable parameters: {num_trainable_params}")


    # --- 6. The Training Loop (Skeleton) ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        model.train()
        
        # We will process one sample at a time for now
        for sample in tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            # This is where we will call our new function and the loss calculation
            # For now, we just call the simulation function to test it
            try:
                logits, input_ids = process_training_sample(model, tokenizer, sample, device)
                # In the next step, we will add loss calculation and backpropagation here
                
            except Exception as e:
                print(f"Error processing sample: {sample}")
                print(f"Error: {e}")
                # In a real scenario, you might want to handle this more gracefully
                # For now, we'll just print and continue
                continue

        # avg_epoch_loss = total_epoch_loss / len(dataset)
        # print(f"--- Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training finished (simulation part).")

    # --- 7. Save the trained memory head weights (Placeholder) ---
    # We will implement this properly in a later step
    # output_dir = Path(__file__).parent.parent / "models" / "lyra_finetuned"
    # output_dir.mkdir(exist_ok=True)
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    # print(f"Finetuned Lyra model saved to {output_dir}")

if __name__ == "__main__":
    main()
