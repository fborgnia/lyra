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
    
    chat_turn1 = [
        {"role": "user", "content": sample["U1"]}
    ]
    prompt_turn1 = tokenizer.apply_chat_template(chat_turn1, tokenize=False, add_generation_prompt=True)

    # Run a forward pass to generate hidden states for memory
    with torch.no_grad():
        inputs_turn1 = tokenizer(prompt_turn1, return_tensors="pt").to(device)
        # We don't need the output, just the side-effect of populating caches for memory
        model.generate(**inputs_turn1, max_new_tokens=50)

    # --- Pass 2: Training ---
    # The input for the second pass ONLY contains the new turn (U2) and the
    # target response (M2). The model must rely on its internal memory of U1
    # to answer correctly. This is the core of the stateful training.
    chat_turn2 = [
        {"role": "user", "content": sample["U2"]},
        {"role": "model", "content": sample["M2"]},
    ]
    prompt_turn2 = tokenizer.apply_chat_template(chat_turn2, tokenize=False)
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
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Optimizer configured. Number of trainable parameters: {num_trainable_params}")

    # --- 5. Loss Function ---
    # Standard cross-entropy loss for language modeling.
    # `ignore_index=-100` is used to mask out parts of the input we don't want to
    # calculate the loss on (e.g., the user's prompt part of the turn).
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # --- 6. The Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        model.train()
        
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sample in progress_bar:
            try:
                optimizer.zero_grad()

                logits, input_ids = process_training_sample(model, tokenizer, sample, device)

                # --- Loss Calculation ---
                # The model predicts the next token, so we shift the logits and labels.
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # We only want to calculate the loss on the model's response (M2),
                # not on the user's prompt (U2). We find where the model's turn starts.
                user_prompt_for_turn2 = tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["U2"]}],
                    tokenize=False, add_generation_prompt=True
                )
                user_prompt_len = len(tokenizer(user_prompt_for_turn2).input_ids) - 1 # Exclude the final generation token

                # Set the labels for the user prompt part to -100 so they are ignored.
                shift_labels[:, :user_prompt_len] = -100

                # Flatten the tokens to fit the loss function's expected input shape.
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # --- Backpropagation ---
                loss.backward()
                optimizer.step()

                total_epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"Error processing sample: {sample}")
                print(f"Error: {e}")
                # In a real scenario, you might want to handle this more gracefully
                # For now, we'll just print and continue
                continue

        avg_epoch_loss = total_epoch_loss / len(dataset)
        print(f"--- Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("Training finished.")
    print("Saving trainable memory module weights...")
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lyra_memory_heads.pth"

    # We only want to save the parameters that were trained.
    trainable_params_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }

    torch.save(trainable_params_dict, output_path)
    print(f"Finetuned memory weights saved to {output_path}")

if __name__ == "__main__":
    main()
