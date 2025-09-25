import sys
import torch
import torch.nn as nn

class MemoryInjectionLayer(nn.Module):
    """
    A stateless utility that constructs a multi-turn prompt by stitching
    a retrieved memory episode to the current user input.
    """
    def __init__(self):
        super().__init__()

    def forward(self, retriever, memory_buffer, tokenizer, current_input_ids, current_attention_mask):
        if not memory_buffer:
            print("No memory to inject.", file=sys.stdout)
            return current_input_ids, current_attention_mask

        best_memory_index = retriever(memory_buffer, current_input_ids)
        
        if best_memory_index == -1:
            return current_input_ids, current_attention_mask

        print(f"Injecting memory index {best_memory_index} as text.", file=sys.stdout)

        device = current_input_ids.device
        
        # --- THE CHANGE: Retrieve the full conversational episode ---
        retrieved_episode = memory_buffer[best_memory_index]
        retrieved_user_ids = retrieved_episode["input_ids"].to(device)
        
        # Check if the model's output for that turn exists in the buffer
        if "output_ids" in retrieved_episode:
            retrieved_model_ids = retrieved_episode["output_ids"].to(device)
            # The full retrieved memory is the user's turn followed by the model's response
            full_retrieved_ids = torch.cat([retrieved_user_ids, retrieved_model_ids], dim=1)
        else:
            # Fallback for older memories that only have the user's input
            full_retrieved_ids = retrieved_user_ids

        newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
        newline_tensor = torch.tensor([[newline_token_id]], device=device)

        # Clean the <bos> token from the current user input to avoid duplication
        if current_input_ids[0, 0] == tokenizer.bos_token_id:
            current_input_ids_cleaned = current_input_ids[:, 1:]
        else:
            current_input_ids_cleaned = current_input_ids
        
        # Construct the final prompt: [RETRIEVED_EPISODE] + [NEWLINE] + [CURRENT_USER_INPUT]
        modified_input_ids = torch.cat([full_retrieved_ids, newline_tensor, current_input_ids_cleaned], dim=1)
        modified_attention_mask = torch.ones_like(modified_input_ids)

        return modified_input_ids, modified_attention_mask