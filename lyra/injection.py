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

    def forward(self, retriever, memory_buffer, tokenizer, current_input_ids, current_attention_mask, top_k: int = 1):
        if not memory_buffer:
            print("No memory to inject.", file=sys.stdout)
            return current_input_ids, current_attention_mask

        # Implicitly treat the first memory as the instruction context.
        # It will always be injected.
        best_memory_indices = [0]
        
        # Only search for other relevant memories if there are more than one.
        if len(memory_buffer) > 1:
            # Search in the rest of the buffer (excluding the instruction).
            searchable_memories = memory_buffer[1:]
            
            # The retriever returns indices relative to the `searchable_memories` slice.
            relative_indices = retriever(searchable_memories, current_input_ids, top_k=top_k)
            
            # Adjust indices to be absolute (add 1) and append them.
            # This ensures we don't re-inject the instruction.
            absolute_indices = [i + 1 for i in relative_indices]
            best_memory_indices.extend(absolute_indices)

        print(f"Injecting memory indices {best_memory_indices} as text with top_k {top_k}.", file=sys.stdout)

        # --- 2. Reconstruct the input_ids with injected memory ---
        # Sort the indices to maintain chronological order in the conversation.    
        best_memory_indices.sort()

        device = current_input_ids.device
        
        # --- RECONSTRUCT CONVERSATIONAL HISTORY ---
        # Start with the BOS token.
        bos_token_id = tokenizer.bos_token_id
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        
        # List to hold all parts of the conversation history
        conversation_history_parts = [bos_tensor]
        
        newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
        newline_tensor = torch.tensor([[newline_token_id]], device=device)

        for index in best_memory_indices:
            retrieved_episode = memory_buffer[index]
            retrieved_user_ids = retrieved_episode["input_ids"].to(device)
            
            # Clean the <bos> token from the retrieved user input to avoid duplication
            if retrieved_user_ids[0, 0] == tokenizer.bos_token_id:
                retrieved_user_ids = retrieved_user_ids[:, 1:]

            conversation_history_parts.append(newline_tensor)
            conversation_history_parts.append(retrieved_user_ids)

            if "output_ids" in retrieved_episode:
                retrieved_model_ids = retrieved_episode["output_ids"].to(device)
                conversation_history_parts.append(retrieved_model_ids)

            # Add a newline after each full turn (user + model)
            #conversation_history_parts.append(newline_tensor)

        # Clean the <bos> token from the current user input
        if current_input_ids[0, 0] == tokenizer.bos_token_id:
            current_input_ids_cleaned = current_input_ids[:, 1:]
        else:
            current_input_ids_cleaned = current_input_ids
        
        conversation_history_parts.append(current_input_ids_cleaned)
        
        # Construct the final prompt by concatenating all parts
        modified_input_ids = torch.cat(conversation_history_parts, dim=1)
        modified_attention_mask = torch.ones_like(modified_input_ids)

        return modified_input_ids, modified_attention_mask