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

        best_memory_indices = retriever(memory_buffer, current_input_ids, top_k=top_k)
        
        if not best_memory_indices:
            return current_input_ids, current_attention_mask

        print(f"Injecting memory indices {best_memory_indices} as text.", file=sys.stdout)

        # Chronologically sort the indices to reconstruct the conversation order.
        # This has pros & cons, it ruins the ability to reconstruct a story, but makes it smarter remembering facts.
        # Gemma has little context, so lost in the middle is a real thing, after 4 turns gemma forgets the middle.
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
            conversation_history_parts.append(newline_tensor)

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