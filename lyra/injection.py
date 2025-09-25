import sys
import torch
import torch.nn as nn

class MemoryInjectionLayer(nn.Module):
    """
    This layer is now stateless. It receives all necessary components during its
    forward call to construct a multi-turn prompt.
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

        # 1. Retrieve the full, templated input_ids of the chosen memory turn
        device = current_input_ids.device
        retrieved_memory_ids = memory_buffer[best_memory_index]["input_ids"].to(device)

        # 2. Remove the <bos> token from the CURRENT turn
        if current_input_ids[0, 0] == tokenizer.bos_token_id:
            current_input_ids_cleaned = current_input_ids[:, 1:]
        else:
            current_input_ids_cleaned = current_input_ids
        
        # 3. Concatenate the two turns to form a valid multi-turn prompt
        modified_input_ids = torch.cat([retrieved_memory_ids, current_input_ids_cleaned], dim=1)
        
        # 4. Create a new attention mask for the full sequence
        modified_attention_mask = torch.ones_like(modified_input_ids)

        return modified_input_ids, modified_attention_mask