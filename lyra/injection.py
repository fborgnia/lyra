import sys
import torch
import torch.nn as nn

class MemoryInjectionLayer(nn.Module):
    """
    This layer retrieves a fully-formatted memory turn and concatenates it
    with the current turn to create a valid multi-turn prompt.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, current_input_ids, current_attention_mask):
        if not self.model.memory_graph:
            print("No memory to inject.", file=sys.stdout)
            return current_input_ids, current_attention_mask

        best_memory_index = self.model.gnn(self.model.memory_graph, current_input_ids)
        
        if best_memory_index == -1:
            return current_input_ids, current_attention_mask

        print(f"Injecting memory index {best_memory_index} as text.", file=sys.stdout)

        # 1. Retrieve the full, templated input_ids of the chosen memory turn
        retrieved_memory_ids = self.model.memory_graph[best_memory_index]["input_ids"].to(self.model.device)

        # 2. THE CRITICAL FIX: Remove the <bos> token from the CURRENT turn
        # The retrieved memory will provide the one and only <bos> token for the sequence.
        if current_input_ids[0, 0] == self.model.tokenizer.bos_token_id:
            current_input_ids_cleaned = current_input_ids[:, 1:]
        else:
            current_input_ids_cleaned = current_input_ids
        
        # 3. Concatenate the two turns to form a valid multi-turn prompt
        modified_input_ids = torch.cat([retrieved_memory_ids, current_input_ids_cleaned], dim=1)
        
        # 4. Create a new attention mask for the full sequence
        modified_attention_mask = torch.ones_like(modified_input_ids)

        return modified_input_ids, modified_attention_mask