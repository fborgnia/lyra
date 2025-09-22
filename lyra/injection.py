import torch
import torch.nn as nn
import sys

class MemoryInjectionLayer(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        # For now, this layer just adds the retrieved context.
        # Later, it could have its own parameters.

    def forward(self, hidden_states, attention_mask, memory_graph, query_vector):
        # This line is a trick to disable memory injection for testing.
        #return hidden_states, attention_mask
        """
        Retrieves memory and injects it into the prompt embeddings by prepending it.
        """
        # 1. Use the GNN to retrieve the relevant memory context
        # retrieved_memory shape: [batch, hidden_dim]
        retrieved_memory = self.gnn(query_vector, memory_graph)
        
        print("Injecting retrieved memory into the model.", file=sys.stdout)
        
        # 2. Prepare memory for prepending. It needs a sequence length of 1.
        # The retrieved memory is a hidden state, but we need to treat it as an embedding.
        # Shape becomes: [batch, 1, hidden_dim]
        memory_to_prepend = retrieved_memory.unsqueeze(1)

        # 3. Prepend the memory to the original prompt embeddings.
        # hidden_states (here, embeddings) shape: [batch, seq_len, hidden_dim]
        # modified_embeds shape: [batch, 1 + seq_len, hidden_dim]
        #modified_embeds = torch.cat([memory_to_prepend], dim=1)
        modified_embeds = torch.cat([memory_to_prepend, hidden_states], dim=1)
        

        # 4. Create a new attention mask for the prepended memory.
        # It's a tensor of ones with shape [batch, 1].
        memory_attention_mask = torch.ones(
            (attention_mask.shape[0], 1), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )

        # 5. Prepend the memory's attention mask to the original attention mask.
        # final_attention_mask shape: [batch, 1 + seq_len]
        final_attention_mask = torch.cat([memory_attention_mask, attention_mask], dim=1)
        
        return modified_embeds, final_attention_mask