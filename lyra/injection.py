import torch
import torch.nn as nn
import sys

class MemoryInjectionLayer(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        # For now, this layer just adds the retrieved context.
        # Later, it could have its own parameters.

    def forward(self, hidden_states, memory_graph, query_vector):
        """
        Retrieves memory and injects it into the hidden states.
        """
        # 1. Use the GNN to retrieve the relevant memory context
        retrieved_memory = self.gnn(query_vector, memory_graph)
        
        print("Injecting retrieved memory into the model.", file=sys.stderr)
        
        # 2. Inject the memory by adding it to the prompt's hidden states.
        # We add the single context vector to every token's hidden state.
        # hidden_states: [batch, seq_len, hidden_dim]
        # retrieved_memory.unsqueeze(1): [batch, 1, hidden_dim]
        # Broadcasting handles the addition across the sequence length.
        modified_hidden_states = hidden_states + retrieved_memory.unsqueeze(1)
        
        return modified_hidden_states