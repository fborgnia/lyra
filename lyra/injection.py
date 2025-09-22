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
        return hidden_states, attention_mask