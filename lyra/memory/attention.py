import torch.nn as nn

class MemoryCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # In the future, this will contain a nn.MultiheadAttention layer
    
    def forward(self, query, key, value):
        # Placeholder: does nothing but return the query
        print("    - MemoryCrossAttention placeholder called")
        return query
