import torch.nn as nn
from .store import EpisodicMemoryStore
from .attention import MemoryCrossAttention

class MemoryInjectionBlock(nn.Module):
    """
    This block is responsible for retrieving memories and applying them
    to the hidden states.
    """
    def __init__(self, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store
        self.cross_attention = MemoryCrossAttention()

    def forward(self, hidden_states, **kwargs):
        memories_with_masks = self.memory_store.retrieve_all()
        if not memories_with_masks:
            return hidden_states

        # Placeholder logic
        print("I'm the memory injection block")
        print(f"  Retrieved {len(memories_with_masks)} memory packages.")
        
        # In the future, we will apply the memories to the hidden_states here
        return hidden_states
