import torch.nn as nn
from .store import EpisodicMemoryStore, MemoryPackage
from .attention import MemoryCrossAttention
from typing import List

class MemoryInjectionBlock(nn.Module):
    """
    This block is responsible for retrieving memories and applying them
    to the hidden states.
    """
    def __init__(self, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store
        self.cross_attention = MemoryCrossAttention()

    def _select_memories(self) -> List[MemoryPackage]:
        """
        Selects memories based on the defined strategy.
        - Always selects the first memory (initial prompt).
        - Always selects the most recent memory.
        - Handles de-duplication if there is only one memory.
        """
        selected_memories = []
        num_memories = self.memory_store.count()

        if num_memories == 0:
            return selected_memories

        # Always retrieve the first memory
        first_memory = self.memory_store.get_first()
        if first_memory:
            selected_memories.append(first_memory)

        # Retrieve the last memory, but only if it's not the same as the first
        if num_memories > 1:
            last_memory = self.memory_store.get_last()
            if last_memory:
                selected_memories.append(last_memory)
        
        return selected_memories

    def forward(self, hidden_states, **kwargs):
        selected_memories = self._select_memories()

        if not selected_memories:
            return hidden_states

        print("\n--- Memory Injection Block ---")
        print(f"  - Selected {len(selected_memories)} memories for injection.")
        
        # In the future, we will apply the memories to the hidden_states here
        print("------------------------------\n")
        return hidden_states
