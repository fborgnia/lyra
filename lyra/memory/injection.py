import torch
import torch.nn as nn
from .store import EpisodicMemoryStore, MemoryPackage
from .attention import MemoryCrossAttention
from typing import List

class MemoryInjectionBlock(nn.Module):
    """
    This block is responsible for retrieving memories and applying them
    to the hidden states.
    """
    def __init__(self, config, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store
        self.cross_attention = MemoryCrossAttention(config)
        # Each layer gets its own query projection to map its specific hidden state
        # into a space comparable with the memory keys.
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

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
            # Return a zero tensor to maintain a consistent return type
            return torch.zeros_like(hidden_states)

        print("\n--- Memory Injection Block ---")
        print(f"  - Selected {len(selected_memories)} memories for injection.")
        
        # Concatenate memory states and attention masks
        # The order is preserved from _select_memories, ensuring chronological composition
        memory_states = [mem[0] for mem in selected_memories]
        attention_masks = [mem[1] for mem in selected_memories]

        concatenated_memory_states = torch.cat(memory_states, dim=1)
        concatenated_attention_mask = torch.cat(attention_masks, dim=1)

        print(f"  - Concatenated memory states shape: {concatenated_memory_states.shape}")
        print(f"  - Concatenated attention mask shape: {concatenated_attention_mask.shape}")
        
        print("------------------------------\n")
        
        # Return a zero tensor as per the plan for this milestone
        return torch.zeros_like(hidden_states)
