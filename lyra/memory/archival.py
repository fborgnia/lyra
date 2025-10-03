import torch
import torch.nn as nn
from typing import Optional
from .store import EpisodicMemoryStore

class MemoryArchivalBlock(nn.Module):
    """
    This block is responsible for processing the final hidden states of a model
    after a generation pass and archiving them into the memory store.
    """
    def __init__(self, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store

    def forward(self, hidden_states, attention_mask):
        #print("\n--- Memory Archival Block ---")
        if hidden_states is not None and attention_mask is not None:
            # 1. Create the index vector via masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden_states = hidden_states * mask_expanded
            summed_hidden_states = masked_hidden_states.sum(dim=1)
            # Get the number of actual tokens, avoiding division by zero
            num_tokens = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            index_vector = summed_hidden_states / num_tokens

            #print(f"  - Hidden state shape: {hidden_states.shape}")
            #print(f"  - Attention mask shape: {attention_mask.shape}")
            #print(f"  - Created index vector shape: {index_vector.shape}")

            # 2. Detach all tensors from the computation graph before storing
            detached_hs = hidden_states.detach()
            detached_mask = attention_mask.detach()
            detached_index = index_vector.detach()

            # 3. Add the complete memory package to the store
            self.memory_store.add(detached_hs, detached_mask, detached_index)
        else:
            print("  - Received None for hidden_states or attention_mask, not archiving.")

        #print("-----------------------------\n")
        return
