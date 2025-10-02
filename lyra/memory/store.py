import torch
from typing import Optional

class EpisodicMemoryStore:
    """A simple store for episodic memories."""
    def __init__(self):
        self.memories = []

    def add(self, memory: torch.Tensor, attention_mask: Optional[torch.Tensor], index_vector: torch.Tensor):
        """Adds a new memory package to the store."""
        self.memories.append((memory, attention_mask, index_vector))

    def retrieve_all(self) -> list:
        """Retrieves all memories from the store."""
        return self.memories
