import torch
from typing import Optional, Tuple

MemoryPackage = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]

class EpisodicMemoryStore:
    """A simple store for episodic memories."""
    def __init__(self):
        self.memories: list[MemoryPackage] = []

    def add(self, memory: torch.Tensor, attention_mask: Optional[torch.Tensor], index_vector: torch.Tensor):
        """Adds a new memory package to the store."""
        self.memories.append((memory, attention_mask, index_vector))

    def count(self) -> int:
        """Returns the number of memories in the store."""
        return len(self.memories)

    def get_first(self) -> Optional[MemoryPackage]:
        """Retrieves the first memory package, if it exists."""
        if not self.memories:
            return None
        return self.memories[0]

    def get_last(self) -> Optional[MemoryPackage]:
        """Retrieves the last memory package, if it exists."""
        if not self.memories:
            return None
        return self.memories[-1]

    def retrieve_all(self) -> list[MemoryPackage]:
        """Retrieves all memories from the store."""
        return self.memories
    
    def clear(self):
        """Resets the memory store to its initial empty state."""
        self.memories = []
