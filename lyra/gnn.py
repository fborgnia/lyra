# Phase 1: Standalone GNN Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import sys

class EpisodicMemoryGNN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # This is the trainable component. It's a standard linear layer that will
        # learn to project the query vector into a more effective search space.
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, memory_graph, query_vector):
        """
        Performs a single pass of attention over the memory graph.

        Args:
            memory_graph (torch_geometric.data.Data): The current memory graph.
            query_vector (torch.Tensor): The query vector from the current prompt.

        Returns:
            torch.Tensor: The context vector (retrieved memory).
        """
        memory_nodes = memory_graph.x
        
        if memory_nodes.shape[0] == 0:
            return torch.zeros_like(query_vector)

        # 1. Apply the trainable projection to the incoming query vector.
        projected_query = self.query_projection(query_vector)

        # 2. Calculate similarity scores using the *projected* query.
        attention_scores = torch.matmul(projected_query, memory_nodes.t())
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        print(f"GNN Attention Weights: {attention_weights.tolist()}", file=sys.stdout)

        # 3. Compute the context vector as a weighted sum of the original memories.
        context_vector = torch.matmul(attention_weights, memory_nodes)

        return context_vector