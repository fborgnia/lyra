# Phase 1: Standalone GNN Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class EpisodicMemoryGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # In a more complex GNN, we would have layers here.
        # For now, it's a simple attention-based retrieval.
        pass

    def forward(self, query_vector, memory_graph):
        """
        Performs attention-based retrieval of memory nodes.
        
        Args:
            query_vector (Tensor): Shape [batch_size, hidden_dim]
            memory_graph (Data): The graph containing memory nodes.

        Returns:
            Tensor: The retrieved memory context vector, shape [batch_size, hidden_dim]
        """
        # Get all memory nodes from the graph
        memory_nodes = memory_graph.x
        
        # Calculate similarity scores (dot-product attention)
        # query_vector: [1, 1152], memory_nodes.t(): [1152, num_nodes]
        # scores: [1, num_nodes]
        scores = torch.matmul(query_vector, memory_nodes.t())
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        print(f"GNN Attention Weights: {attention_weights.detach().cpu().numpy()}", file=sys.stderr)
        
        # Create context vector as a weighted sum of memories
        # attention_weights: [1, num_nodes], memory_nodes: [num_nodes, 1152]
        # context_vector: [1, 1152]
        context_vector = torch.matmul(attention_weights, memory_nodes)
        
        return context_vector