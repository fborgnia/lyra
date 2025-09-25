import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpisodicBuffer(nn.Module):
    """
    A Graph Neural Network module responsible for finding the most relevant memory.
    """
    def __init__(self, config, embed_tokens_layer):
        super().__init__()
        # --- DECOUPLING FIX ---
        # Store only the specific components needed, not the whole model.
        self.config = config
        self.embed_tokens = embed_tokens_layer
        
        embedding_dim = self.config.hidden_size
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, memory_graph, query_input_ids):
        """
        Finds the index of the most relevant memory node from the memory graph list.
        """
        memory_vectors = [item["vector"] for item in memory_graph]

        if not memory_vectors:
            return -1

        # The device should be inferred from the parameters, not from the model object.
        device = self.query_projection.weight.device
        memory_vectors_tensor = torch.cat(memory_vectors, dim=0).to(device)

        # Create the query vector from the current input_ids.
        with torch.no_grad():
            query_embedding = self.embed_tokens(query_input_ids)
            query_vector = torch.mean(query_embedding, dim=1)

        # Project both the query and the memory nodes into the learned search space.
        projected_query = self.query_projection(query_vector)
        projected_memory_nodes = self.query_projection(memory_vectors_tensor)

        # Calculate similarity scores (dot-product attention) in the projected space.
        attention_scores = torch.matmul(projected_query, projected_memory_nodes.t())
        print(f"GNN Attention Scores: {attention_scores.tolist()}", file=sys.stdout)

        # Find the index of the memory with the highest score.
        best_memory_index = torch.argmax(attention_scores, dim=-1).item()
        print(f"GNN chose memory index: {best_memory_index}", file=sys.stdout)

        return best_memory_index