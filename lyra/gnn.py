import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class EpisodicMemoryGNN(nn.Module):
    """
    A Graph Neural Network module responsible for finding the most relevant memory.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        embedding_dim = self.model.config.hidden_size
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, memory_graph, query_input_ids):
        """
        Finds the index of the most relevant memory node from the memory graph list.
        """
        # --- THE CRITICAL FIX ---
        # 1. Extract the summary vectors from the list of dictionaries.
        memory_vectors = [item["vector"] for item in memory_graph]

        # 2. If the list is empty, there's nothing to do.
        if not memory_vectors:
            return -1 # Return an invalid index

        # 3. Stack the list of vectors into a single tensor for processing.
        memory_vectors_tensor = torch.cat(memory_vectors, dim=0).to(self.model.device)

        # 4. Create the query vector from the current input_ids.
        with torch.no_grad():
            query_embedding = self.model.model.embed_tokens(query_input_ids)
            query_vector = torch.mean(query_embedding, dim=1)

        # 5. Project both the query and the memory nodes into the learned search space.
        projected_query = self.query_projection(query_vector)
        projected_memory_nodes = self.query_projection(memory_vectors_tensor)

        # 6. Calculate similarity scores (dot-product attention) in the projected space.
        attention_scores = torch.matmul(projected_query, projected_memory_nodes.t())
        print(f"GNN Attention Scores: {attention_scores.tolist()}", file=sys.stdout)

        # 7. Find the index of the memory with the highest score.
        best_memory_index = torch.argmax(attention_scores, dim=-1).item()
        print(f"GNN chose memory index: {best_memory_index}", file=sys.stdout)

        return best_memory_index