import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerRetrieverHead(nn.Module):
    """
    Selects the most relevant memory for a given layer.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # A small network to generate a query vector from the hidden state
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def _create_summary_vector(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Creates a summary vector from a hidden state using masked averaging.
        Handles the case where the attention mask is longer than the hidden state
        during autoregressive generation.
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # During autoregressive generation, the hidden_state is for the current token (seq_len=1),
        # but the attention_mask is for the full sequence. We slice the mask.
        current_seq_len = hidden_state.shape[1]
        sliced_attention_mask = attention_mask[:, -current_seq_len:]

        expanded_mask = sliced_attention_mask.unsqueeze(-1).expand_as(hidden_state)
        masked_hs = hidden_state * expanded_mask
        summed_hs = masked_hs.sum(dim=1)
        num_real_tokens = expanded_mask.sum(dim=1)
        # Avoid division by zero
        num_real_tokens = torch.max(num_real_tokens, torch.ones_like(num_real_tokens))
        return summed_hs / num_real_tokens

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, memory_buffer: list):
        """
        Retrieves the most relevant memory's hidden state.
        """
        if not memory_buffer:
            return None, None

        # 1. Create query from current hidden state
        current_query = self._create_summary_vector(hidden_states, attention_mask)
        projected_query = self.query_proj(current_query)

        # 2. Find best memory
        best_score = -1
        best_memory = None
        
        with torch.no_grad():
            for memory in memory_buffer:
                memory_hs = memory['hidden_state'].to(hidden_states.device)
                memory_mask = memory['attention_mask'].to(hidden_states.device)
                
                memory_key = self._create_summary_vector(memory_hs, memory_mask)
                
                score = F.cosine_similarity(projected_query, memory_key)
                
                if score > best_score:
                    best_score = score
                    best_memory = memory

        return best_memory, projected_query


class LayerProjectionHead(nn.Module):
    """
    Projects a memory summary vector from the final layer's space
    to the current layer's representational space.
    """
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, memory_summary: torch.Tensor) -> torch.Tensor:
        return self.projection(memory_summary)


class GatedFusion(nn.Module):
    """
    Fuses a memory vector into the hidden state using a trainable gate.
    """
    def __init__(self, config):
        super().__init__()
        self.gate_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, memory_vector: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_linear(hidden_states))
        fused_states = (1 - gate) * hidden_states + gate * memory_vector
        return fused_states
