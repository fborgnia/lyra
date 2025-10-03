import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from typing import Optional, Tuple

class MemoryCrossAttention(nn.Module):
    """
    A cross-attention module that allows a 'current' sequence of tokens to attend
    to a 'memory' sequence. It mirrors the architecture of Gemma3's self-attention
    block but uses two different sources for its projections.
    """
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projections for Key, Value (from memory state), and Output.
        # Query is projected externally and passed in.
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_states (`torch.Tensor`): The pre-projected hidden states of the current sequence (Query source).
            memory_states (`torch.Tensor`): The hidden states of the past memory (Key/Value source).
            memory_attention_mask (`torch.Tensor`, *optional*): The attention mask for the memory sequence.
        """
        batch_size, q_len, _ = query_states.size()
        _, kv_seq_len, _ = memory_states.size()

        # Project the memory inputs into K, V
        key_states = self.k_proj(memory_states)
        value_states = self.v_proj(memory_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V heads to match Q heads for Grouped-Query Attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Apply scaled dot-product attention
        # We do not apply RoPE, as we are crossing time-steps.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=memory_attention_mask, is_causal=False
        )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output
