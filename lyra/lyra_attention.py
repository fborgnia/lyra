import torch
import torch.nn as nn
from typing import Optional, Callable
import os

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.get_logger(__name__)

# Copied from transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm
class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# Copied from transformers.models.gemma3.modeling_gemma3.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.gemma3.modeling_gemma3.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.gemma3.modeling_gemma3.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LyraGemma3Attention(nn.Module):
    """
    This is an identical copy of transformers.models.gemma3.modeling_gemma3.Gemma3Attention
    It serves as the baseline for implementing parallel attention heads.
    """
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        # --- Lyra: Load static context ---
        lyra_context_path = "data/test_kv_cache_with_p_m.pth"
        if os.path.exists(lyra_context_path):
            print(f"Layer {self.layer_idx}: Loading Lyra context from {lyra_context_path}...")
            loaded_data = torch.load(lyra_context_path, map_location="cuda", weights_only=False)

            lyra_k, lyra_v = loaded_data["kv_cache"][self.layer_idx]
            self.register_buffer("lyra_key_cache", lyra_k, persistent=False)
            self.register_buffer("lyra_value_cache", lyra_v, persistent=False)
            print(f"Layer {self.layer_idx}: Lyra context loaded successfully.")
            # --- DIAGNOSTICS ---
            is_global_str = "Global" if not self.is_sliding else "Sliding"
            print(f"-> DIAGNOSTIC (Layer {self.layer_idx} - {is_global_str}):")
            print(f"   - Loaded Lyra KV cache sequence length: {self.lyra_key_cache.shape[2]}")
            # --- END DIAGNOSTICS ---
        else:
            print(f"Warning: Lyra context file not found at {lyra_context_path}.")
            self.register_buffer("lyra_key_cache", torch.empty(0), persistent=False)
            self.register_buffer("lyra_value_cache", torch.empty(0), persistent=False)
        # --- End Lyra static context load ---
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def _eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_key_value_groups: int,  # Explicitly pass the group size for the stream
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scaling is None:
            scaling = self.head_dim**-0.5

        # Use the provided num_key_value_groups, not the module's global one
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.shape
        
        # 1. Project Q, K, V (Once)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply Normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # --- 2. Split the Query Heads ---
        # Key and Value states (with 1 head each) are shared by both streams.
        vanilla_q, lyra_q = torch.chunk(query_states, 2, dim=1)

        # --- 3. Process Vanilla Stream (Main Context) ---
        
        # Apply RoPE using dynamic position embeddings.
        # `key_states` is shared and will be rotated here for the vanilla stream.
        cos, sin = position_embeddings
        vanilla_q_rope, vanilla_k_rope = apply_rotary_pos_emb(vanilla_q, key_states, cos, sin)

        # Update the main KV cache with the rotated K and original V.
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # The `update` method returns the full cache including past keys
            vanilla_k_from_cache, vanilla_v_from_cache = past_key_values.update(vanilla_k_rope, value_states, self.layer_idx, cache_kwargs)
        else:
             # if no cache, the k states are the ones with rope applied
             vanilla_k_from_cache = vanilla_k_rope
             vanilla_v_from_cache = value_states

        # Determine attention implementation
        attention_interface: Callable = self._eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Calculate attention for the vanilla stream.
        stream_num_key_value_groups = self.num_attention_heads // 2 // self.num_key_value_heads
        vanilla_attn_output, vanilla_attn_weights = attention_interface(
            query=vanilla_q_rope,
            key=vanilla_k_from_cache,
            value=vanilla_v_from_cache,
            attention_mask=attention_mask,
            num_key_value_groups=stream_num_key_value_groups, # Pass the correct group size (2)
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        # --- 4. Process Lyra Stream (Static Context) ---

        # Apply RoPE to the new query and key states for the Lyra stream.
        # `cos` and `sin` are for the new tokens only.
        lyra_q_rope, lyra_k_rope = apply_rotary_pos_emb(lyra_q, key_states, cos, sin)

        # Combine the pre-rotated static cache with the newly rotated K/V for the current tokens.
        lyra_k_cache_expanded = self.lyra_key_cache.expand(bsz, -1, -1, -1)
        lyra_v_cache_expanded = self.lyra_value_cache.expand(bsz, -1, -1, -1)
        
        lyra_k_full = torch.cat([lyra_k_cache_expanded, lyra_k_rope], dim=2)
        lyra_v_full = torch.cat([lyra_v_cache_expanded, value_states], dim=2)

        # Dynamically create the correct causal mask for the Lyra stream.
        # This mask must allow new queries to see the full cache + causally see other new tokens.
        lyra_query_length = lyra_q_rope.shape[-2]
        lyra_key_length = lyra_k_full.shape[-2]
        cache_len = self.lyra_key_cache.shape[2]
        
        # Create a mask that is offset by the length of the static cache.
        # This ensures new tokens can see the whole cache, but can only causally see other new tokens.
        lyra_causal_mask = torch.zeros(
            (lyra_query_length, lyra_key_length), dtype=torch.bool, device=hidden_states.device
        )
        row_indices = torch.arange(lyra_query_length, device=hidden_states.device).unsqueeze(1)
        col_indices = torch.arange(lyra_key_length, device=hidden_states.device)
        lyra_causal_mask[row_indices < col_indices - cache_len] = True
        lyra_causal_mask = lyra_causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)

        # Calculate attention for the Lyra stream.
        lyra_attn_output, _ = attention_interface(
            query=lyra_q_rope,
            key=lyra_k_full,
            value=lyra_v_full,
            attention_mask=lyra_causal_mask,
            num_key_value_groups=stream_num_key_value_groups,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        # --- 5. Combine and Project ---
        
        # Concatenate the outputs of the two streams along the head dimension
        combined_attn_output = torch.cat([vanilla_attn_output, lyra_attn_output], dim=1)
        
        # Reshape and apply the final output projection
        combined_attn_output = combined_attn_output.transpose(1, 2).contiguous()
        combined_attn_output = combined_attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(combined_attn_output)

        # Return the final output and the attention weights from the vanilla stream for inspection
        return attn_output, vanilla_attn_weights