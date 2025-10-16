import torch
import torch.nn as nn
from typing import Optional, Callable
import os

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
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

# Copied from transformers.models.gemma3.modeling_gemma3.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

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
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        # --- Lyra: Load static context ---
        lyra_context_path = "data/test_kv_cache_with_p_m.pth"
        if os.path.exists(lyra_context_path):
            print(f"Layer {self.layer_idx}: Loading Lyra context from {lyra_context_path}...")
            # Load to cuda first to avoid device mismatches during model loading
            loaded_data = torch.load(lyra_context_path, map_location="cuda", weights_only=False)

            # The loaded kv_cache is a DynamicCache object (list of tuples)
            # We extract the key and value tensors for this specific layer
            lyra_k, lyra_v = loaded_data["kv_cache"][self.layer_idx]
            self.register_buffer("lyra_key_cache", lyra_k, persistent=False)
            self.register_buffer("lyra_value_cache", lyra_v, persistent=False)

            # The position embeddings are a tuple of (cos, sin)
            lyra_cos, lyra_sin = loaded_data["position_embeddings"]
            self.register_buffer("lyra_cos", lyra_cos, persistent=False)
            self.register_buffer("lyra_sin", lyra_sin, persistent=False)

            # The attention mask is a single tensor
            self.register_buffer("lyra_attention_mask", loaded_data["attention_mask"], persistent=False)
            print(f"Layer {self.layer_idx}: Lyra context loaded successfully.")
        else:
            print(f"Warning: Lyra context file not found at {lyra_context_path}. Lyra heads will not function.")
            # Register empty buffers so the module doesn't crash if the file is missing
            self.register_buffer("lyra_key_cache", torch.empty(0), persistent=False)
            self.register_buffer("lyra_value_cache", torch.empty(0), persistent=False)
            self.register_buffer("lyra_cos", torch.empty(0), persistent=False)
            self.register_buffer("lyra_sin", torch.empty(0), persistent=False)
            self.register_buffer("lyra_attention_mask", torch.empty(0), persistent=False)
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
        
        # 1. Project Q, K, V using original, full-size layers
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose for attention calculation
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. Apply Normalization to full tensors
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # 3. Split the tensors into two streams
        vanilla_q, lyra_q = torch.chunk(query_states, 2, dim=1)
        vanilla_k_states, lyra_k_states = torch.chunk(key_states, 2, dim=1)
        vanilla_v_states, lyra_v_states = torch.chunk(value_states, 2, dim=1)

        # --- 4. Process Vanilla Heads (Main Growing Context) ---
        
        # Apply RoPE using dynamic position embeddings
        cos, sin = position_embeddings
        vanilla_q, vanilla_k_states_with_rope = apply_rotary_pos_emb(vanilla_q, vanilla_k_states, cos, sin)

        # Update the main KV cache
        if past_key_values is not None:
            # The cache object expects the full K/V tensors. We only update the vanilla part.
            # We pass the vanilla K/V states with RoPE applied, and the original vanilla V states.
            # The Lyra part is filled with zeros as it's not part of the main cache.
            zero_k = torch.zeros_like(lyra_k_states)
            zero_v = torch.zeros_like(lyra_v_states)
            
            # We must apply RoPE to the vanilla_k_states before caching
            full_k_to_cache = torch.cat([vanilla_k_states_with_rope, zero_k], dim=1)
            full_v_to_cache = torch.cat([vanilla_v_states, zero_v], dim=1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # The `update` method returns the full cache including past keys
            full_k_from_cache, full_v_from_cache = past_key_values.update(full_k_to_cache, full_v_to_cache, self.layer_idx, cache_kwargs)
            
            # We only need the vanilla part from the full cache for this stream's attention
            vanilla_k_states, _ = torch.chunk(full_k_from_cache, 2, dim=1)
            vanilla_v_states, _ = torch.chunk(full_v_from_cache, 2, dim=1)
        else:
             # if no cache, the k states are the ones with rope applied
             vanilla_k_states = vanilla_k_states_with_rope

        # Determine attention implementation
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Calculate attention for the vanilla stream
        vanilla_attn_output, vanilla_attn_weights = attention_interface(
            self,
            vanilla_q,
            vanilla_k_states,
            vanilla_v_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        # --- 5. Process Lyra Heads (Static Context) ---

        # Apply RoPE using pre-loaded static position embeddings
        lyra_q, lyra_k_states = apply_rotary_pos_emb(lyra_q, lyra_k_states, self.lyra_cos, self.lyra_sin)

        # Combine static cache with current token's K/V
        # Note: We need to expand the batch dimension of the cache to match the input batch size
        lyra_k_cache_expanded = self.lyra_key_cache.expand(bsz, -1, -1, -1)
        lyra_v_cache_expanded = self.lyra_value_cache.expand(bsz, -1, -1, -1)
        lyra_k_states = torch.cat([lyra_k_cache_expanded, lyra_k_states], dim=2)
        lyra_v_states = torch.cat([lyra_v_cache_expanded, lyra_v_states], dim=2)

        # Calculate attention for the Lyra stream
        lyra_attn_output, _ = attention_interface(
            self,
            lyra_q,
            lyra_k_states,
            lyra_v_states,
            self.lyra_attention_mask, # Use the static mask
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        # --- 6. Combine and Project ---
        
        # Concatenate the outputs of the two streams along the head dimension
        combined_attn_output = torch.cat([vanilla_attn_output, lyra_attn_output], dim=1)
        
        # Reshape and apply the final output projection
        combined_attn_output = combined_attn_output.transpose(1, 2).contiguous()
        combined_attn_output = combined_attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(combined_attn_output)

        # Return the final output and the attention weights from the vanilla stream for inspection
        return attn_output, vanilla_attn_weights