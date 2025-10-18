import torch
import types
from typing import Optional

from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3DecoderLayer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings_global: torch.Tensor,
    position_embeddings_local: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
     # --- Lyra Arguments ---
    position_embeddings_lyra: Optional[torch.Tensor] = None,
    lyra_attention_mask: Optional[torch.Tensor] = None,
    lyra_past_key_values: Optional[Cache] = None,
    lyra_position_ids: Optional[torch.LongTensor] = None,
    lyra_cache_position: Optional[torch.LongTensor] = None,
    # --- End Lyra Arguments ---
    **kwargs,
) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # apply global RoPE to non-sliding layer only
    if self.self_attn.is_sliding:
        position_embeddings = position_embeddings_local
    else:
        # Highjack the global layer with lyra cache
        #position_embeddings = position_embeddings_global
        position_embeddings = position_embeddings_lyra
        past_key_values = lyra_past_key_values
        attention_mask = lyra_attention_mask
        position_ids = lyra_position_ids
        cache_position = lyra_cache_position

    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs
