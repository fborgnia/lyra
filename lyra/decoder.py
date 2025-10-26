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
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    layer_idx = kwargs.get("layer_idx", -1)
    is_lyra_layer = "full_cross_attention" in self.attention_type
    #print(f" [DEBUG] DECODER LAYER {layer_idx} - ATTN Type: {self.attention_type} ")

    #if is_lyra_layer and past_key_values:
    #    print(f"  [DEBUG] DECODER L{layer_idx} - Received Lyra Cache ID: {id(past_key_values)}")
    #    pre_attn_len = past_key_values.layers[layer_idx].get_seq_length()

    
    hidden_states = self.input_layernorm(hidden_states)
    #print(f" [DEBUG] DECODER LAYER - USE CACHE - {use_cache} ")
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

    #hidden_states = residual + (hidden_states * 1.01)

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs
