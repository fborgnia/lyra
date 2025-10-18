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
    output_attentions: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    This function will replace Gemma3DecoderLayer.forward.
    For the baseline, it simply calls the original forward method for this layer.
    """
    # print(f"[Injector] In forward for layer {self.layer_idx}") # Uncomment for debugging

    # Call the original method for this specific layer
    return self.original_decoder_layer_forward(
        hidden_states=hidden_states,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
