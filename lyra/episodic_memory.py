import torch
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3Attention, Gemma3MLP
from typing import Optional, Tuple

class MemoryInjectionBlock(nn.Module):
    """
    A dummy memory injection block that prints a message.
    This block will be responsible for retrieving memories and applying them
    to the hidden states.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        print("I'm the memory injection block")
        return hidden_states

class MemoryArchivalBlock(nn.Module):
    """
    A dummy memory archival block that prints a message.
    This block will be responsible for storing memories after a full
    generation pass.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask):
        print("I'm the memory archival block")
        # In the future, this will store the hidden_states and attention_mask
        return

class LyraDecoderLayer(Gemma3DecoderLayer):
    """
    This class is a container for the modified `forward` method that includes
    the memory injection block. It is not intended to be instantiated directly
    but to be used for monkey-patching the `forward` method of existing
    Gemma3DecoderLayer instances.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.memory_injection_block(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

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

        if use_cache:
            outputs += (past_key_values,)

        return outputs
