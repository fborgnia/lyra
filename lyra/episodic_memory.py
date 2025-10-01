import torch
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3Attention, Gemma3MLP
from typing import Optional, Tuple
from transformers.cache_utils import Cache


class EpisodicMemoryStore:
    """A simple store for episodic memories."""
    def __init__(self):
        self.memories = []

    def add(self, memory: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """Adds a new memory to the store."""
        self.memories.append((memory, attention_mask))

    def retrieve_all(self) -> list:
        """Retrieves all memories from the store."""
        return self.memories


class MemoryInjectionBlock(nn.Module):
    """
    A dummy memory injection block that prints a message.
    This block will be responsible for retrieving memories and applying them
    to the hidden states.
    """
    def __init__(self, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store

    def forward(self, hidden_states, **kwargs):
        memories_with_masks = self.memory_store.retrieve_all()
        if not memories_with_masks:
            return hidden_states

        print("I'm the memory injection block")
        print(f"  Retrieved {len(memories_with_masks)} memories.")
        # Unpack for future use
        memories, attention_masks = zip(*memories_with_masks)
        # In the future, we will apply the memories to the hidden_states here
        return hidden_states

class MemoryArchivalBlock(nn.Module):
    """
    A dummy memory archival block that prints a message.
    This block will be responsible for storing memories after a full
    generation pass.
    """
    def __init__(self, memory_store: EpisodicMemoryStore):
        super().__init__()
        self.memory_store = memory_store

    def forward(self, hidden_states, attention_mask):
        print("\n--- Memory Archival Block ---")
        if hidden_states is not None:
            print(f"  Hidden state shape: {hidden_states.shape}")
            print(f"  Hidden state dtype: {hidden_states.dtype}")
            print(f"  Hidden state mean: {hidden_states.mean().item():.4f}")
            print(f"  Hidden state std: {hidden_states.std().item():.4f}")
            # Detach the tensor from the computation graph before storing
            self.memory_store.add(
                hidden_states.detach(),
                attention_mask.detach() if attention_mask is not None else None
            )
        else:
            print("  Received None for hidden_states, not archiving.")

        if attention_mask is not None:
            print(f"  Attention mask shape: {attention_mask.shape}")
        else:
            print("  Received None for attention_mask.")
        print("-----------------------------\n")
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
        past_key_values: Optional[Cache] = None,
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
