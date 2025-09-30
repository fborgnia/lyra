import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3Attention, Gemma3MLP

from .memory_components import LayerRetrieverHead, LayerProjectionHead, GatedFusion


class LyraDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # Memory-specific components
        self.retriever_head = LayerRetrieverHead(config)
        self.projection_head = LayerProjectionHead(config)
        self.memory_norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gated_fusion = GatedFusion(config)
        self.similarity_threshold = 0.95 # Configurable

    def _create_summary_vector(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Creates a summary vector from a hidden state using masked averaging.
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
            
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_state)
        masked_hs = hidden_state * expanded_mask
        summed_hs = masked_hs.sum(dim=1)
        num_real_tokens = expanded_mask.sum(dim=1)
        num_real_tokens = torch.max(num_real_tokens, torch.ones_like(num_real_tokens))
        return summed_hs / num_real_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings_global: torch.FloatTensor = None,
        position_embeddings_local: torch.FloatTensor = None,
        memory_buffer: list = None,
        previous_layer_query: torch.Tensor = None,
        retriever_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> tuple:
        
        current_query = None
        # The state before the memory block, for the final residual connection.
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # MEMORY BLOCK - only runs if a retriever_attention_mask is provided
        if retriever_attention_mask is not None:
            retrieved_memory, current_query = self.retriever_head(hidden_states, retriever_attention_mask, memory_buffer)
            
            if retrieved_memory is not None:
                # Conditional skip logic
                should_skip = False
                if previous_layer_query is not None and current_query is not None:
                    similarity = F.cosine_similarity(current_query, previous_layer_query)
                    if similarity > self.similarity_threshold:
                        should_skip = True
                
                if not should_skip:
                    memory_hs = retrieved_memory['hidden_state'].to(hidden_states.device)
                    memory_mask = retrieved_memory['attention_mask'].to(hidden_states.device)

                    memory_summary = self._create_summary_vector(memory_hs, memory_mask)
                    projected_memory = self.projection_head(memory_summary)
                    normalized_memory = self.memory_norm(projected_memory)
                    
                    # The memory is fused into the main hidden_state path
                    hidden_states = self.gated_fusion(hidden_states, normalized_memory)

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # The attention layer returns attn_output and attn_weights.
        # The past_key_values object is updated in-place.
        layer_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings_global, # Lyra layers are global
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, current_query)

        if output_attentions:
            outputs += (layer_outputs[1],)

        return outputs
